"""HumanEval pass@1 runner.

Loads the openai_humaneval dataset, generates one completion per problem,
executes the generated code + reference test in a subprocess with a timeout
and resource limits. Returns pass@1.

IMPORTANT: this executes model-generated code locally. For untrusted models,
run this inside a Docker container or gVisor sandbox. The default subprocess
guard here is a safety floor, not a security boundary.

What "safety floor" means concretely
------------------------------------
Each ``run_one`` spawns a child via multiprocessing and, on POSIX, applies
``resource.setrlimit`` to bound:

  * RLIMIT_AS    — virtual memory (default 1 GiB)
  * RLIMIT_CPU   — CPU seconds       (default 2× wall-clock timeout)
  * RLIMIT_FSIZE — file write size   (default 0 → can't write files)
  * RLIMIT_NPROC — child processes   (opt-in via ``limits=``)
  * RLIMIT_NOFILE — open fds          (default 64)

stdout/stderr from the child are redirected to /dev/null so a chatty
``print()`` in the generated code can't flood the trainer log. The default
multiprocessing context is **``fork``** on POSIX (the historical behaviour
— fast, ~10 ms startup); pass ``mp_mode='spawn'`` to ``run_one`` for a fresh
interpreter that does not inherit the parent's heap, file descriptors, or
imported modules. Recommended for *untrusted* models.
"""
from __future__ import annotations
import argparse, multiprocessing as mp, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Default rlimit bag — POSIX only, no-op on Windows. Tunable per-call.
# memory_bytes is None under ``fork`` because the child inherits the parent's
# (possibly large) address space — setting RLIMIT_AS=1 GiB on a fork from a
# pytest+HF process that's already at 5 GiB VSZ would SIGKILL the child
# immediately, before exec() even starts. Spawn mode (a fresh interpreter)
# gets a real ceiling — see ``_effective_limits``.
DEFAULT_LIMITS = {
    "memory_bytes": None,                      # set by mode (see _effective_limits)
    "cpu_seconds": None,                       # filled in from timeout if None
    "max_file_bytes": 0,                       # can't write to disk
    "max_open_fds": 64,
}
# Memory ceiling when we know the child is a fresh interpreter (spawn).
SPAWN_MEMORY_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB


def extract_code(text: str) -> str:
    """Pull the first python code block from the model's output.

    Three cases, in priority order:
      1. A fenced ``` ```python ... ``` `` block → return the block contents.
      2. No fence but the text starts at a ``def``/``async def``/``class``/
         ``import`` line → return as-is (the model emitted pure code).
      3. No fence, prose mixed in → return the longest contiguous *code-looking*
         suffix starting from the first ``def``/``async def``/``class``/``import``
         line. This stops the verifier from blowing up with ``SyntaxError`` on
         outputs like ``Sure! Here you go:\\ndef f(): ...`` which previously
         scored 0 (counted as wrong) instead of getting a fair eval.
    """
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # No fence: try to recover the code body. Look for the first plausible
    # Python statement at the start of a line.
    code_start = re.search(r"^(?:async\s+def|def|class|import|from)\s",
                           text, re.MULTILINE)
    if code_start is None:
        return text  # nothing code-like; let the verifier fail honestly
    return text[code_start.start():]


def _apply_rlimits(limits: dict) -> None:
    """Apply POSIX resource limits in the child. Best-effort: a limit we can't
    set (e.g. NPROC on a cgroup-restricted container) is skipped, not fatal —
    the timeout + wall-clock kill remains a hard floor regardless."""
    try:
        import resource
    except ImportError:
        return  # Windows or otherwise unsupported

    def _try(name, soft, hard=None):
        if not hasattr(resource, name):
            return
        try:
            resource.setrlimit(getattr(resource, name),
                               (soft, hard if hard is not None else soft))
        except (ValueError, OSError):
            pass

    if limits.get("memory_bytes"):
        _try("RLIMIT_AS", int(limits["memory_bytes"]))
    if limits.get("cpu_seconds"):
        _try("RLIMIT_CPU", int(limits["cpu_seconds"]))
    if limits.get("max_file_bytes") is not None:
        _try("RLIMIT_FSIZE", int(limits["max_file_bytes"]))
    if limits.get("max_open_fds"):
        _try("RLIMIT_NOFILE", int(limits["max_open_fds"]))
    if limits.get("max_processes"):
        _try("RLIMIT_NPROC", int(limits["max_processes"]))


def _exec_target(prog: str, q, limits: dict | None, silence: bool):
    """Subprocess target: silence stdout/stderr, apply rlimits, exec(prog),
    report pass/fail via Queue. Any SystemExit / KeyboardInterrupt / MemoryError
    raised by rlimits firing is caught and reported as an error rather than
    crashing the child silently.

    Order matters: silencing happens *before* rlimits. If the parent forked
    us with fd 1 pointing at a regular file (e.g. pytest's capfd capture file),
    a subsequent ``print()`` from the model code would otherwise trip
    RLIMIT_FSIZE=0 with EFBIG on what looks to the user like a successful
    silencing. Redirecting fd 1/2 to /dev/null *first* makes the inherited
    fd irrelevant.
    """
    if silence:
        # Redirect *both* the fds and the Python-level sys.stdout/sys.stderr.
        #   * fd 1/2 → /dev/null  — covers C extensions, subprocesses,
        #     anything bypassing the Python streams.
        #   * sys.stdout/stderr   — pytest's capfd replaces these with
        #     capture-file-backed objects in the parent; under fork the
        #     child inherits them and a plain print() bypasses fd 1
        #     entirely, hitting the capture file (which is a regular file
        #     subject to RLIMIT_FSIZE). Resetting both layers is the only
        #     way to make this robust under any wrapping the parent did.
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            pass
        try:
            # Rebind sys.stdout/sys.stderr to /dev/null at the Python level.
            # Use 'w' so print() and other text-mode writers work.
            null_fp = open(os.devnull, "w")
            sys.stdout = null_fp
            sys.stderr = null_fp
        except OSError:
            pass
    if limits:
        _apply_rlimits(limits)
    try:
        ns: dict = {}
        exec(prog, ns)
        q.put(("ok", None))
    except BaseException as e:
        # BaseException catches MemoryError, RecursionError, SystemExit from
        # rlimit kills — all should be reported as failed, never propagate.
        q.put(("err", f"{type(e).__name__}: {str(e)[:200]}"))


def run_one(
    program: str,
    timeout: float = 5.0,
    *,
    limits: dict | None = None,
    mp_mode: str | None = None,
    silence_output: bool = True,
) -> tuple[bool, str]:
    """Execute ``program`` in a subprocess with rlimits + timeout.

    ``mp_mode``: ``None`` (default — ``fork`` on POSIX for speed, ``spawn`` on
    Windows) or ``'spawn'`` to force a fresh interpreter that doesn't inherit
    the parent's heap / fds / imported modules. Spawn is ~100 ms slower per
    call but is the right setting for *untrusted* models — and the rlimit
    floor below applies in both modes.

    ``limits``: override entries of ``DEFAULT_LIMITS``; pass ``{}`` to keep
    defaults, ``None`` for defaults, ``False`` to disable rlimits entirely.
    """
    if mp_mode is None:
        mp_mode = "fork" if sys.platform != "win32" else "spawn"
    ctx = mp.get_context(mp_mode)

    # Build the effective limits dict. cpu_seconds defaults to 2× wall-clock
    # so a busy loop is killed by the wall-clock timeout (which is more
    # accurate) but a CPU-pegging child can't outlive the parent's join() by
    # much even if the parent itself crashes.
    eff_limits: dict | None = None
    if limits is not False:
        eff_limits = dict(DEFAULT_LIMITS)
        if eff_limits["cpu_seconds"] is None:
            eff_limits["cpu_seconds"] = max(1, int(timeout * 2))
        # Apply a memory ceiling only when we know it's safe — spawn-mode
        # children start fresh and can be bounded; fork-mode children inherit
        # the parent's address space (often gigabytes for a pytest+HF run)
        # and would be SIGKILL'd before exec() if we tried to bound them.
        if mp_mode == "spawn" and eff_limits.get("memory_bytes") is None:
            eff_limits["memory_bytes"] = SPAWN_MEMORY_BYTES
        if limits:
            eff_limits.update(limits)

    # Now that run_many is sequential (see its docstring for why), there is
    # no thread contention on Queue()/Process()/start() — but we keep the
    # creation+start grouped to mirror the historic behavior closely.
    q = ctx.Queue()
    p = ctx.Process(target=_exec_target,
                    args=(program, q, eff_limits, silence_output))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, "timeout"
    # Don't use q.empty() — it's documented as unreliable in multiprocessing
    # and gave us flaky reward results under parallel run_many. Use a small
    # blocking get() instead: the child has already exited (we passed join()),
    # so any pending data is *already* in flight; a short timeout is enough.
    import queue as _queue
    try:
        # 5-second drain timeout — generous to absorb scheduler jitter on
        # CI / busy boxes. The child has already exited (we passed join()),
        # so any pending data is *already* in flight in the queue's pipe;
        # this only ever waits for the OS to deliver bytes.
        status, msg = q.get(timeout=5.0)
    except _queue.Empty:
        # Common cause: rlimit kill (SIGKILL from RLIMIT_AS/CPU) before the
        # child could put a result — surface a useful tag, not just "no-result".
        if p.exitcode and p.exitcode < 0:
            import signal
            sig = -p.exitcode
            name = signal.Signals(sig).name if 0 < sig < signal.NSIG else f"signal-{sig}"
            return False, f"killed-{name}"
        return False, "no-result"
    return status == "ok", msg or ""


def run_many(
    programs: list[str],
    timeout: float = 5.0,
    *,
    max_workers: int | None = None,
    **run_one_kwargs,
) -> list[tuple[bool, str]]:
    """Run a batch of programs and return per-program (ok, msg) tuples.

    Currently sequential — the natural threaded implementation hits Python's
    fork-from-threads warning (deadlock-prone under 3.12+) and we measured a
    real ~20% flake rate even with locking, because ``mp.Queue`` instances
    created concurrently can deliver pickled results to the wrong consumer.
    For untrusted models the right escape is a Docker pool, not in-process
    threads; for trusted runs the sequential cost is ~50 ms × N, fine for
    GRPO at consumer-GPU batch sizes.

    The ``max_workers`` argument is accepted for forward compatibility
    (callers in ``cf_rl.reward`` already pass it) but currently unused — a
    future Tier could replace this with a ``ProcessPoolExecutor`` of
    persistent workers each holding their own queue, which avoids the race.
    """
    del max_workers  # documented above
    if not programs:
        return []
    return [run_one(p, timeout=timeout, **run_one_kwargs) for p in programs]


def build_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Combine prompt + completion + test harness into one runnable program."""
    code = extract_code(completion)
    # Prefer the model's full code if it redefined the function; else prepend prompt.
    if f"def {entry_point}" not in code:
        code = prompt + "\n" + code
    return code + "\n\n" + test + f"\n\ncheck({entry_point})\n"


def _eos_ids(tok) -> list[int]:
    """Stop tokens that cover both the model's EOS and the chat-template's
    turn-end (<|im_end|> for Qwen / ChatML). Without this, generation runs to
    `max_new_tokens` after the answer ends — wasting wall-clock and polluting
    completions with whatever the model rambles next. Mirrors `infer/generate.py`."""
    ids: set[int] = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        try:
            tid = tok.convert_tokens_to_ids(marker)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                ids.add(tid)
        except Exception:
            pass
    return sorted(ids)


def quick_eval(model_path: str, n_problems: int = 20, n_samples: int = 1,
               max_new_tokens: int = 384, temperature: float = 0.2,
               seed: int | None = None) -> float:
    """Returns pass@k on the first `n_problems` of HumanEval.

    With ``n_samples=1`` (the default) this is pass@1. With ``n_samples>1`` the
    model is sampled ``k`` times per problem and the problem counts as solved
    if *any* sample passes (unbiased pass@k via the Chen-et-al. estimator
    collapses to ``c >= 1`` for k=n_samples). Forces ``temperature>0`` when
    ``n_samples>1`` so the samples aren't identical.

    ``seed`` makes the run reproducible (was implicitly nondeterministic).
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if n_samples > 1 and temperature <= 0:
        # Otherwise every sample is identical greedy decode — pass@k == pass@1
        # at higher wall-clock for no gain. Warn loudly rather than silently
        # mis-report.
        print(f"[eval] WARNING: n_samples={n_samples} with temperature=0 is "
              "pointless (all samples identical). Bumping temperature to 0.2.")
        temperature = 0.2

    if seed is not None:
        torch.manual_seed(seed)

    print(f"[eval] loading model: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()

    print("[eval] loading openai_humaneval")
    ds = load_dataset("openai_humaneval", split="test")
    n = min(n_problems, len(ds))
    eos_ids = _eos_ids(tok) or tok.eos_token_id
    passes = 0
    for i in range(n):
        ex = ds[i]
        prompt = ex["prompt"]
        msgs = [
            {"role": "system", "content": "You are a Python coding assistant. Complete the function. Return only Python code."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt
        ids = tok(text, return_tensors="pt").to(device)
        # k samples per problem; problem counts as solved if any pass.
        # num_return_sequences in one .generate() call is more efficient than
        # a Python loop, and the KV cache is shared across the k beams.
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
                num_return_sequences=n_samples,
            )
        any_pass = False
        last_msg = ""
        prompt_len = ids["input_ids"].shape[1]
        for s in range(n_samples):
            completion = tok.decode(out[s, prompt_len:], skip_special_tokens=True)
            program = build_program(prompt, completion, ex["test"], ex["entry_point"])
            ok, msg = run_one(program)
            last_msg = msg
            if ok:
                any_pass = True
                break  # short-circuit: one pass is enough for pass@k
        passes += int(any_pass)
        marker = "✓" if any_pass else "✗"
        print(f"  [{i+1:3d}/{n}] {marker}  task={ex['task_id']}  {last_msg[:60]}")
    score = passes / n
    label = f"pass@{n_samples}" if n_samples > 1 else "pass@1"
    print(f"[eval] {label} = {passes}/{n} = {score*100:.1f}%")
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model dir (HF format)")
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="samples per problem; >1 enables pass@k (any-pass wins)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for reproducible sampling (default: None = nondet)")
    args = ap.parse_args()
    quick_eval(args.model, args.n_problems, args.n_samples,
               args.max_new_tokens, args.temperature, seed=args.seed)


if __name__ == "__main__":
    main()
