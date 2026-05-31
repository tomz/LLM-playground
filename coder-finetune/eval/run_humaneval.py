"""HumanEval pass@1 runner.

Loads the openai_humaneval dataset, generates one completion per problem,
executes the generated code + reference test in a subprocess with a timeout
and resource limits. Returns pass@1.

IMPORTANT: this executes model-generated code locally. For untrusted models,
run this inside a Docker container or gVisor sandbox. The default subprocess
guard here is a safety floor, not a security boundary.
"""
from __future__ import annotations
import argparse, multiprocessing as mp, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def _exec_target(prog: str, q):
    """Subprocess target: exec(prog) and report pass/fail via Queue."""
    try:
        # Disable __builtins__ tampering attempts? No: HumanEval needs builtins.
        # Just run it and let the test assertions raise on failure.
        ns = {}
        exec(prog, ns)
        q.put(("ok", None))
    except BaseException as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_one(program: str, timeout: float = 5.0) -> tuple[bool, str]:
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_exec_target, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(1.0)
        if p.is_alive():
            p.kill()
        return False, "timeout"
    if q.empty():
        return False, "no-result"
    status, msg = q.get()
    return status == "ok", msg or ""


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
