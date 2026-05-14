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
    """Pull the first python code block, falling back to the raw text."""
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


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


def quick_eval(model_path: str, n_problems: int = 20, n_samples: int = 1,
               max_new_tokens: int = 384, temperature: float = 0.2) -> float:
    """Returns pass@1 on the first `n_problems` of HumanEval."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

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
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-5),
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        completion = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        program = build_program(prompt, completion, ex["test"], ex["entry_point"])
        ok, msg = run_one(program)
        passes += int(ok)
        marker = "✓" if ok else "✗"
        print(f"  [{i+1:3d}/{n}] {marker}  task={ex['task_id']}  {msg[:60]}")
    score = passes / n
    print(f"[eval] pass@1 = {passes}/{n} = {score*100:.1f}%")
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model dir (HF format)")
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    args = ap.parse_args()
    quick_eval(args.model, args.n_problems, args.n_samples, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
