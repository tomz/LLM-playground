"""Run ``lm-evaluation-harness`` against a midgpt checkpoint.

Pipeline (mirrors ``distgpt/eval/lm_eval_runner.py``):

  1. Build a midgpt ``GPT`` from the ckpt's embedded config + load weights.
  2. Export to a temp HF-format directory via :mod:`export_hf` (writes
     ``GPT2LMHeadModel``-shaped config + safetensors).
  3. Invoke ``lm_eval.simple_evaluate`` pointing at that directory.
  4. Print the results table and write ``results.json`` if asked.

Why not use ``lm_eval``'s in-process API directly? Because lm-evaluation-harness
is a *heavy* optional dep (transformers + datasets + accelerate + pandas).
Resolving it lazily keeps the rest of midgpt installable on a 1.5 GB venv.

Usage::

    python lm_eval_runner.py \\
        --ckpt out/gpt2_350m_fweb_5060ti/ckpt_best.pt \\
        --tasks hellaswag,lambada_openai \\
        --num-fewshot 0 --device cuda --output results.json

For mid-training spot checks the existing ``eval.py`` HellaSwag harness is
faster (no harness install, no HF round-trip); use this when you want
side-by-side numbers against published GPT-2 / Pythia / GPT-J on the full
canonical task list.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path

import torch

from export_hf import export_to_hf
from model import GPT, GPTConfig


def run_lm_eval(*, ckpt_path: str | Path,
                tasks: list[str],
                num_fewshot: int = 0,
                limit: int | None = None,
                batch_size: int = 1,
                device: str = "cuda",
                dtype: str = "bfloat16",
                tokenizer: str = "gpt2",
                output_path: str | Path | None = None) -> dict:
    """Export a midgpt ckpt to HF format and run lm-eval-harness.

    Args:
      ckpt_path: trained checkpoint (``torch.save``-d dict containing
        ``model`` and ``cfg`` keys, as produced by ``train.py``).
      tasks: list of lm-eval task names, e.g. ``["hellaswag", "arc_easy"]``.
      num_fewshot, limit, batch_size, dtype: passed through to lm-eval.
      device: where to run the eval forward passes (``"cuda"``, ``"cpu"``,
        ``"mps"``).
      tokenizer: HF tokenizer name to bundle with the export (default
        ``"gpt2"`` — matches midgpt's only tokenizer).
      output_path: write ``results.json`` here for archival. None = skip.

    Returns:
      lm-eval-harness's results dict
      (``{"results": {...}, "configs": {...}, ...}``).

    Raises:
      ImportError if lm-eval-harness isn't installed.
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm-eval-harness is required. Install with "
            "`pip install lm-eval` (https://github.com/EleutherAI/"
            "lm-evaluation-harness)."
        ) from e

    # Build the model on CPU, load the trained weights, then hand off to the
    # export. CPU-first build means we can run this against a 1.5B ckpt even
    # on a machine without enough GPU VRAM to hold the optimizer state — we
    # only need *inference* VRAM, which lm-eval allocates itself.
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not (isinstance(sd, dict) and "cfg" in sd and "model" in sd):
        raise ValueError(
            f"checkpoint {ckpt_path} doesn't look like a midgpt ckpt "
            "(expected dict with 'cfg' and 'model' keys)"
        )
    mcfg = GPTConfig(**sd["cfg"]["model"])
    model = GPT(mcfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)

    # Export to a tmp HF dir; lm-eval's HFLM loader takes it from there.
    tmp = Path(tempfile.mkdtemp(prefix="midgpt_hfexport_"))
    export_to_hf(model, mcfg, tmp, tokenizer_name=tokenizer)

    # Free the in-memory midgpt model before lm-eval reloads it on `device`
    # (otherwise we pay 2× the weights in RAM during the handoff).
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lm = HFLM(pretrained=str(tmp), device=device,
              batch_size=batch_size, dtype=dtype)
    results = simple_evaluate(model=lm, tasks=tasks,
                              num_fewshot=num_fewshot, limit=limit)

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    if "results" in results:
        print("\n=== lm-eval results ===")
        for task, metrics in results["results"].items():
            print(f"  {task}: {metrics}")

    return results


def main():
    ap = argparse.ArgumentParser(
        description="Run lm-evaluation-harness against a midgpt checkpoint."
    )
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True,
                    help="Comma-separated lm-eval task names (e.g. "
                         "'hellaswag,lambada_openai,arc_easy')")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples per task (smoke runs).")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--output", default=None,
                    help="Write results.json here.")
    args = ap.parse_args()

    run_lm_eval(
        ckpt_path=args.ckpt,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        tokenizer=args.tokenizer,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
