"""Run `lm-evaluation-harness` against a distgpt checkpoint.

Usage from the CLI (`distgpt eval --lm-eval-tasks ...`):
  1. Build a distgpt GPT from the config + load the trained ckpt
  2. Export to a temporary HF-format directory
  3. Invoke `lm_eval.simple_evaluate` pointing at that directory
  4. Print the results table and return the raw dict

Why not just import lm_eval and use its API directly? Because lm-eval-harness
is a HEAVY optional dep (pulls transformers, datasets, accelerate, pandas,
…). We resolve it lazily so `distgpt eval` (in-cluster loss-only mode)
remains usable without it.

Tokenizer
---------
lm-eval-harness needs a tokenizer alongside the model dir. The user is
expected to drop their tokenizer files (tokenizer.json,
tokenizer_config.json, special_tokens_map.json) into the run's `tokenizer/`
subdir; this runner copies them into the exported HF dir. If they're
missing we fall back to GPT-2 BPE (which is what `tiktoken-gpt2` produces
when you preprocess with the supplied tools) but warn loudly.
"""
from __future__ import annotations
import json
import tempfile
import warnings
from pathlib import Path

import torch

from ..model.config import ModelConfig
from ..model.transformer import GPT
from .export_hf import export_to_hf


def _resolve_tokenizer_files(tok_dir: str | Path | None,
                              fallback_to_gpt2: bool = True) -> dict[str, str]:
    """Build the `tokenizer_files` map for `export_to_hf`.

    If `tok_dir` is given and contains the expected files, use them. If
    `tok_dir` is None or empty, fall back to GPT-2 BPE downloaded via
    transformers (when available). Returns {} on total failure so the
    export still goes through and lm-eval gets a clearer error.
    """
    if tok_dir is not None:
        tok_dir = Path(tok_dir)
        names = ["tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"]
        present = {n: str(tok_dir / n) for n in names if (tok_dir / n).exists()}
        if present:
            return present
        warnings.warn(
            f"tokenizer dir {tok_dir} has no tokenizer files; trying GPT-2 fallback"
        )
    if not fallback_to_gpt2:
        return {}
    # Lazy import — transformers is heavy.
    try:
        from transformers import GPT2TokenizerFast
        tmp = Path(tempfile.mkdtemp(prefix="distgpt_tok_"))
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.save_pretrained(str(tmp))
        return {n: str(tmp / n) for n in
                ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json") if (tmp / n).exists()}
    except Exception as e:
        warnings.warn(
            f"GPT-2 tokenizer fallback failed: {e}; lm-eval may not find a tokenizer"
        )
        return {}


def run_lm_eval(*, config_path: str | Path, ckpt_path: str | Path,
                tasks: list[str], tokenizer_dir: str | Path | None = None,
                num_fewshot: int = 0, limit: int | None = None,
                batch_size: int = 1, device: str = "cuda",
                output_path: str | Path | None = None) -> dict:
    """Export a distgpt checkpoint to HF format and run lm-eval-harness.

    Args:
      config_path: distgpt YAML config used to build the model.
      ckpt_path: trained checkpoint to load (a `.pt`/`.bin` state_dict
        or a directory written by `torch.save`).
      tasks: list of lm-eval task names (e.g. `["hellaswag", "arc_easy"]`).
      tokenizer_dir: dir with tokenizer.json etc. If None, falls back to
        GPT-2 BPE.
      num_fewshot, limit, batch_size: passed straight to lm-eval.
      device: where to run the eval-side forward passes.
      output_path: write `results.json` here for archival. None = skip.

    Returns:
      lm-eval-harness's results dict (`{"results": {...}, "configs": ...}`).

    Raises:
      ImportError if lm-eval-harness isn't installed (with the install
      command in the message).
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError(
            "lm-eval-harness is required for `distgpt eval --lm-eval-tasks`. "
            "Install with `pip install lm-eval` (https://github.com/EleutherAI/"
            "lm-evaluation-harness)."
        ) from e

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build model on CPU first, load weights, then move to the eval device.
    # CPU-first avoids OOM on the construction-allocator path for huge models.
    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg)
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Accept both raw state_dict and wrapped {"model": sd} forms.
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)

    # Export to a temp HF dir; lm-eval's HFLM loader takes it from there.
    tmp = Path(tempfile.mkdtemp(prefix="distgpt_hfexport_"))
    tok_files = _resolve_tokenizer_files(tokenizer_dir)
    export_to_hf(model, mcfg, tmp, tokenizer_files=tok_files)

    # Free the in-memory model before lm-eval reloads it on `device`.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lm = HFLM(pretrained=str(tmp), device=device, batch_size=batch_size,
                dtype="bfloat16")
    results = simple_evaluate(model=lm, tasks=tasks,
                                num_fewshot=num_fewshot, limit=limit)

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Pretty-print a short summary so the CLI is informative even without
    # the user piping output to a file.
    if "results" in results:
        print("\n=== lm-eval results ===")
        for task, metrics in results["results"].items():
            print(f"  {task}: {metrics}")

    return results
