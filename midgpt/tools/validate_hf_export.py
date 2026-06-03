#!/usr/bin/env python3
"""Validate a midgpt HuggingFace export and print serving commands.

The fast path is dependency-light and suitable for CI: verify config, tokenizer
metadata, and weights exist. ``--bench`` optionally runs a tiny vLLM generation
smoke when vLLM is installed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def validate_export(path: str | Path) -> dict:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    cfg_path = root / "config.json"
    gen_path = root / "generation_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")
    if not gen_path.exists():
        raise FileNotFoundError(f"missing {gen_path}")
    weights = [p.name for p in (root / "model.safetensors", root / "pytorch_model.bin") if p.exists()]
    if not weights:
        raise FileNotFoundError(f"missing model.safetensors or pytorch_model.bin in {root}")
    cfg = json.loads(cfg_path.read_text())
    required = {"model_type", "vocab_size", "n_layer", "n_head", "n_embd", "n_positions"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"config.json missing required GPT-2 fields: {missing}")
    return {
        "path": str(root),
        "model_type": cfg["model_type"],
        "params_hint": {
            "layers": cfg["n_layer"],
            "heads": cfg["n_head"],
            "hidden": cfg["n_embd"],
            "ctx": cfg["n_positions"],
        },
        "weights": weights,
        "has_tokenizer": (root / "tokenizer.json").exists() or (root / "vocab.json").exists(),
        "vllm_command": f"python -m vllm.entrypoints.openai.api_server --model {root}",
    }


def run_vllm_smoke(path: str | Path, prompt: str, max_tokens: int) -> dict:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise SystemExit("--bench requires `pip install vllm`") from e
    t0 = time.time()
    llm = LLM(model=str(path))
    outputs = llm.generate([prompt], SamplingParams(max_tokens=max_tokens, temperature=0.0))
    dt = time.time() - t0
    text = outputs[0].outputs[0].text
    return {"seconds": dt, "chars": len(text), "text_preview": text[:120]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--bench", action="store_true", help="run an optional vLLM generation smoke")
    ap.add_argument("--prompt", default="The answer is")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()
    report = validate_export(args.export_dir)
    if args.bench:
        report["vllm_smoke"] = run_vllm_smoke(args.export_dir, args.prompt, args.max_tokens)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
