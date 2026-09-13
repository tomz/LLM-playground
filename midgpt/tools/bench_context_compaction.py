#!/usr/bin/env python3
"""Measure full-history versus compacted-state prefill on real GPUs.

One process runs per arm so two GPUs can measure concurrently without sharing a
CUDA context.  This benchmark measures systems cost only; semantic fidelity is
covered by ``tests/test_context.py``.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def _arm(checkpoint: str, gpu: int, seq_len: int, repeats: int, queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import torch
    from model import GPT, GPTConfig

    device = torch.device("cuda:0")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["cfg"]
    model = GPT(GPTConfig(**cfg["model"])).to(device).eval()
    model.load_state_dict({key.replace("_orig_mod.", ""): value
                           for key, value in saved["model"].items()})
    if seq_len > model.cfg.block_size:
        raise ValueError(f"seq_len {seq_len} exceeds block_size {model.cfg.block_size}")
    ids = torch.randint(0, model.cfg.vocab_size, (1, seq_len), device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            model(ids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(repeats):
            model(ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    queue.put({
        "gpu": gpu,
        "seq_len": seq_len,
        "repeats": repeats,
        "mean_ms": elapsed * 1000 / repeats,
        "tokens_per_s": seq_len * repeats / elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--full-tokens", type=int, default=1024)
    parser.add_argument("--compact-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    jobs = [
        ctx.Process(target=_arm, args=(args.checkpoint, 0, args.full_tokens,
                                      args.repeats, queue)),
        ctx.Process(target=_arm, args=(args.checkpoint, 1, args.compact_tokens,
                                      args.repeats, queue)),
    ]
    for job in jobs:
        job.start()
    rows = [queue.get() for _ in jobs]
    for job in jobs:
        job.join()
        if job.exitcode:
            raise SystemExit(f"worker exited with {job.exitcode}")
    rows.sort(key=lambda row: row["seq_len"], reverse=True)
    full, compact = rows
    report = {
        "checkpoint": "/".join(Path(args.checkpoint).parts[-2:]),
        "full": full,
        "compacted": compact,
        "token_reduction": 1.0 - compact["seq_len"] / full["seq_len"],
        "latency_reduction": 1.0 - compact["mean_ms"] / full["mean_ms"],
        "methodology": "concurrent one-arm-per-GPU bf16 prefill; three warmups",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
