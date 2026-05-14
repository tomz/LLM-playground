"""Lightweight in-cluster eval: held-out loss + perplexity over N batches.

For academic benchmarks (MMLU, HellaSwag, etc.) export the model to HF format
and pipe through `lm-evaluation-harness`.
"""
from __future__ import annotations
import math
import torch


@torch.no_grad()
def held_out_loss(model, loader, n_batches: int) -> dict:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = loader.next_batch()
        _, loss = model(x, y)
        losses.append(loss.float().item())
    model.train()
    L = sum(losses) / len(losses)
    return {"loss": L, "ppl": math.exp(L)}
