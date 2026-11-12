"""Render P0 roadmap figures from shipped code and measured receipts.

* ``lr_schedules.png``  — cosine vs WSD (linear, 1-sqrt) from ``utils``.
* ``doc_mask.png``      — attention mask from ``model.document_layout``.
* ``kv_cache_speed.png`` — from ``docs/figures/kv_cache_bench.json``.

Usage: python tools/plot_sota_p0.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import document_layout  # noqa: E402
from utils import cosine_lr, wsd_lr  # noqa: E402

OUT = ROOT / "docs" / "figures"


def schedules() -> None:
    W, T, LR, MIN = 200, 5000, 6e-4, 6e-5
    it = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(it, [cosine_lr(i, W, T, LR, MIN) for i in it], label="cosine (default)", lw=2)
    ax.plot(it, [wsd_lr(i, W, T, LR, MIN) for i in it], label="WSD, linear cooldown (20%)", lw=2)
    ax.plot(it, [wsd_lr(i, W, T, LR, MIN, shape="1-sqrt") for i in it], label="WSD, 1-sqrt cooldown (20%)", lw=2, ls="--")
    ax.axvspan(T - 0.2 * (T - W), T, color="0.9", zorder=0)
    ax.text(T - 0.1 * (T - W), LR * 0.55, "cooldown", ha="center", color="0.35")
    ax.set(xlabel="iteration", ylabel="learning rate", title="LR schedules (warmup 200, horizon 5000)")
    ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "lr_schedules.png", dpi=150); plt.close(fig)


def doc_mask() -> None:
    EOT = 0
    idx = torch.tensor([[5, 6, 7, EOT, 8, 9, EOT, 4, 3, 2, 1]])
    doc, pos = document_layout(idx, EOT)
    T = idx.size(1)
    causal = np.tril(np.ones((T, T), bool))
    masked = causal & (doc[0, :, None] == doc[0, None, :]).numpy()
    labels = [f"{'EOT' if t == EOT else t}\np{p}" for t, p in zip(idx[0].tolist(), pos[0].tolist())]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, m, title in zip(axes, (causal, masked), ("plain causal (default)", "doc_mask_eot (per-document)")):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1.4)
        ax.set_xticks(range(T), labels, fontsize=7); ax.set_yticks(range(T), labels, fontsize=7)
        ax.set(title=title, xlabel="key (token / RoPE position)", ylabel="query")
    fig.suptitle("Three packed documents: masking removes cross-document attention; positions restart", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "doc_mask.png", dpi=150); plt.close(fig)


def kv_cache() -> None:
    r = json.loads((OUT / "kv_cache_bench.json").read_text())
    rows = r["rows"]; x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(x - w / 2, [q["full_tok_s"] for q in rows], w, label="generate (full recompute)")
    ax.bar(x + w / 2, [q["cached_tok_s"] for q in rows], w, label="generate_cached (KV cache)")
    for i, q in enumerate(rows):
        ax.text(i + w / 2, q["cached_tok_s"] + 6, f"{q['speedup']:.2f}×", ha="center", fontsize=9)
    ax.set_xticks(x, [str(q["prompt_tokens"]) for q in rows])
    ax.set(xlabel="prompt tokens (+256 generated)", ylabel="decode tokens / s",
           title=f"midgpt {r['params_m']}M, {r['device']}, {r['dtype']}, batch {r['batch']}")
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(q["cached_tok_s"] for q in rows) * 1.25)
    fig.tight_layout(); fig.savefig(OUT / "kv_cache_speed.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    schedules(); doc_mask(); kv_cache()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
