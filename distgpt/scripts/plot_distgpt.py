#!/usr/bin/env python3
"""Plot loss / LR / step-time from a distgpt log.jsonl.

distgpt log format::

    {"loss": 11.02, "lr": 2e-06, "ms": 316.3, "tok_per_s": 103602.3, "step": 0, "wall_s": 3.163}
    {"eval_loss": 4.10, "step": 2800, "wall_s": 3745.956}

Emits a 3-panel PNG.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ema(vals: list[float], alpha: float = 0.05) -> list[float]:
    if not vals:
        return []
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to distgpt log.jsonl")
    ap.add_argument("--title", default="distgpt run")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <log_dir>/loss.png)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.log).read_text().splitlines()]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "eval_loss" in r]

    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    lrs = [r["lr"] for r in train]
    ms = [r["ms"] for r in train]
    tps = [r["tok_per_s"] for r in train]
    ev_steps = [r["step"] for r in evals]
    ev_losses = [r["eval_loss"] for r in evals]

    if not steps:
        print(f"no train rows in {args.log}", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.0, 1.0], hspace=0.4,
                          left=0.10, right=0.96, top=0.88, bottom=0.06)
    ax_loss = fig.add_subplot(gs[0])
    ax_lr = fig.add_subplot(gs[1], sharex=ax_loss)
    ax_ms = fig.add_subplot(gs[2], sharex=ax_loss)

    # Loss panel
    ax_loss.plot(steps, losses, color="#A9C9E8", lw=0.9, alpha=0.85,
                 label="train (per-step)")
    ax_loss.plot(steps, _ema(losses, 0.05), color="#0072B2", lw=1.8,
                 label="train (EMA α=0.05)")
    if ev_losses:
        ax_loss.plot(ev_steps, ev_losses, "o-", color="#D55E00",
                     lw=2.0, ms=6, mec="white", mew=1.0, label="validation")
        bi = min(range(len(ev_losses)), key=lambda i: ev_losses[i])
        bx, by = ev_steps[bi], ev_losses[bi]
        import math
        ppl = math.exp(by)
        ax_loss.scatter([bx], [by], s=160, marker="*", color="#222", zorder=6,
                        label=f"best val {by:.3f} (ppl {ppl:.0f})")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss curves")
    ax_loss.grid(alpha=0.3); ax_loss.legend(loc="upper right")
    if losses and max(losses) / max(min(losses), 1e-3) > 5:
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("cross-entropy loss (log)")

    # LR
    ax_lr.plot(steps, lrs, color="#117733", lw=1.6)
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("Learning-rate schedule (cosine + warmup)")
    ax_lr.grid(alpha=0.3)
    if max(lrs) > 0:
        ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    # Step time
    ax_ms.plot(steps, ms, color="#555555", lw=0.9, alpha=0.55, label="per-step")
    ax_ms.plot(steps, _ema(ms, 0.05), color="#222", lw=1.6, label="EMA")
    med = sorted(ms)[len(ms) // 2]
    mean_tok = sum(tps) / len(tps)
    ax_ms.axhline(med, color="#D55E00", lw=1.0, ls="--", alpha=0.7,
                  label=f"median {med:.0f} ms/step")
    ax_ms.set_ylabel("ms per step")
    ax_ms.set_xlabel("step")
    ax_ms.set_title(f"Step time   ·   mean throughput ≈ {mean_tok/1000:.1f}k tok/s")
    ax_ms.grid(alpha=0.3); ax_ms.legend(loc="upper right")

    fig.suptitle(args.title, fontsize=14, fontweight="bold", y=0.96)
    if args.subtitle:
        fig.text(0.5, 0.915, args.subtitle, ha="center", fontsize=10, color="#666")

    out = args.out or str(Path(args.log).parent / "loss.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
