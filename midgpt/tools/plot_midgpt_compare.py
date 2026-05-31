#!/usr/bin/env python3
"""Overlay two midgpt runs from their log.jsonl files.

Built for the fused_ce showcase: prove the fused-linear-CE val/train curves
land ON TOP of the baseline (numerically exact) while annotating the
throughput / VRAM win that is fused_ce's actual point.

Usage::

    python tools/plot_midgpt_compare.py \
        --run  out/gpt2_350m_fweb_5060ti_fusedce/log.jsonl "fused-CE (Liger)" \
        --base out/gpt2_350m_fweb_5060ti/log.jsonl "baseline (dense CE)" \
        --out  out/gpt2_350m_fweb_5060ti_fusedce/compare_fusedce.png \
        --hardware "RTX 5060 Ti 16 GB (Blackwell sm_120, bf16)" \
        --dataset "FineWeb-Edu (1B-token slice)" \
        --note "peak VRAM 9.7 vs 12.8 GiB"

Reads the JSONL train.py logger rows: {"iter","loss","lr","ms","tok_per_s"}
for train rows and {"iter","eval_val"} for eval rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NamedTuple


class Series(NamedTuple):
    label: str
    iters: list[int]
    losses: list[float]
    ms: list[float]
    tok_s: list[float]
    eval_iters: list[int]
    eval_val: list[float]


def load_jsonl(path: str, label: str, max_iter: int | None = None) -> Series:
    it, lo, ms, tk, ei, ev = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            i = d.get("iter")
            if i is None:
                continue
            if max_iter is not None and i > max_iter:
                continue
            if "loss" in d:
                it.append(i); lo.append(d["loss"])
                ms.append(d.get("ms", float("nan")))
                tk.append(d.get("tok_per_s", float("nan")))
            if "eval_val" in d:
                ei.append(i); ev.append(d["eval_val"])
    return Series(label, it, lo, ms, tk, ei, ev)


# Okabe–Ito palette (matches plot_midgpt.py)
C_BASE = "#0072B2"   # blue
C_RUN = "#D55E00"    # vermillion
C_BASE_E = "#56B4E9"
C_RUN_E = "#E69F00"
GRID = "#DDDDDD"
SPINE = "#444444"
MUTED = "#666666"
MAIN = "#222222"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": SPINE, "axes.linewidth": 0.8,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlepad": 8,
        "axes.labelsize": 10.5, "axes.labelpad": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 10.5,
        "text.color": MAIN, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })
    return plt


def _ema(v, alpha=0.05):
    if not v:
        return []
    out = [v[0]]
    for x in v[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def _median(v):
    v = [x for x in v if x == x]
    return sorted(v)[len(v) // 2] if v else float("nan")


def _mean(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, metavar=("JSONL", "LABEL"), required=True)
    ap.add_argument("--base", nargs=2, metavar=("JSONL", "LABEL"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--note", default="", help="extra annotation, e.g. VRAM win")
    ap.add_argument("--title", default="midgpt · fused-linear-CE vs dense CE")
    args = ap.parse_args()

    run = load_jsonl(args.run[0], args.run[1])
    # clip baseline to the run's horizon so the overlay is apples-to-apples
    max_it = max(run.iters) if run.iters else None
    base = load_jsonl(args.base[0], args.base[1], max_iter=max_it)
    if not run.iters or not base.iters:
        print("no data parsed", file=sys.stderr)
        return 1

    plt = _mpl()
    fig = plt.figure(figsize=(10, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.32,
                          left=0.10, right=0.96, top=0.86, bottom=0.09)
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1], sharex=ax)

    # Loss overlay: EMA train + eval markers for both
    ax.plot(base.iters, _ema(base.losses), color=C_BASE, lw=1.9,
            label=f"{base.label} — train EMA")
    ax.plot(run.iters, _ema(run.losses), color=C_RUN, lw=1.9, ls=(0, (4, 2)),
            label=f"{run.label} — train EMA")
    if base.eval_iters:
        ax.plot(base.eval_iters, base.eval_val, "o-", color=C_BASE_E, lw=1.6,
                ms=6, mec="white", mew=1.0, label=f"{base.label} — val")
    if run.eval_iters:
        ax.plot(run.eval_iters, run.eval_val, "s--", color=C_RUN_E, lw=1.6,
                ms=6, mec="white", mew=1.0, label=f"{run.label} — val")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss overlay — curves coincide (fused-CE is numerically exact)")
    ax.legend(loc="upper right", handlelength=2.6)

    # max abs val-loss gap as a hard exactness number
    gap = ""
    if base.eval_iters and run.eval_iters:
        bmap = dict(zip(base.eval_iters, base.eval_val))
        diffs = [abs(v - bmap[i]) for i, v in zip(run.eval_iters, run.eval_val) if i in bmap]
        if diffs:
            gap = f"max |Δ val| = {max(diffs):.3f}"
            ax.text(0.015, 0.04, gap, transform=ax.transAxes, fontsize=9.5,
                    color=MUTED, style="italic")

    # Throughput panel
    axp.plot(base.iters, base.ms, color=C_BASE, lw=0.8, alpha=0.5)
    axp.plot(base.iters, _ema(base.ms), color=C_BASE, lw=1.7,
             label=f"{base.label}  ·  {_median(base.ms):.0f} ms/it")
    axp.plot(run.iters, run.ms, color=C_RUN, lw=0.8, alpha=0.5)
    axp.plot(run.iters, _ema(run.ms), color=C_RUN, lw=1.7, ls=(0, (4, 2)),
             label=f"{run.label}  ·  {_median(run.ms):.0f} ms/it")
    axp.set_ylabel("ms per iter")
    axp.set_xlabel("iteration")
    bt, rt = _mean(base.tok_s) / 1e3, _mean(run.tok_s) / 1e3
    axp.set_title(f"Step time  ·  mean throughput  {base.label} ≈ {bt:.1f}k  |  "
                  f"{run.label} ≈ {rt:.1f}k tok/s")
    axp.legend(loc="upper right")

    fig.suptitle(args.title, fontsize=14, fontweight="bold", y=0.955)
    sub = [f"baseline best val {min(base.eval_val):.3f}" if base.eval_val else "",
           f"fused best val {min(run.eval_val):.3f}" if run.eval_val else "",
           args.note]
    sub = [s for s in sub if s]
    if sub:
        fig.text(0.5, 0.905, "   ·   ".join(sub), ha="center", fontsize=10, color=MUTED)
    foot = [b for b in [args.hardware, args.dataset] if b]
    if foot:
        fig.text(0.5, 0.012, "   ·   ".join(foot), ha="center",
                 fontsize=8.5, color=MUTED, style="italic")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")
    if gap:
        print(f"  exactness: {gap}")
    print(f"  throughput: baseline {bt:.1f}k  vs  fused {rt:.1f}k tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
