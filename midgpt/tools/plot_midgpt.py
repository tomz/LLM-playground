#!/usr/bin/env python3
"""Parse midgpt train.py stdout into a publication-quality plot.

midgpt log format::

    world_size=1  device=mps  dtype=torch.bfloat16
    model: 123.59M non-emb params
    iter      0 | loss 10.9564 | lr 3.00e-05 | 230 ms | 4.5k tok/s
    iter     10 | loss 9.0669  | lr 3.30e-04 | 473 ms | 2.2k tok/s
               eval | val 7.3271 | ppl 1520.94
               saved ckpt -> out/<run>/ckpt.pt

Emits ``<run>/loss.png`` (matplotlib) + an ASCII sparkline on stdout.

Usage::

    python tools/plot_midgpt.py out/smoke_124m_train.log \
        --out-dir out/smoke_124m \
        --hardware "Apple M1 Pro (24 GB, MPS bf16)" \
        --dataset "WikiText-103 (~119M tokens)"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_RE_ITER = re.compile(
    r"^iter\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\deE.+-]+)"
    r"\s*\|\s*(\d+)\s*ms\s*\|\s*([\d.]+)k\s*tok/s"
)
_RE_EVAL = re.compile(r"eval\s*\|\s*val\s+([\d.]+)\s*\|\s*ppl\s+([\d.]+)")
_RE_PARAMS = re.compile(r"model:\s*([\d.]+)M")
_RE_HEAD = re.compile(r"world_size=\d+\s+device=(\S+)\s+dtype=(\S+)")


class Run(NamedTuple):
    name: str
    params_m: float
    device: str
    dtype: str
    iters: list[int]
    losses: list[float]
    lrs: list[float]
    ms_per_it: list[float]
    tok_per_s: list[float]
    eval_iters: list[int]
    eval_val: list[float]
    eval_ppl: list[float]


def parse_log(path: str, name: str | None = None) -> Run:
    iters: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    ms: list[float] = []
    tok_s: list[float] = []
    ev_iters: list[int] = []
    ev_val: list[float] = []
    ev_ppl: list[float] = []
    params_m = 0.0
    device = ""
    dtype = ""
    last_iter = 0
    with open(path) as f:
        for line in f:
            m = _RE_HEAD.search(line)
            if m:
                device, dtype = m.group(1), m.group(2)
                continue
            m = _RE_PARAMS.search(line)
            if m:
                params_m = float(m.group(1))
                continue
            m = _RE_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                iters.append(last_iter)
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                ms.append(float(m.group(4)))
                tok_s.append(float(m.group(5)) * 1000.0)
                continue
            m = _RE_EVAL.search(line)
            if m:
                ev_iters.append(last_iter)
                ev_val.append(float(m.group(1)))
                ev_ppl.append(float(m.group(2)))
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0].replace("_train", "")
    return Run(
        name, params_m, device, dtype,
        iters, losses, lrs, ms, tok_s,
        ev_iters, ev_val, ev_ppl,
    )


# ---------------------------------------------------------------------------
# Style (Okabe–Ito, same family as nanogpt-edu plotter)
# ---------------------------------------------------------------------------

COLOR_TRAIN_RAW = "#A9C9E8"
COLOR_TRAIN_SMOOTH = "#0072B2"
COLOR_VAL = "#D55E00"
COLOR_LR = "#117733"
COLOR_MS = "#555555"
GRID_COLOR = "#DDDDDD"
SPINE_COLOR = "#444444"
TEXT_MUTED = "#666666"
TEXT_MAIN = "#222222"


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlepad": 8,
        "axes.labelsize": 10.5,
        "axes.labelpad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.7,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "text.color": TEXT_MAIN,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
    })
    return plt


def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def ascii_sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return "(no data)"
    blocks = "▁▂▃▄▅▆▇█"
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in values)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_run(run: Run, out_dir: str, hardware: str, dataset: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    try:
        plt = _setup_mpl()
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping PNG", file=sys.stderr)
        return written

    fig = plt.figure(figsize=(10, 9.4))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[2.4, 1.0, 1.0], hspace=0.45,
        left=0.10, right=0.96, top=0.87, bottom=0.07,
    )
    ax_loss = fig.add_subplot(gs[0])
    ax_lr = fig.add_subplot(gs[1], sharex=ax_loss)
    ax_ms = fig.add_subplot(gs[2], sharex=ax_loss)

    # Loss panel: train (raw + EMA) + val (right axis: perplexity)
    smooth = _ema(run.losses, alpha=0.05)
    ax_loss.plot(run.iters, run.losses, color=COLOR_TRAIN_RAW, lw=0.9,
                 alpha=0.85, label="train (per-iter)")
    ax_loss.plot(run.iters, smooth, color=COLOR_TRAIN_SMOOTH, lw=1.8,
                 label="train (EMA, α=0.05)")
    if run.eval_iters:
        ax_loss.plot(run.eval_iters, run.eval_val, "o-", color=COLOR_VAL,
                     lw=2.0, ms=6, mec="white", mew=1.0,
                     label="validation")
        best_i = min(range(len(run.eval_val)), key=lambda i: run.eval_val[i])
        bx, by = run.eval_iters[best_i], run.eval_val[best_i]
        ax_loss.scatter([bx], [by], s=160, marker="*", color="#222", zorder=6,
                        label=f"best val {by:.3f} (ppl {run.eval_ppl[best_i]:.0f})")

    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss curves")
    ax_loss.legend(loc="upper right", handlelength=2.5)
    if run.losses and max(run.losses) / max(min(run.losses), 1e-3) > 25:
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("cross-entropy loss (log)")

    # LR panel
    ax_lr.plot(run.iters, run.lrs, color=COLOR_LR, lw=1.6)
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("Learning-rate schedule (cosine + warmup)")
    if max(run.lrs) > 0:
        ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    # Step-time panel
    ax_ms.plot(run.iters, run.ms_per_it, color=COLOR_MS, lw=0.9, alpha=0.55,
               label="per-iter")
    ax_ms.plot(run.iters, _ema(run.ms_per_it, 0.05), color="#222", lw=1.6,
               label="EMA")
    med = sorted(run.ms_per_it)[len(run.ms_per_it) // 2] if run.ms_per_it else 0
    ax_ms.axhline(med, color=COLOR_VAL, lw=1.0, ls="--", alpha=0.7,
                  label=f"median {med:.0f} ms/it")
    mean_tok = sum(run.tok_per_s) / max(1, len(run.tok_per_s))
    ax_ms.set_ylabel("ms per iter")
    ax_ms.set_xlabel("iteration")
    ax_ms.set_title(f"Step time   ·   mean throughput ≈ {mean_tok/1000:.1f}k tok/s")
    ax_ms.legend(loc="upper right")

    # Header / footer
    final_train = run.losses[-1] if run.losses else float("nan")
    best_val = min(run.eval_val) if run.eval_val else float("nan")
    best_ppl = min(run.eval_ppl) if run.eval_ppl else float("nan")
    title = f"midgpt  •  {run.name}  •  {run.params_m:.2f} M parameters"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.96)
    sub_bits = [
        f"{len(run.iters):,} iters logged",
        f"final train {final_train:.3f}",
        f"best val {best_val:.3f}",
        f"best ppl {best_ppl:.0f}",
    ]
    fig.text(0.5, 0.91, "   ·   ".join(sub_bits),
             ha="center", fontsize=10, color=TEXT_MUTED)

    footer_bits = [b for b in [hardware, dataset] if b]
    if footer_bits:
        fig.text(0.5, 0.012, "   ·   ".join(footer_bits),
                 ha="center", fontsize=8.5, color=TEXT_MUTED, style="italic")

    pp = os.path.join(out_dir, "loss.png")
    fig.savefig(pp)
    plt.close(fig)
    written.append(pp)
    return written


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to train.py stdout log")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: out/<runname>)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--dataset", default="")
    args = ap.parse_args()

    run = parse_log(args.log, name=args.name)
    if not run.iters:
        print(f"no iter lines parsed from {args.log}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or os.path.join("out", run.name)
    written = plot_run(run, out_dir, args.hardware, args.dataset)

    print(f"midgpt  ·  run={run.name}  ·  {run.params_m:.2f} M params"
          f"  ·  device={run.device}  ·  dtype={run.dtype}")
    print(f"  iters {len(run.iters):,}   final-train {run.losses[-1]:.3f}", end="")
    if run.eval_val:
        print(f"   best-val {min(run.eval_val):.3f}"
              f"   best-ppl {min(run.eval_ppl):.0f}")
    else:
        print()
    print(f"  loss : {ascii_sparkline(run.losses)}")
    if run.ms_per_it:
        med = sorted(run.ms_per_it)[len(run.ms_per_it) // 2]
        mean_tok = sum(run.tok_per_s) / len(run.tok_per_s)
        print(f"  perf : median {med:.0f} ms/it  ·  mean {mean_tok/1000:.1f}k tok/s")
    for p in written:
        print(f"  wrote: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
