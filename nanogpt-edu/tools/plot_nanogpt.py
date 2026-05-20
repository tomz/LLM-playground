#!/usr/bin/env python3
"""Parse nanogpt-edu train.py stdout logs into plots.

Reads the lines printed by train.py:
    iter   <N> | loss <L> | lr <X> | <ms> ms/it
               eval | train <T> | val <V>

Emits <run>/loss.png + <run>/loss.svg + a sparkline to stdout.

Usage:
    python tools/plot_nanogpt.py out/tiny/train.log
    python tools/plot_nanogpt.py out/*/train.log --compare out/compare.png
"""
from __future__ import annotations
import argparse, os, re, sys
from typing import NamedTuple


_RE_ITER = re.compile(
    r"^iter\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\deE.+-]+)\s*\|\s*([\d.]+)\s*ms/it"
)
_RE_EVAL = re.compile(
    r"eval\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)"
)
_RE_PARAMS = re.compile(r"model:\s*([\d.]+)M params")


class Run(NamedTuple):
    name: str
    params_m: float
    iters: list[int]
    losses: list[float]
    lrs: list[float]
    ms_per_it: list[float]
    eval_iters: list[int]
    eval_train: list[float]
    eval_val: list[float]


def parse_log(path: str, name: str | None = None) -> Run:
    iters, losses, lrs, ms = [], [], [], []
    ev_iters, ev_train, ev_val = [], [], []
    params_m = 0.0
    last_iter = 0
    with open(path) as f:
        for line in f:
            m = _RE_PARAMS.search(line)
            if m:
                params_m = float(m.group(1)); continue
            m = _RE_ITER.search(line)
            if m:
                last_iter = int(m.group(1))
                iters.append(last_iter)
                losses.append(float(m.group(2)))
                lrs.append(float(m.group(3)))
                ms.append(float(m.group(4)))
                continue
            m = _RE_EVAL.search(line)
            if m:
                ev_iters.append(last_iter)
                ev_train.append(float(m.group(1)))
                ev_val.append(float(m.group(2)))
    if name is None:
        # use parent dir as run name (e.g. "out/tiny/train.log" → "tiny")
        name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return Run(name, params_m, iters, losses, lrs, ms, ev_iters, ev_train, ev_val)


def ascii_sparkline(values, width: int = 60, height: int = 10) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    bucket = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), bucket)][:width]
    rows = []
    for r in range(height, 0, -1):
        thr = lo + rng * (r - 0.5) / height
        rows.append("  " + "".join("█" if v >= thr else " " for v in sampled))
    rows.append(f"  hi={hi:.3f}  lo={lo:.3f}  n={len(values)}")
    return "\n".join(rows)


def plot_run(run: Run, out_dir: str) -> list[str]:
    written = []
    # SVG always (no deps)
    svg = _render_svg(run)
    sp = os.path.join(out_dir, "loss.svg")
    with open(sp, "w") as f:
        f.write(svg)
    written.append(sp)
    # PNG if matplotlib present
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return written
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(run.iters, run.losses, color="#0066cc", lw=0.9, alpha=0.6,
                 label="train (per-iter)")
    if run.eval_iters:
        axes[0].plot(run.eval_iters, run.eval_train, "o-", color="#003366",
                     lw=1.5, ms=5, label="train (eval)")
        axes[0].plot(run.eval_iters, run.eval_val, "s-", color="#cc3300",
                     lw=1.5, ms=5, label="val (eval)")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"nanogpt-edu / {run.name}  ({run.params_m:.2f}M params)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(run.iters, run.lrs, color="#339933", lw=1.2)
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("LR schedule (cosine + warmup)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(run.iters, run.ms_per_it, color="#996633", lw=0.9, alpha=0.7)
    axes[2].set_ylabel("ms / iter")
    axes[2].set_xlabel("iteration")
    axes[2].set_title("step time")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    pp = os.path.join(out_dir, "loss.png")
    fig.savefig(pp, dpi=120)
    plt.close(fig)
    written.append(pp)
    return written


def _render_svg(run: Run) -> str:
    W, H, M = 900, 360, 60
    if not run.iters:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"/>'
    xs = run.iters
    ys_train = run.losses
    xs0, xs1 = min(xs), max(xs) or 1
    ys = ys_train + run.eval_val
    ys0, ys1 = min(ys), max(ys)
    if ys1 == ys0: ys1 = ys0 + 1
    def sx(x): return M + (x - xs0) / (xs1 - xs0) * (W - 2 * M)
    def sy(y): return M + (H - 2 * M) - (y - ys0) / (ys1 - ys0) * (H - 2 * M)
    train_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys_train))
    val_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}"
                       for x, y in zip(run.eval_iters, run.eval_val))
    axes = (
        f'<line x1="{M}" y1="{M}" x2="{M}" y2="{H-M}" stroke="#888"/>'
        f'<line x1="{M}" y1="{H-M}" x2="{W-M}" y2="{H-M}" stroke="#888"/>'
    )
    ticks = []
    for i in range(5):
        yv = ys0 + (ys1 - ys0) * i / 4
        yp = sy(yv)
        ticks.append(
            f'<line x1="{M-4}" y1="{yp}" x2="{M}" y2="{yp}" stroke="#888"/>'
            f'<text x="{M-8}" y="{yp+4}" text-anchor="end" font-size="10" fill="#444">'
            f'{yv:.2f}</text>'
        )
        xv = xs0 + (xs1 - xs0) * i / 4
        xp = sx(xv)
        ticks.append(
            f'<line x1="{xp}" y1="{H-M}" x2="{xp}" y2="{H-M+4}" stroke="#888"/>'
            f'<text x="{xp}" y="{H-M+18}" text-anchor="middle" font-size="10" fill="#444">'
            f'{int(xv)}</text>'
        )
    legend = (
        f'<rect x="{W-200}" y="{M+10}" width="180" height="50" fill="white" stroke="#aaa"/>'
        f'<line x1="{W-190}" y1="{M+28}" x2="{W-160}" y2="{M+28}" stroke="#0066cc" stroke-width="2"/>'
        f'<text x="{W-155}" y="{M+32}" font-size="11" fill="#222">train loss</text>'
        f'<line x1="{W-190}" y1="{M+48}" x2="{W-160}" y2="{M+48}" stroke="#cc3300" stroke-width="2"/>'
        f'<text x="{W-155}" y="{M+52}" font-size="11" fill="#222">val (eval)</text>'
    )
    return (
        f'<?xml version="1.0"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif">'
        f'<rect width="{W}" height="{H}" fill="white"/>'
        f'<text x="{W//2}" y="28" text-anchor="middle" font-size="14" font-weight="bold" '
        f'fill="#111">nanogpt-edu / {run.name}  ({run.params_m:.2f}M params, '
        f'{len(xs)} log points)</text>'
        + axes + "".join(ticks)
        + f'<polyline fill="none" stroke="#0066cc" stroke-width="1.2" '
          f'opacity="0.7" points="{train_pts}"/>'
        + (f'<polyline fill="none" stroke="#cc3300" stroke-width="2" '
           f'points="{val_pts}"/>' if val_pts else "")
        + legend + "</svg>"
    )


def plot_compare(runs: list[Run], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib required for --compare", file=sys.stderr)
        return
    colors = ["#0066cc", "#cc6600", "#009933", "#cc0066", "#663399"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, r in enumerate(runs):
        c = colors[i % len(colors)]
        label = f"{r.name} ({r.params_m:.2f}M)"
        axes[0].plot(r.iters, r.losses, color=c, lw=0.8, alpha=0.55)
        if r.eval_iters:
            axes[0].plot(r.eval_iters, r.eval_val, "s-", color=c, lw=2, ms=5, label=label)
        else:
            axes[0].plot([], [], color=c, lw=2, label=label)
        # tokens-seen on x-axis for second plot — but we don't always have block/batch
        # so just plot loss-vs-iteration on log-x
        axes[1].plot(r.iters, r.losses, color=c, lw=0.8, alpha=0.55)
        if r.eval_iters:
            axes[1].plot(r.eval_iters, r.eval_val, "s-", color=c, lw=2, ms=5, label=label)
    axes[0].set_title("val loss (linear x)")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_title("val loss (log x)")
    axes[1].set_xlabel("iteration (log)"); axes[1].set_xscale("log")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=9)
    fig.suptitle("nanogpt-edu — model-size sweep", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="train.log files (one per run)")
    ap.add_argument("--compare", default=None, help="path to overlay PNG")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    runs = []
    for log in args.logs:
        r = parse_log(log)
        runs.append(r)
        out_dir = os.path.dirname(os.path.abspath(log))
        written = plot_run(r, out_dir)
        if not args.quiet:
            print(f"\n=== {r.name}  ({r.params_m:.2f}M, {len(r.iters)} log points, "
                  f"final loss {r.losses[-1] if r.losses else float('nan'):.3f}"
                  + (f", final val {r.eval_val[-1]:.3f}" if r.eval_val else "")
                  + ")")
            print(ascii_sparkline(r.losses))
            for p in written:
                print(f"  wrote -> {p}")

    if args.compare and runs:
        plot_compare(runs, args.compare)


if __name__ == "__main__":
    main()
