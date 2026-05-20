#!/usr/bin/env python3
"""Parse nanogpt-edu train.py stdout logs into publication-quality plots.

Reads the lines printed by train.py:
    iter   <N> | loss <L> | lr <X> | <ms> ms/it
               eval | train <T> | val <V>

Emits <run>/loss.png + <run>/loss.svg (per run) and an optional cross-run
overlay. The PNG path uses matplotlib with a polished, color-blind-safe
palette suitable for blog posts and papers; the SVG path is a hand-rolled
zero-dependency fallback in the same style.

Usage:
    python tools/plot_nanogpt.py out/tiny/train.log
    python tools/plot_nanogpt.py out/*/train.log --compare out/compare.png \\
        --hardware "NVIDIA RTX 3050 (8 GB, bf16)" --dataset "Tiny Shakespeare (1 MB)"
"""
from __future__ import annotations
import argparse, os, re, sys
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Okabe–Ito 8-color palette: color-blind safe and good in print.
PALETTE = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish-green
    "#CC79A7",   # reddish-purple
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#F0E442",   # yellow
    "#000000",   # black
]

# Semantic colors used by single-run panels
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
    """Apply a clean, publication-friendly rcParams. Returns plt."""
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
        "axes.labelcolor": TEXT_MAIN,
        "axes.labelpad": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": TEXT_MAIN,
        "ytick.color": TEXT_MAIN,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
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
    """Standard EMA smoothing — for the noisy per-iter train loss."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# Per-run plot (PNG via matplotlib)
# ---------------------------------------------------------------------------

def plot_run(run: Run, out_dir: str,
             hardware: str = "", dataset: str = "") -> list[str]:
    written = []
    # SVG always works
    sp = os.path.join(out_dir, "loss.svg")
    with open(sp, "w") as f:
        f.write(_render_svg(run, hardware=hardware, dataset=dataset))
    written.append(sp)

    try:
        plt = _setup_mpl()
    except ModuleNotFoundError:
        return written

    fig = plt.figure(figsize=(10, 8.8))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[2.4, 1.0, 1.0], hspace=0.42,
        left=0.09, right=0.96, top=0.90, bottom=0.07,
    )
    ax_loss = fig.add_subplot(gs[0])
    ax_lr = fig.add_subplot(gs[1], sharex=ax_loss)
    ax_ms = fig.add_subplot(gs[2], sharex=ax_loss)

    # --- loss panel ---
    smooth = _ema(run.losses, alpha=0.05)
    ax_loss.plot(run.iters, run.losses, color=COLOR_TRAIN_RAW, lw=0.9,
                 alpha=0.85, label="train (per-iter)")
    ax_loss.plot(run.iters, smooth, color=COLOR_TRAIN_SMOOTH, lw=1.8,
                 label="train (EMA, α=0.05)")
    if run.eval_iters:
        ax_loss.plot(run.eval_iters, run.eval_val, "o-", color=COLOR_VAL,
                     lw=2.0, ms=6, mec="white", mew=1.0,
                     label="validation (eval)")
        # mark best val
        best_i = min(range(len(run.eval_val)), key=lambda i: run.eval_val[i])
        bx, by = run.eval_iters[best_i], run.eval_val[best_i]
        ax_loss.scatter([bx], [by], s=140, marker="*", color="#222",
                        zorder=6, label=f"best val: {by:.3f} @ iter {bx}")
        ax_loss.annotate(
            f"  best val {by:.3f}\n  @ iter {bx:,}",
            xy=(bx, by), xytext=(8, 12), textcoords="offset points",
            fontsize=9, color=TEXT_MUTED,
        )

    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss curves")
    ax_loss.legend(loc="upper right", ncol=1, handlelength=2.5)
    # Use log-y when there's a wide range (helps the early decay show)
    if run.losses and max(run.losses) / max(min(run.losses), 1e-3) > 25:
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("cross-entropy loss (log)")

    # --- LR panel ---
    ax_lr.plot(run.iters, run.lrs, color=COLOR_LR, lw=1.6)
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_title("Learning-rate schedule (cosine + warmup)")
    if max(run.lrs) > 0:
        ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))

    # --- ms/it panel ---
    ax_ms.plot(run.iters, run.ms_per_it, color=COLOR_MS, lw=0.9, alpha=0.55)
    ax_ms.plot(run.iters, _ema(run.ms_per_it, 0.05), color="#222",
               lw=1.6, label="EMA")
    med = sorted(run.ms_per_it)[len(run.ms_per_it) // 2]
    ax_ms.axhline(med, color=COLOR_VAL, lw=1.0, ls="--", alpha=0.7,
                  label=f"median {med:.1f} ms/it")
    ax_ms.set_ylabel("ms per iter")
    ax_ms.set_xlabel("iteration")
    ax_ms.set_title("Step time")
    ax_ms.legend(loc="upper right")

    # --- header + footer ---
    final_train = run.losses[-1] if run.losses else float("nan")
    best_val = min(run.eval_val) if run.eval_val else float("nan")
    final_val = run.eval_val[-1] if run.eval_val else float("nan")
    title = f"nanogpt-edu  •  {run.name}  •  {run.params_m:.2f} M parameters"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.975)
    sub_bits = [f"{len(run.iters):,} iterations logged",
                f"final train {final_train:.3f}",
                f"best val {best_val:.3f}",
                f"final val {final_val:.3f}"]
    fig.text(0.5, 0.937, "   ·   ".join(sub_bits),
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
# Hand-rolled SVG fallback (no matplotlib)
# ---------------------------------------------------------------------------

def _render_svg(run: Run, hardware: str = "", dataset: str = "") -> str:
    W, H = 960, 420
    L, R, T, B = 80, 40, 80, 70   # margins
    if not run.iters:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"/>'
    xs = run.iters
    ys_train = run.losses
    xs0, xs1 = min(xs), max(xs) or 1
    ys = ys_train + run.eval_val
    ys0, ys1 = min(ys), max(ys)
    if ys1 == ys0:
        ys1 = ys0 + 1
    pw, ph = W - L - R, H - T - B
    def sx(x): return L + (x - xs0) / (xs1 - xs0) * pw
    def sy(y): return T + ph - (y - ys0) / (ys1 - ys0) * ph

    smooth = _ema(ys_train, 0.05)
    train_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, smooth))
    raw_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys_train))
    val_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}"
                       for x, y in zip(run.eval_iters, run.eval_val))

    # grid + ticks
    grid, ticks = [], []
    for i in range(6):
        yv = ys0 + (ys1 - ys0) * i / 5
        yp = sy(yv)
        grid.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{W-R}" y2="{yp:.1f}" '
                    f'stroke="{GRID_COLOR}" stroke-width="0.7"/>')
        ticks.append(f'<text x="{L-8}" y="{yp+4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{TEXT_MAIN}">{yv:.2f}</text>')
    for i in range(6):
        xv = xs0 + (xs1 - xs0) * i / 5
        xp = sx(xv)
        grid.append(f'<line x1="{xp:.1f}" y1="{T}" x2="{xp:.1f}" y2="{T+ph}" '
                    f'stroke="{GRID_COLOR}" stroke-width="0.7"/>')
        ticks.append(f'<text x="{xp:.1f}" y="{T+ph+18}" text-anchor="middle" '
                     f'font-size="11" fill="{TEXT_MAIN}">{int(xv):,}</text>')

    # spines (left + bottom only)
    spines = (
        f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" '
        f'stroke="{SPINE_COLOR}" stroke-width="0.9"/>'
        f'<line x1="{L}" y1="{T+ph}" x2="{W-R}" y2="{T+ph}" '
        f'stroke="{SPINE_COLOR}" stroke-width="0.9"/>'
    )

    # legend (top-right, inside plot)
    lx, ly = W - R - 240, T + 12
    legend = (
        f'<g font-size="11" fill="{TEXT_MAIN}">'
        f'<rect x="{lx-8}" y="{ly-12}" width="240" height="62" '
        f'fill="white" fill-opacity="0.92" stroke="{GRID_COLOR}"/>'
        f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" '
        f'stroke="{COLOR_TRAIN_RAW}" stroke-width="1.2"/>'
        f'<text x="{lx+30}" y="{ly+4}">train (per-iter)</text>'
        f'<line x1="{lx}" y1="{ly+18}" x2="{lx+24}" y2="{ly+18}" '
        f'stroke="{COLOR_TRAIN_SMOOTH}" stroke-width="2.2"/>'
        f'<text x="{lx+30}" y="{ly+22}">train (EMA)</text>'
        f'<line x1="{lx}" y1="{ly+36}" x2="{lx+24}" y2="{ly+36}" '
        f'stroke="{COLOR_VAL}" stroke-width="2.2"/>'
        f'<circle cx="{lx+12}" cy="{ly+36}" r="3" fill="{COLOR_VAL}"/>'
        f'<text x="{lx+30}" y="{ly+40}">validation</text>'
        f'</g>'
    )

    # best-val marker
    star = ""
    if run.eval_val:
        bi = min(range(len(run.eval_val)), key=lambda i: run.eval_val[i])
        bx, by = sx(run.eval_iters[bi]), sy(run.eval_val[bi])
        star = (
            f'<polygon points="{bx:.1f},{by-7:.1f} {bx+2:.1f},{by-2:.1f} '
            f'{bx+7:.1f},{by-2:.1f} {bx+3:.1f},{by+1:.1f} '
            f'{bx+4:.1f},{by+6:.1f} {bx:.1f},{by+3:.1f} '
            f'{bx-4:.1f},{by+6:.1f} {bx-3:.1f},{by+1:.1f} '
            f'{bx-7:.1f},{by-2:.1f} {bx-2:.1f},{by-2:.1f}" '
            f'fill="#222" stroke="white" stroke-width="0.8"/>'
            f'<text x="{bx+10:.1f}" y="{by-6:.1f}" font-size="10" '
            f'fill="{TEXT_MUTED}">best val {run.eval_val[bi]:.3f}</text>'
        )

    # header
    final = run.losses[-1] if run.losses else float("nan")
    bestv = min(run.eval_val) if run.eval_val else float("nan")
    finalv = run.eval_val[-1] if run.eval_val else float("nan")
    header = (
        f'<text x="{L}" y="36" font-size="17" font-weight="bold" '
        f'fill="{TEXT_MAIN}">nanogpt-edu · {run.name} · {run.params_m:.2f}M params</text>'
        f'<text x="{L}" y="56" font-size="11" fill="{TEXT_MUTED}">'
        f'{len(xs):,} iterations · final train {final:.3f} · '
        f'best val {bestv:.3f} · final val {finalv:.3f}</text>'
    )
    # axis labels
    labels = (
        f'<text x="{L + pw/2:.1f}" y="{H-22}" text-anchor="middle" '
        f'font-size="12" fill="{TEXT_MAIN}">iteration</text>'
        f'<text x="22" y="{T+ph/2:.1f}" font-size="12" fill="{TEXT_MAIN}" '
        f'transform="rotate(-90 22,{T+ph/2:.1f})" text-anchor="middle">'
        f'cross-entropy loss</text>'
    )
    footer = ""
    bits = [b for b in [hardware, dataset] if b]
    if bits:
        footer = (
            f'<text x="{W/2}" y="{H-6}" text-anchor="middle" '
            f'font-size="10" font-style="italic" fill="{TEXT_MUTED}">'
            f'{" · ".join(bits)}</text>'
        )

    return (
        f'<?xml version="1.0"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">'
        f'<rect width="{W}" height="{H}" fill="white"/>'
        + header + "".join(grid) + spines
        + f'<polyline fill="none" stroke="{COLOR_TRAIN_RAW}" stroke-width="0.9" '
          f'opacity="0.85" points="{raw_pts}"/>'
        + f'<polyline fill="none" stroke="{COLOR_TRAIN_SMOOTH}" stroke-width="2" '
          f'points="{train_pts}"/>'
        + (f'<polyline fill="none" stroke="{COLOR_VAL}" stroke-width="2.2" '
           f'points="{val_pts}"/>' if val_pts else "")
        + star + "".join(ticks) + legend + labels + footer + "</svg>"
    )


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------

def plot_compare(runs: list[Run], out_path: str,
                 hardware: str = "", dataset: str = "") -> None:
    try:
        plt = _setup_mpl()
    except ModuleNotFoundError:
        print("matplotlib required for --compare", file=sys.stderr)
        return

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.22,
                          left=0.07, right=0.985, top=0.86, bottom=0.13)
    ax_lin = fig.add_subplot(gs[0])
    ax_log = fig.add_subplot(gs[1])

    for i, r in enumerate(runs):
        c = PALETTE[i % len(PALETTE)]
        label = f"{r.name} · {r.params_m:.2f}M"
        # train (faint, EMA) + val (bold)
        smooth = _ema(r.losses, 0.05)
        for ax in (ax_lin, ax_log):
            ax.plot(r.iters, smooth, color=c, lw=1.2, alpha=0.35)
            if r.eval_iters:
                ax.plot(r.eval_iters, r.eval_val, "o-", color=c,
                        lw=2.0, ms=5.5, mec="white", mew=0.9, label=label)
            else:
                ax.plot([], [], color=c, lw=2.0, label=label)
            if r.eval_val:
                bi = min(range(len(r.eval_val)), key=lambda i: r.eval_val[i])
                bx, by = r.eval_iters[bi], r.eval_val[bi]
                ax.scatter([bx], [by], s=80, marker="*", color=c,
                           edgecolor="#222", linewidth=0.8, zorder=6)

    for ax, title, xscale in (
        (ax_lin, "Validation loss (linear x)", "linear"),
        (ax_log, "Validation loss (log x)", "log"),
    ):
        ax.set_title(title)
        ax.set_xlabel("iteration" + (" (log)" if xscale == "log" else ""))
        ax.set_ylabel("cross-entropy loss")
        if xscale == "log":
            ax.set_xscale("log")
        ax.legend(loc="upper left" if xscale == "log" else "upper right",
                  ncol=1, handlelength=2.5)
        # Faint annotation: dotted "train" hint
        ax.plot([], [], color="#888", lw=1.2, alpha=0.35,
                label="(faint lines = train EMA)")

    fig.suptitle("nanogpt-edu — model-size sweep", fontsize=15,
                 fontweight="bold", y=0.965)
    sub = (f"{len(runs)} runs · "
           + " · ".join(f"{r.name} {r.params_m:.1f}M" for r in runs))
    fig.text(0.5, 0.925, sub, ha="center", fontsize=10.5, color=TEXT_MUTED)

    bits = [b for b in [hardware, dataset, "★ marks best val per run"] if b]
    if bits:
        fig.text(0.5, 0.015, "   ·   ".join(bits),
                 ha="center", fontsize=9, color=TEXT_MUTED, style="italic")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="train.log files (one per run)")
    ap.add_argument("--compare", default=None, help="path to overlay PNG")
    ap.add_argument("--hardware", default="",
                    help="hardware string for figure footer (e.g. 'RTX 3050 bf16')")
    ap.add_argument("--dataset", default="",
                    help="dataset string for figure footer")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    runs = []
    for log in args.logs:
        r = parse_log(log)
        runs.append(r)
        out_dir = os.path.dirname(os.path.abspath(log))
        written = plot_run(r, out_dir,
                           hardware=args.hardware, dataset=args.dataset)
        if not args.quiet:
            final = r.losses[-1] if r.losses else float("nan")
            fval = r.eval_val[-1] if r.eval_val else float("nan")
            print(f"\n=== {r.name}  ({r.params_m:.2f}M params, "
                  f"{len(r.iters):,} iters logged, final train {final:.3f}"
                  + (f", final val {fval:.3f}" if r.eval_val else "")
                  + ")")
            print(ascii_sparkline(r.losses))
            for p in written:
                print(f"  wrote -> {p}")

    if args.compare and runs:
        plot_compare(runs, args.compare,
                     hardware=args.hardware, dataset=args.dataset)


if __name__ == "__main__":
    main()
