#!/usr/bin/env python3
"""Plot simulator output: loss curve, $/day cumulative, GPU failures.

Reads ``events.jsonl`` produced by ``scripts/simulate.py`` and emits
``loss.svg`` / ``loss.png`` in a polished, publication-friendly style
(Okabe-Ito palette, no top/right spines, 200 DPI). Always works — never
hard-fails on missing optional deps; the SVG path is zero-dep.

Usage:
    python scripts/plot_sim.py out/sim/7b
    python scripts/plot_sim.py out/sim/7b --no-png
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Style (shared with plot_compare.py)
# ---------------------------------------------------------------------------

PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish-green
    "#CC79A7",  # reddish-purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
COLOR_LOSS = "#0072B2"
COLOR_DOLLARS = "#D55E00"
COLOR_FAILURES = "#CC3311"
GRID_COLOR = "#DDDDDD"
SPINE_COLOR = "#444444"
TEXT_MAIN = "#222222"
TEXT_MUTED = "#666666"


def setup_mpl():
    """Apply clean, publication-ready rcParams. Returns plt."""
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


def fmt_dollars(x: float, _pos=None) -> str:
    if x >= 1e9:  return f"${x/1e9:.1f}B"
    if x >= 1e6:  return f"${x/1e6:.1f}M"
    if x >= 1e3:  return f"${x/1e3:.0f}k"
    return f"${x:.0f}"


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

def load_events(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_series(events: list[dict]) -> dict[str, Any]:
    steps, losses, days, dollars_log = [], [], [], []
    cum_failures = []
    meta: dict[str, Any] = {}
    for e in events:
        k = e["kind"]
        if k == "program.start":
            meta["name"] = e.get("name", "")
            meta["n_params"] = e.get("n_params", 0)
            meta["total_tokens"] = e.get("total_tokens", 0)
            meta["pretrain_gpus"] = e.get("pretrain_gpus", 0)
            meta["pretrain_gpu_type"] = e.get("pretrain_gpu_type", "")
        elif k == "pretrain.start":
            meta["total_steps"] = e.get("total_steps", 0)
            meta["seconds_per_step"] = e.get("seconds_per_step", 0)
            meta["throughput_source"] = e.get("throughput_source", "modeled")
        elif k == "pretrain.spike":
            meta.setdefault("spikes", []).append({"step": e["step"], "loss": e["loss"]})
        elif k == "pretrain.log":
            steps.append(e["step"])
            losses.append(e["loss"])
            days.append(e["day"])
            dollars_log.append(e.get("dollars", 0.0))
            cum_failures.append(e.get("failures", 0))
        elif k == "pretrain.done":
            meta["final_loss"] = e.get("final_loss")
            meta["total_dollars"] = e.get("dollars")
    return {
        "step": steps, "loss": losses, "day": days,
        "dollars_per_log": dollars_log, "cum_failures": cum_failures,
        "meta": meta,
    }


def ascii_sparkline(values: list[float], width: int = 60, height: int = 10) -> str:
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
# Polished SVG fallback (zero deps)
# ---------------------------------------------------------------------------

def write_svg(steps, losses, dollars_cum, cum_failures, meta, out_path: str) -> None:
    W, H = 960, 760
    L, R = 80, 50
    TOP_HDR = 90      # space for title + subtitle
    BOT_AXIS = 35     # x-axis labels at the bottom of the last panel
    GAP = 60          # gap between panels
    panel_h = (H - TOP_HDR - BOT_AXIS - 2 * GAP) // 3
    pw = W - L - R

    def panel(top, x_vals, y_vals, color, title, y_fmt, fill=False):
        if not x_vals:
            return ""
        xs0, xs1 = min(x_vals), max(x_vals) or 1
        ys0 = min(y_vals + [0])
        ys1 = max(y_vals)
        if ys1 == ys0:
            ys1 = ys0 + 1
        def sx(x): return L + (x - xs0) / (xs1 - xs0) * pw
        def sy(y): return top + panel_h - (y - ys0) / (ys1 - ys0) * panel_h
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_vals, y_vals))
        # grid + ticks
        grid, ticks = [], []
        for i in range(5):
            yv = ys0 + (ys1 - ys0) * i / 4
            yp = sy(yv)
            grid.append(f'<line x1="{L}" y1="{yp:.1f}" x2="{W-R}" y2="{yp:.1f}" '
                        f'stroke="{GRID_COLOR}" stroke-width="0.7"/>')
            ticks.append(f'<text x="{L-8}" y="{yp+4:.1f}" text-anchor="end" '
                         f'font-size="10" fill="{TEXT_MAIN}">{y_fmt(yv)}</text>')
        for i in range(5):
            xv = xs0 + (xs1 - xs0) * i / 4
            xp = sx(xv)
            grid.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" '
                        f'y2="{top+panel_h}" stroke="{GRID_COLOR}" stroke-width="0.7"/>')
            ticks.append(f'<text x="{xp:.1f}" y="{top+panel_h+16}" text-anchor="middle" '
                         f'font-size="10" fill="{TEXT_MAIN}">{int(xv):,}</text>')
        spines = (
            f'<line x1="{L}" y1="{top}" x2="{L}" y2="{top+panel_h}" '
            f'stroke="{SPINE_COLOR}" stroke-width="0.9"/>'
            f'<line x1="{L}" y1="{top+panel_h}" x2="{W-R}" y2="{top+panel_h}" '
            f'stroke="{SPINE_COLOR}" stroke-width="0.9"/>'
        )
        fill_el = ""
        if fill:
            base_y = sy(ys0)
            fill_pts = (f"{L:.1f},{base_y:.1f} " + pts +
                        f" {W-R:.1f},{base_y:.1f}")
            fill_el = (f'<polygon points="{fill_pts}" fill="{color}" '
                       f'fill-opacity="0.10"/>')
        title_el = (
            f'<text x="{L}" y="{top-10}" font-size="12" font-weight="bold" '
            f'fill="{TEXT_MAIN}">{title}</text>'
        )
        return ("".join(grid) + spines + fill_el +
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{pts}"/>' + "".join(ticks) + title_el)

    name = meta.get("name", os.path.basename(os.path.dirname(out_path)))
    n_params = meta.get("n_params", 0)
    total_tokens = meta.get("total_tokens", 0)
    gpus = meta.get("pretrain_gpus", 0)
    gpu_type = meta.get("pretrain_gpu_type", "")
    src = meta.get("throughput_source", "modeled")

    header = (
        f'<text x="{L}" y="36" font-size="18" font-weight="bold" '
        f'fill="{TEXT_MAIN}">frontier-platform · {name}</text>'
        f'<text x="{L}" y="60" font-size="11.5" fill="{TEXT_MUTED}">'
        f'{n_params/1e9:.1f}B params · {total_tokens/1e12:.1f}T tokens · '
        f'{gpus:,}× {gpu_type} · throughput: {src}</text>'
    )

    body = [
        panel(TOP_HDR, steps, losses, COLOR_LOSS,
              "training loss", lambda v: f"{v:.2f}"),
        panel(TOP_HDR + panel_h + GAP, steps, dollars_cum, COLOR_DOLLARS,
              "cumulative pretrain GPU spend", fmt_dollars, fill=True),
        panel(TOP_HDR + 2 * (panel_h + GAP), steps, cum_failures, COLOR_FAILURES,
              "cumulative GPU/node failures", lambda v: f"{int(v):,}"),
    ]
    xlabel = (
        f'<text x="{L + pw/2:.1f}" y="{H-6}" text-anchor="middle" '
        f'font-size="11" fill="{TEXT_MUTED}">training step</text>'
    )
    svg = (
        f'<?xml version="1.0"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" '
        f'font-family="DejaVu Sans, sans-serif">'
        f'<rect width="{W}" height="{H}" fill="white"/>'
        + header + "".join(body) + xlabel + "</svg>"
    )
    with open(out_path, "w") as f:
        f.write(svg)


# ---------------------------------------------------------------------------
# Polished PNG (matplotlib)
# ---------------------------------------------------------------------------

def write_png(steps, losses, days, dollars_cum, cum_failures, meta,
              out_path: str) -> bool:
    try:
        plt = setup_mpl()
    except ModuleNotFoundError:
        return False
    import matplotlib.ticker as mtick

    fig = plt.figure(figsize=(10, 9.4))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.4, 1.0, 0.9], hspace=0.50,
        left=0.10, right=0.96, top=0.87, bottom=0.07,
    )
    ax_loss = fig.add_subplot(gs[0])
    ax_dol = fig.add_subplot(gs[1], sharex=ax_loss)
    ax_fail = fig.add_subplot(gs[2], sharex=ax_loss)

    # --- loss panel ---
    ax_loss.plot(steps, losses, color=COLOR_LOSS, lw=1.6, label="loss")
    # mark any spikes
    spikes = meta.get("spikes", [])
    if spikes:
        sx = [s["step"] for s in spikes]
        sy = [s["loss"] for s in spikes]
        ax_loss.scatter(sx, sy, marker="x", s=40, color="#CC3311",
                        linewidths=1.2, label=f"{len(spikes)} loss spikes")
        ax_loss.legend(loc="upper right")
    ax_loss.set_ylabel("training loss")
    ax_loss.set_title("Training loss (Chinchilla-fit curve + sampled spikes)")

    # --- cost panel ---
    ax_dol.plot(steps, dollars_cum, color=COLOR_DOLLARS, lw=1.8)
    ax_dol.fill_between(steps, 0, dollars_cum, color=COLOR_DOLLARS, alpha=0.12)
    ax_dol.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_dollars))
    ax_dol.set_ylabel("$ cumulative")
    final_dollars = dollars_cum[-1] if dollars_cum else 0
    ax_dol.set_title(
        f"Cumulative pretrain GPU spend  (final: {fmt_dollars(final_dollars)})"
    )

    # --- failures panel ---
    ax_fail.step(steps, cum_failures, where="post",
                 color=COLOR_FAILURES, lw=1.6)
    ax_fail.set_ylabel("# failures")
    ax_fail.set_xlabel("training step")
    final_fail = cum_failures[-1] if cum_failures else 0
    ax_fail.set_title(f"Cumulative GPU/node failures  (final: {final_fail:,})")

    # --- header ---
    name = meta.get("name", "?")
    n_params = meta.get("n_params", 0)
    total_tokens = meta.get("total_tokens", 0)
    gpus = meta.get("pretrain_gpus", 0)
    gpu_type = meta.get("pretrain_gpu_type", "")
    src = meta.get("throughput_source", "modeled")
    final_loss = losses[-1] if losses else float("nan")
    final_day = days[-1] if days else 0
    title = f"frontier-platform  •  {name}  •  {n_params/1e9:.1f}B params · {total_tokens/1e12:.1f}T tokens"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.965)
    sub_bits = [
        f"{gpus:,}× {gpu_type}",
        f"throughput: {src}",
        f"wall-clock: {final_day:.1f} d",
        f"final loss {final_loss:.3f}",
        f"total: {fmt_dollars(final_dollars)}",
    ]
    fig.text(0.5, 0.918, "   ·   ".join(sub_bits),
             ha="center", fontsize=10, color=TEXT_MUTED)

    fig.savefig(out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing events.jsonl")
    ap.add_argument("--no-png", action="store_true",
                    help="skip the matplotlib PNG output")
    ap.add_argument("--no-svg", action="store_true",
                    help="skip the pure-Python SVG output")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    events_path = os.path.join(args.run_dir, "events.jsonl")
    if not os.path.exists(events_path):
        print(f"ERROR: {events_path} not found", file=sys.stderr)
        sys.exit(1)
    events = load_events(events_path)
    s = extract_series(events)
    if not s["step"]:
        print("ERROR: no pretrain.log events found in events.jsonl",
              file=sys.stderr)
        sys.exit(1)

    cum, running = [], 0.0
    for d in s["dollars_per_log"]:
        running += d
        cum.append(running)

    written = []
    if not args.no_svg:
        sp = os.path.join(args.run_dir, "loss.svg")
        write_svg(s["step"], s["loss"], cum, s["cum_failures"], s["meta"], sp)
        written.append(sp)
    if not args.no_png:
        pp = os.path.join(args.run_dir, "loss.png")
        if write_png(s["step"], s["loss"], s["day"], cum,
                     s["cum_failures"], s["meta"], pp):
            written.append(pp)
        elif not args.quiet:
            print(" (matplotlib not installed — skipping PNG)")

    if not args.quiet:
        print(f"\n LOSS CURVE  ({len(s['loss'])} points, "
              f"step 0 → {s['step'][-1]:,}, "
              f"loss {s['loss'][0]:.3f} → {s['loss'][-1]:.3f})")
        print(ascii_sparkline(s["loss"]))
        print()
        if cum and cum[-1] > 0:
            print(f" CUM $  (peak {fmt_dollars(cum[-1])})")
            print(ascii_sparkline(cum))
            print()
        for path in written:
            print(f" wrote -> {path}")


if __name__ == "__main__":
    main()
