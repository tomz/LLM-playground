#!/usr/bin/env python3
"""Plot simulator output: loss curve, $/day cumulative, GPU failures.

Reads ``events.jsonl`` produced by ``scripts/simulate.py`` and emits
``loss.svg`` / ``loss.png`` (matplotlib if available, otherwise a
hand-rolled SVG) plus an ASCII sparkline straight to stdout. Always
works — never hard-fails on missing optional deps.

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


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_series(events: list[dict]) -> dict[str, list[Any]]:
    steps, losses, days, dollars_log = [], [], [], []
    cum_failures = []
    for e in events:
        if e["kind"] == "pretrain.log":
            steps.append(e["step"])
            losses.append(e["loss"])
            days.append(e["day"])
            dollars_log.append(e.get("dollars", 0.0))
            cum_failures.append(e.get("failures", 0))
    return {
        "step": steps, "loss": losses, "day": days,
        "dollars_per_log": dollars_log, "cum_failures": cum_failures,
    }


def ascii_sparkline(values: list[float], width: int = 60, height: int = 12) -> str:
    """Tiny ASCII line chart — no deps."""
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    rng = hi - lo or 1.0
    # Downsample to `width` buckets
    n = len(values)
    bucket = max(1, n // width)
    sampled = [values[i] for i in range(0, n, bucket)][:width]
    rows = []
    for r in range(height, 0, -1):
        thresh = lo + rng * (r - 0.5) / height
        line = "".join("█" if v >= thresh else " " for v in sampled)
        rows.append(f"  {line}")
    rows.append(f"  hi={hi:.3f}  lo={lo:.3f}  n={n}")
    return "\n".join(rows)


def write_svg(steps: list[int], losses: list[float], days: list[float],
              dollars_cum: list[float], out_path: str) -> None:
    """Hand-rolled 2-panel SVG so we don't need matplotlib."""
    W, H, M = 900, 540, 60
    panel_h = (H - 3 * M) // 2

    def _panel(x_vals, y_vals, top, color, label, y_fmt):
        if not x_vals:
            return ""
        xs0, xs1 = min(x_vals), max(x_vals)
        ys0, ys1 = min(y_vals), max(y_vals)
        if xs1 == xs0: xs1 = xs0 + 1
        if ys1 == ys0: ys1 = ys0 + 1
        def sx(x): return M + (x - xs0) / (xs1 - xs0) * (W - 2 * M)
        def sy(y): return top + panel_h - (y - ys0) / (ys1 - ys0) * panel_h
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(x_vals, y_vals))
        # axes
        axes = (
            f'<line x1="{M}" y1="{top}" x2="{M}" y2="{top+panel_h}" stroke="#888"/>'
            f'<line x1="{M}" y1="{top+panel_h}" x2="{W-M}" y2="{top+panel_h}" stroke="#888"/>'
        )
        # y ticks
        ticks = []
        for i in range(5):
            yv = ys0 + (ys1 - ys0) * i / 4
            yp = sy(yv)
            ticks.append(f'<line x1="{M-4}" y1="{yp}" x2="{M}" y2="{yp}" stroke="#888"/>'
                         f'<text x="{M-8}" y="{yp+4}" text-anchor="end" font-size="11" fill="#444">{y_fmt.format(yv)}</text>')
        # x ticks
        for i in range(5):
            xv = xs0 + (xs1 - xs0) * i / 4
            xp = sx(xv)
            ticks.append(f'<line x1="{xp}" y1="{top+panel_h}" x2="{xp}" y2="{top+panel_h+4}" stroke="#888"/>'
                         f'<text x="{xp}" y="{top+panel_h+18}" text-anchor="middle" font-size="11" fill="#444">{int(xv):,}</text>')
        return (
            axes + "".join(ticks) +
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
            f'<text x="{M}" y="{top-8}" font-size="13" fill="#222" font-weight="bold">{label}</text>'
        )

    body = []
    body.append(_panel(steps, losses, M, "#0066cc",
                       "training loss", "{:.2f}"))
    body.append(_panel(steps, dollars_cum, 2 * M + panel_h, "#cc6600",
                       "cumulative $ (pretrain GPU spend)", "${:,.0f}"))
    svg = (
        f'<?xml version="1.0"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif">'
        f'<rect width="{W}" height="{H}" fill="white"/>'
        f'<text x="{W//2}" y="24" text-anchor="middle" font-size="16" font-weight="bold" '
        f'fill="#111">frontier-platform simulation — {os.path.basename(os.path.dirname(out_path))}</text>'
        + "".join(body) + "</svg>"
    )
    with open(out_path, "w") as f:
        f.write(svg)


def write_png_with_matplotlib(steps, losses, days, dollars_cum, cum_failures,
                              out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(steps, losses, color="#0066cc", lw=1.5)
    axes[0].set_ylabel("loss"); axes[0].set_title("training loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(steps, dollars_cum, color="#cc6600", lw=1.5)
    axes[1].set_ylabel("$ cumulative"); axes[1].set_title("pretrain GPU spend")
    axes[1].grid(alpha=0.3)
    axes[2].step(steps, cum_failures, where="post", color="#990000", lw=1.5)
    axes[2].set_ylabel("# failures"); axes[2].set_title("cumulative GPU failures")
    axes[2].set_xlabel("step"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


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
        print("ERROR: no pretrain.log events found in events.jsonl", file=sys.stderr)
        sys.exit(1)

    # cumulative $ (dollars per log-window are charged in pretrain phase)
    cum = []
    running = 0.0
    for d in s["dollars_per_log"]:
        running += d
        cum.append(running)

    written = []
    if not args.no_svg:
        svg_path = os.path.join(args.run_dir, "loss.svg")
        write_svg(s["step"], s["loss"], s["day"], cum, svg_path)
        written.append(svg_path)
    if not args.no_png:
        png_path = os.path.join(args.run_dir, "loss.png")
        if write_png_with_matplotlib(s["step"], s["loss"], s["day"], cum,
                                     s["cum_failures"], png_path):
            written.append(png_path)
        elif not args.quiet:
            print(" (matplotlib not installed — skipping PNG)")

    if not args.quiet:
        print(f"\n LOSS CURVE  ({len(s['loss'])} points, "
              f"step 0 → {s['step'][-1]:,}, "
              f"loss {s['loss'][0]:.3f} → {s['loss'][-1]:.3f})")
        print(ascii_sparkline(s["loss"]))
        print()
        if cum and cum[-1] > 0:
            print(f" CUM $  (peak ${cum[-1]:,.0f})")
            print(ascii_sparkline(cum))
            print()
        for path in written:
            print(f" wrote -> {path}")


if __name__ == "__main__":
    main()
