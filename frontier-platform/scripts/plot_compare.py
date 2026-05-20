#!/usr/bin/env python3
"""Compare multiple simulator runs on one figure.

Reads each ``<run_dir>/events.jsonl``, overlays training loss + cumulative
GPU spend on a single PNG using the polished style shared with
``plot_sim.py``.

Usage:
    python scripts/plot_compare.py out/sim/1b out/sim/7b out/sim/70b out/sim/400b \\
        --out out/sim/compare_sizes.png \\
        --title "frontier-platform — size sweep"
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# reuse the shared style + formatters from plot_sim.py
from scripts.plot_sim import (  # noqa: E402
    setup_mpl, fmt_dollars, PALETTE, TEXT_MUTED,
)


def load_series(run_dir: str):
    path = os.path.join(run_dir, "events.jsonl")
    steps, losses, dollars = [], [], []
    cum = 0.0
    label = os.path.basename(run_dir.rstrip("/"))
    extra: dict = {}
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "program.start":
                extra["n_params"] = e["n_params"]
                extra["total_tokens"] = e["total_tokens"]
                extra["gpus"] = e["pretrain_gpus"]
                extra["gpu_type"] = e["pretrain_gpu_type"]
            elif e["kind"] == "pretrain.start":
                extra["seconds_per_step"] = e["seconds_per_step"]
                extra["throughput_source"] = e.get("throughput_source", "modeled")
            elif e["kind"] == "pretrain.log":
                steps.append(e["step"])
                losses.append(e["loss"])
                cum += e.get("dollars", 0.0)
                dollars.append(cum)
    return label, steps, losses, dollars, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="frontier-platform — run comparison")
    ap.add_argument("--subtitle", default=None,
                    help="override the auto-generated subtitle")
    args = ap.parse_args()

    series = [load_series(d) for d in args.run_dirs]

    try:
        plt = setup_mpl()
    except ModuleNotFoundError:
        print("matplotlib required for plot_compare.py", file=sys.stderr)
        sys.exit(1)
    import matplotlib.ticker as mtick

    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(
        1, 2, wspace=0.24,
        left=0.07, right=0.985, top=0.80, bottom=0.13,
    )
    ax_loss = fig.add_subplot(gs[0])
    ax_dol = fig.add_subplot(gs[1])

    for i, (label, steps, losses, cum, extra) in enumerate(series):
        c = PALETTE[i % len(PALETTE)]
        nice = (
            f"{label}  ·  {extra['n_params']/1e9:.1f}B × {extra['total_tokens']/1e12:.1f}T  "
            f"·  {extra['gpus']:,}× {extra['gpu_type']}  ·  {extra['throughput_source']}"
        )
        ax_loss.plot(steps, losses, color=c, lw=1.5, alpha=0.9, label=nice)
        ax_dol.semilogy(steps, cum, color=c, lw=1.8, label=label)

    ax_loss.set_title("Training loss")
    ax_loss.set_ylabel("loss")
    ax_loss.set_xlabel("training step")
    ax_loss.legend(fontsize=8.5, loc="upper right", handlelength=2.5)

    ax_dol.set_title("Cumulative pretrain GPU spend  (log scale)")
    ax_dol.set_ylabel("$ cumulative")
    ax_dol.set_xlabel("training step")
    ax_dol.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_dollars))
    ax_dol.legend(fontsize=8.5, loc="lower right")

    fig.suptitle(args.title, fontsize=15, fontweight="bold", y=0.965)

    if args.subtitle is not None:
        sub = args.subtitle
    else:
        sub = f"{len(series)} runs · " + " · ".join(
            f"{lbl} {extra['n_params']/1e9:.1f}B"
            for lbl, _, _, _, extra in series
        )
    fig.text(0.5, 0.905, sub, ha="center", fontsize=10.5, color=TEXT_MUTED)

    fig.savefig(args.out)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
