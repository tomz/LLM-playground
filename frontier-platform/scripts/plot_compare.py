#!/usr/bin/env python3
"""Compare multiple simulator runs on one figure.

Reads each `<run_dir>/events.jsonl`, overlays loss curves + cumulative
$ on a single PNG. Useful for "how does 1B vs 7B vs 70B vs 400B compare"
and for "spec-sheet vs real-GPU-calibrated 7B".

Usage:
    python scripts/plot_compare.py out/sim/1b out/sim/7b out/sim/70b out/sim/400b \\
        --out out/sim/compare_sizes.png
"""
from __future__ import annotations
import argparse, json, os, sys


def load_series(run_dir: str):
    path = os.path.join(run_dir, "events.jsonl")
    steps, losses, dollars = [], [], []
    cum = 0.0
    label = os.path.basename(run_dir.rstrip("/"))
    extra = {}
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
    args = ap.parse_args()

    series = [load_series(d) for d in args.run_dirs]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib required for plot_compare.py", file=sys.stderr)
        sys.exit(1)

    colors = ["#0066cc", "#cc6600", "#009933", "#cc0066", "#663399", "#999900"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for i, (label, steps, losses, cum, extra) in enumerate(series):
        c = colors[i % len(colors)]
        nice = f"{label}  ({extra['n_params']/1e9:.1f}B × {extra['total_tokens']/1e12:.1f}T, " \
               f"{extra['gpus']}× {extra['gpu_type']}, {extra['throughput_source']})"
        axes[0].plot(steps, losses, color=c, lw=1.2, label=nice)
        axes[1].semilogy(steps, cum, color=c, lw=1.5, label=label)

    axes[0].set_title("training loss")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set_title("cumulative pretrain GPU spend  (log scale)")
    axes[1].set_ylabel("$ cumulative (log)")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
