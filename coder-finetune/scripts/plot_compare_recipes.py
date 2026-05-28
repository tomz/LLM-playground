"""Cross-recipe comparison plot for the three coder-finetune LoRA runs.

Plots loss curves of the 0.5B (3050), 1.5B (3050), and 3B (5060 Ti) recipes
on shared axes -- one in "training progress" (% epoch), one in wall-clock
seconds, so the speed/quality tradeoff is visible.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    {
        "label": "0.5B / RTX 3050 (builtin, memorize)",
        "color": "#7f8c8d",
        "trainer_state": ROOT / "out/lora_3050/checkpoint-80/trainer_state.json",
        "wall_s": 83.8,
    },
    {
        "label": "1.5B / RTX 3050 (Magicoder 2k, grad_ckpt)",
        "color": "#2980b9",
        "trainer_state": ROOT / "out/lora_3050_1p5b/checkpoint-250/trainer_state.json",
        "wall_s": 1445.0,  # 24 min 5 s
    },
    {
        "label": "3B / RTX 5060 Ti (Magicoder 2.5k, packed)",
        "color": "#c0392b",
        "trainer_state": ROOT / "out/lora_5060ti/checkpoint-161/trainer_state.json",
        "wall_s": 719.5,
    },
]


def load(path: Path) -> tuple[list[int], list[float], list[float]]:
    log = json.loads(path.read_text())["log_history"]
    rows = [e for e in log if "loss" in e and "epoch" in e]
    steps = [int(e["step"]) for e in rows]
    loss = [float(e["loss"]) for e in rows]
    epoch = [float(e["epoch"]) for e in rows]
    return steps, loss, epoch


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "coder-finetune — LoRA loss curves across three recipes",
        fontsize=13, fontweight="bold",
    )

    for run in RUNS:
        steps, loss, epoch = load(run["trainer_state"])
        # x in % of epoch (all three are 1-epoch runs, so this is also
        # "fraction of training complete")
        pct = [e * 100 for e in epoch]
        ax1.plot(pct, loss, color=run["color"], lw=1.8, label=run["label"], marker="o", markersize=3)

        # x in wall-clock seconds (linear interpolation: assume even step
        # cadence within an epoch -- close enough for visual comparison)
        last = steps[-1]
        secs = [s / last * run["wall_s"] for s in steps]
        ax2.plot(secs, loss, color=run["color"], lw=1.8, label=run["label"], marker="o", markersize=3)

    ax1.set(xlabel="training progress (% of 1 epoch)", ylabel="train loss",
            title="loss vs progress")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(0, 100)

    ax2.set(xlabel="wall-clock seconds", ylabel="train loss",
            title="loss vs wall-clock time")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = ROOT / "examples/compare_recipes.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
