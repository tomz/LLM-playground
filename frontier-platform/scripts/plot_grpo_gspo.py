"""GRPO vs GSPO + RLPR measurement charts.

Reads the JSON written by tools/bench_grpo_gspo.py (--json-out) and produces a
three-panel figure:

  * left   — importance-ratio dispersion: GRPO token-level vs GSPO sequence-level
             std, as a bar (the ~4x variance-reduction headline).
  * middle — GRPO vs GSPO reward trajectory over steps (stability).
  * right  — RLPR verifier-free reward (mean answer-prob) + emit-rate over steps.

    python scripts/plot_grpo_gspo.py out/grpo_gspo_moe.json   # -> examples/grpo_gspo_rlpr.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "grpo_gspo_moe.json"
    d = json.loads(src.read_text())
    var = d["ratio_variance"]
    label = "MoE" if d.get("moe") else "dense"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) variance bar
    ax = axes[0]
    bars = ax.bar(["GRPO\n(token)", "GSPO\n(sequence)"],
                  [var["token_ratio_std"], var["seq_ratio_std"]],
                  color=["#c0392b", "#2980b9"])
    ax.set_title(f"(1) importance-ratio std — {var['variance_reduction']:.1f}× lower (GSPO)")
    ax.set_ylabel("ratio std on identical rollouts")
    for b, v in zip(bars, [var["token_ratio_std"], var["seq_ratio_std"]]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)

    # (2) reward trajectory
    ax = axes[1]
    for arm, color in [("grpo_history", "#c0392b"), ("gspo_history", "#2980b9")]:
        h = d.get(arm, [])
        if h:
            ax.plot([e["step"] for e in h], [e["reward"] for e in h],
                    color=color, lw=1.6, label=arm.split("_")[0].upper())
    ax.set_title("(2) reward trajectory — GRPO vs GSPO")
    ax.set_xlabel("step"); ax.set_ylabel("mean reward"); ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # (3) RLPR
    ax = axes[2]
    h = d.get("rlpr_history", [])
    if h:
        steps = [e["step"] for e in h]
        ax.plot(steps, [e["reward"] for e in h], color="#27ae60", lw=1.6, label="answer-prob (RLPR reward)")
        if "emit_rate" in h[0]:
            ax2 = ax.twinx()
            ax2.plot(steps, [e["emit_rate"] for e in h], color="#8e44ad", lw=1.2, ls="--", label="emit-rate")
            ax2.set_ylabel("answer emit-rate", color="#8e44ad")
            ax2.set_ylim(-0.05, 1.05)
    ax.set_title("(3) RLPR verifier-free reward")
    ax.set_xlabel("step"); ax.set_ylabel("mean answer-prob", color="#27ae60")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"GRPO vs GSPO + RLPR — {label} policy, 1× RTX 5060 Ti", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "examples" / "grpo_gspo_rlpr.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
