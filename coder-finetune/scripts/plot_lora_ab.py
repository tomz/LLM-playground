"""LoRA Without Regret A/B — token-accuracy trajectory charts.

Two panels, both plotting mean-token-accuracy vs optimizer step (the
scaling-invariant convergence signal; see the example doc for why train_loss is
not comparable across ranks under rsLoRA):

  * left  — 8k Python, 3 epochs: r=256 catches up to a tie (the thesis).
  * right — 30k x 9 languages, 2 epochs: r=256 still undertrained, r=16 wins.

The shared shape across both panels is the story: the r=256 curve dips on its
warm-up (16x the adapter params to move) then climbs steeply — given enough steps
it matches (left), but at a fixed epoch budget on harder data it never catches up
(right). Budget, not dataset size, is the binding constraint.

    python scripts/plot_lora_ab.py            # -> examples/lora_without_regret_ab.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PANELS = [
    {
        "title": "8k Python, 3 epochs  →  tie at convergence",
        "arms": [
            ("r=16", "#2980b9", "out/ab_lora_r16_3ep/checkpoint-780/trainer_state.json"),
            ("r=256", "#c0392b", "out/ab_lora_r256_3ep/checkpoint-780/trainer_state.json"),
        ],
    },
    {
        "title": "30k × 9 languages, 2 epochs  →  r=16 wins (r=256 undertrained)",
        "arms": [
            ("r=16", "#2980b9", "out/ab_lora_r16_sep/checkpoint-1996/trainer_state.json"),
            ("r=256", "#c0392b", "out/ab_lora_r256_sep/checkpoint-1996/trainer_state.json"),
        ],
    },
]


def load_acc(path: Path) -> tuple[list[int], list[float]]:
    log = json.loads(path.read_text())["log_history"]
    rows = [e for e in log if "mean_token_accuracy" in e]
    return [int(e["step"]) for e in rows], [e["mean_token_accuracy"] for e in rows]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, panel in zip(axes, PANELS):
        finals = {}
        for label, color, rel in panel["arms"]:
            p = ROOT / rel
            if not p.exists():
                print(f"[warn] missing {p}")
                continue
            steps, acc = load_acc(p)
            ax.plot(steps, acc, color=color, lw=1.8, label=label)
            finals[label] = acc[-1] if acc else float("nan")
        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        # Annotate the final gap.
        if {"r=16", "r=256"} <= set(finals):
            gap = finals["r=16"] - finals["r=256"]
            ax.text(0.03, 0.95, f"final token-acc\n r=16:  {finals['r=16']:.3f}\n r=256: {finals['r=256']:.3f}\n Δ = {gap:+.3f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    axes[0].set_ylabel("mean token accuracy")
    fig.suptitle("LoRA Without Regret — r=16 vs r=256 (Qwen2.5-Coder-0.5B, 2× RTX 5060 Ti)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "examples" / "lora_without_regret_ab.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
