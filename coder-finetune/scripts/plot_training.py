"""Plot loss / accuracy / grad-norm / LR from a TRL trainer_state.json.

Usage:
    python scripts/plot_training.py out/lora_3050_1p5b/checkpoint-250 \
        --title "1.5B LoRA on 3050" --out out/lora_3050_1p5b/training.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(path: Path) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "trainer_state.json"
    return json.loads(p.read_text())["log_history"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="checkpoint dir or trainer_state.json")
    ap.add_argument("--title", default="Training run")
    ap.add_argument("--out", default=None, help="output PNG (default: <path>/training.png)")
    args = ap.parse_args()

    log = load_log(Path(args.path))
    steps = [e["step"] for e in log if "loss" in e]
    loss = [e["loss"] for e in log if "loss" in e]
    acc = [e.get("mean_token_accuracy") for e in log if "loss" in e]
    gn = [e.get("grad_norm") for e in log if "loss" in e]
    lr = [e.get("learning_rate") for e in log if "loss" in e]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, loss, color="#c0392b", lw=1.6)
    ax.set(title="loss", xlabel="step", ylabel="cross-entropy"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, acc, color="#27ae60", lw=1.6)
    ax.set(title="mean token accuracy", xlabel="step", ylabel="acc"); ax.grid(alpha=0.3)
    ax.set_ylim(min(acc) - 0.02, max(acc) + 0.02)

    ax = axes[1, 0]
    ax.plot(steps, gn, color="#2980b9", lw=1.6)
    ax.set(title="grad norm", xlabel="step", ylabel="||g||"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, lr, color="#8e44ad", lw=1.6)
    ax.set(title="learning rate", xlabel="step", ylabel="lr"); ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.path).parent / "training.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
