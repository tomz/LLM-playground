"""DeepConf measurement charts on the addition checkpoint.

Two panels that visualise the honest finding (see examples/deepconf_addition.md):

  * left  — the token-savings vs accuracy tradeoff of online early-abort, swept
            over the abort-floor percentile. The gentle end is ~iso-accuracy at
            ~10% fewer tokens; aggressive trades accuracy for more savings.
  * right — the confidence/correctness separation: histogram of per-trace
            confidence split by whether the trace's answer was correct. The gap
            (correct traces are more confident) is *why* early-abort is safe.

    python tools/plot_deepconf.py --ckpt out/tiny_add3/ckpt_best.pt
    # -> examples/deepconf_addition.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from bench_deepconf import (  # noqa: E402
    _load_ckpt, make_addition_answer_fn, sample_trace, confidence_weighted_vote, softmax_weights,
)


def _eval_once(model, meta, device, *, problems, traces, tokens, window, temperature,
               floor, gen, rng):
    stoi = meta["stoi"]
    answer_fn = make_addition_answer_fn(meta)
    max_val = 10 ** int(meta.get("digits", 2)) - 1
    ok = 0
    emitted = full = 0
    confs_correct: list[float] = []
    confs_wrong: list[float] = []
    for _ in range(problems):
        a, b = rng.randint(0, max_val), rng.randint(0, max_val)
        truth = a + b
        prompt = torch.tensor([[stoi[c] for c in f"{a}+{b}="]], device=device)
        ts = [sample_trace(model, prompt, max_new_tokens=tokens, window=window,
                           temperature=temperature, answer_fn=answer_fn, floor=floor,
                           generator=gen) for _ in range(traces)]
        answers = [t.answer for t in ts]
        scores = [t.confidence for t in ts]
        emitted += sum(t.emitted for t in ts)
        full += traces * tokens
        w = softmax_weights(scores, temperature=1.0)
        winner, _ = confidence_weighted_vote(answers, weights=w)
        ok += int(winner == truth)
        for ans, sc in zip(answers, scores):
            (confs_correct if ans == truth else confs_wrong).append(sc)
    return ok / problems, emitted / full, confs_correct, confs_wrong


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/tiny_add3/ckpt_best.pt")
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--traces", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import random
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, meta = _load_ckpt(args.ckpt, device)

    # Baseline (offline, no floor) for the confidence/correctness histogram + the
    # majority accuracy reference.
    gen = torch.Generator(device=device).manual_seed(0)
    base_acc, _, cc, cw = _eval_once(model, meta, device, problems=args.problems,
                                     traces=args.traces, tokens=args.tokens, window=args.window,
                                     temperature=args.temperature, floor=None, gen=gen,
                                     rng=random.Random(0))

    # Sweep the online abort floor percentile -> (token_frac, accuracy) curve.
    pcts = [0, 10, 20, 30, 40, 50]
    pts = []
    for p in pcts:
        gen = torch.Generator(device=device).manual_seed(0)
        # Calibrate the floor from a warmup set.
        warm = []
        rng = random.Random(99)
        stoi = meta["stoi"]
        af = make_addition_answer_fn(meta)
        max_val = 10 ** int(meta.get("digits", 2)) - 1
        for _ in range(8):
            a, b = rng.randint(0, max_val), rng.randint(0, max_val)
            pr = torch.tensor([[stoi[c] for c in f"{a}+{b}="]], device=device)
            tr = sample_trace(model, pr, max_new_tokens=args.tokens, window=args.window,
                              temperature=args.temperature, answer_fn=af, floor=None, generator=gen)
            warm.extend(tr.group_curve)
        floor = None if p == 0 else float(torch.quantile(torch.tensor(warm), p / 100.0))
        acc, tok_frac, _, _ = _eval_once(model, meta, device, problems=args.problems,
                                         traces=args.traces, tokens=args.tokens, window=args.window,
                                         temperature=args.temperature, floor=floor, gen=gen,
                                         rng=random.Random(0))
        pts.append((tok_frac, acc, p))
        print(f"  p{p:>2}: tokens={tok_frac:.2f}x  acc={acc:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: tradeoff curve
    ax = axes[0]
    xs = [1 - t for t, _, _ in pts]  # token savings
    ys = [a for _, a, _ in pts]
    ax.plot([x * 100 for x in xs], ys, "-o", color="#2980b9")
    for (t, a, p) in pts:
        ax.annotate(f"p{p}", ((1 - t) * 100, a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(base_acc, color="#7f8c8d", ls="--", lw=1, label=f"offline majority acc = {base_acc:.3f}")
    ax.set_title("(left) online early-abort: token savings vs accuracy")
    ax.set_xlabel("token savings (%)"); ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left")

    # right: confidence/correctness separation
    ax = axes[1]
    ax.hist(cw, bins=30, alpha=0.6, color="#c0392b", label=f"wrong (μ={sum(cw)/len(cw):.2f})", density=True)
    ax.hist(cc, bins=30, alpha=0.6, color="#27ae60", label=f"correct (μ={sum(cc)/len(cc):.2f})", density=True)
    ax.set_title("(right) confidence tracks correctness")
    ax.set_xlabel("trace confidence (min windowed logprob)"); ax.set_ylabel("density")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    fig.suptitle("DeepConf on a 10M char-level addition model — 1× RTX 5060 Ti",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "examples" / "deepconf_addition.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
