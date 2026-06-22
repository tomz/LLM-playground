"""Generate a character-level multi-digit ADDITION corpus for nanogpt-edu.

This is a *verifiable* toy task: each line is ``a+b=c\n`` where ``c == a+b``.
Unlike free-form TinyShakespeare, every generated answer has a ground truth, so
it's the right substrate to measure DeepConf's **accuracy lift** (the
confidence-weighted vote can only beat a plain majority vote when "more
confident" actually correlates with "more correct" — which needs a checkable
answer).

We write the same char-level shard format as ``prepare_shakespeare.py``
(``train.bin`` / ``val.bin`` uint16 + ``meta.pkl`` with ``stoi``/``itos``), so
``train.py`` and the DeepConf bench consume it unchanged. The vocab is just the
digits, ``+``, ``=`` and newline.

Operands are sampled up to ``--digits`` digits (default 2, i.e. 0..99). Train and
val draw from **disjoint** operand pairs so val measures generalization, not
memorization. Reverse-order answers are optional (``--reverse``) — writing the
sum least-significant-digit first is the classic trick that makes addition
learnable for a tiny LM, but we keep forward order by default so the answer
parses naturally left-to-right for the bench.

    python prepare_addition.py --digits 2 --train 20000 --val 2000
    python train.py --config configs/tiny_add.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data_add")


def _example(a: int, b: int, *, reverse: bool) -> str:
    s = a + b
    ans = str(s)[::-1] if reverse else str(s)
    return f"{a}+{b}={ans}\n"


def _sample_pairs(n: int, max_val: int, rng: random.Random, *, exclude: set) -> list[tuple[int, int]]:
    """Sample ``n`` distinct (a, b) operand pairs not in ``exclude``."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    # Cap attempts so a too-small operand space can't spin forever.
    attempts = 0
    limit = n * 50 + 10_000
    while len(out) < n and attempts < limit:
        attempts += 1
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        key = (a, b)
        if key in seen or key in exclude:
            continue
        seen.add(key)
        out.append(key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=2, help="max digits per operand (2 -> 0..99)")
    ap.add_argument("--train", type=int, default=20000, help="number of train examples")
    ap.add_argument("--val", type=int, default=2000, help="number of val examples")
    ap.add_argument("--reverse", action="store_true", help="write the sum least-significant-digit first")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    rng = random.Random(args.seed)
    max_val = 10 ** args.digits - 1

    # Disjoint train/val operand pairs: sample val first, exclude it from train.
    val_pairs = _sample_pairs(args.val, max_val, rng, exclude=set())
    train_pairs = _sample_pairs(args.train, max_val, rng, exclude=set(val_pairs))

    train_text = "".join(_example(a, b, reverse=args.reverse) for a, b in train_pairs)
    val_text = "".join(_example(a, b, reverse=args.reverse) for a, b in val_pairs)

    # Char vocab over the union (stable, sorted) — digits, '+', '=', '\n'.
    chars = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA, "train.bin"))
    val_ids.tofile(os.path.join(DATA, "val.bin"))
    with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
        pickle.dump(
            {"vocab_size": len(chars), "stoi": stoi, "itos": itos,
             "task": "addition", "digits": args.digits, "reverse": args.reverse},
            f,
        )
    print(f"vocab={len(chars)} chars: {''.join(chars)!r}")
    print(f"train: {len(train_pairs)} examples, {len(train_ids):,} chars")
    print(f"val:   {len(val_pairs)} examples, {len(val_ids):,} chars (disjoint operands)")
    print(f"example: {train_text.splitlines()[0]!r}  reverse={args.reverse}")


if __name__ == "__main__":
    main()
