"""Deterministic synthetic corpora for tests."""
from __future__ import annotations
import random
from pathlib import Path

_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "alpha beta gamma delta epsilon zeta eta theta iota "
    "model training data pipeline gradient descent loss curve"
).split()


def write_corpus(root: str | Path, n_files: int = 20, words_per_file: int = 200, seed: int = 0) -> Path:
    rng = random.Random(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        words = [rng.choice(_WORDS) for _ in range(words_per_file)]
        (root / f"doc_{i:04d}.txt").write_text(" ".join(words) + ".\n", encoding="utf-8")
    return root
