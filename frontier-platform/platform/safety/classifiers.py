"""Input/output content classifiers (the 'classifier sandwich').

Real impl wraps Llama-Guard-3 or a fine-tuned RoBERTa per category. This toy
implementation uses keyword + length heuristics and exists so the safety
pipeline can be exercised end-to-end in tests without a GPU.
"""
from __future__ import annotations
import re

_CATS = {
    "hate": ["hate", "slur", "racist", "bigot"],
    "violence": ["kill", "murder", "attack", "shoot", "bomb"],
    "sexual": ["porn", "explicit", "nsfw"],
    "selfharm": ["suicide", "selfharm", "self-harm"],
}


def _score(text: str) -> dict[str, float]:
    low = text.lower()
    tokens = re.findall(r"[a-z-]+", low)
    n = max(1, len(tokens))
    out: dict[str, float] = {}
    for cat, words in _CATS.items():
        hits = sum(1 for t in tokens if t in words)
        out[cat] = min(1.0, hits / n * 10.0)
    return out


class InputClassifier:
    def score(self, prompt: str) -> dict[str, float]:
        return _score(prompt)


class OutputClassifier:
    def score(self, prompt: str, completion: str) -> dict[str, float]:
        # Combine: max over prompt and completion (output usually dominates).
        a, b = _score(prompt), _score(completion)
        return {k: max(a[k], b[k]) for k in a}
