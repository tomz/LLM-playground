"""Input/output content classifiers (the 'classifier sandwich')."""
from __future__ import annotations


class InputClassifier:
    def score(self, prompt: str) -> dict[str, float]:
        raise NotImplementedError


class OutputClassifier:
    def score(self, prompt: str, completion: str) -> dict[str, float]:
        raise NotImplementedError
