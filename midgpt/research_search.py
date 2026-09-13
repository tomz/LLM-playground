"""Deterministic candidate-generation benchmark with full search cost."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

Generator = Callable[[str, int], Iterable[str]]
Verifier = Callable[[str], tuple[bool, float]]


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    valid: bool
    generation_tokens: int
    verifier_s: float


@dataclass(frozen=True)
class SearchReport:
    attempts: list[CandidateResult]

    @property
    def valid_rate(self) -> float:
        return sum(row.valid for row in self.attempts) / max(1, len(self.attempts))

    @property
    def generation_tokens(self) -> int:
        return sum(row.generation_tokens for row in self.attempts)

    @property
    def verifier_s(self) -> float:
        return sum(row.verifier_s for row in self.attempts)

    @property
    def first_valid(self) -> CandidateResult | None:
        return next((row for row in self.attempts if row.valid), None)


def benchmark_search(prompt: str, generator: Generator, verifier: Verifier,
                     attempts: int) -> SearchReport:
    """Evaluate every generated branch; failed candidates count toward cost."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    rows: list[CandidateResult] = []
    for candidate in generator(prompt, attempts):
        if len(rows) >= attempts:
            break
        start = time.perf_counter()
        valid, measured_s = verifier(candidate)
        elapsed = time.perf_counter() - start
        rows.append(CandidateResult(
            candidate=candidate,
            valid=bool(valid),
            generation_tokens=max(1, (len(candidate) + 3) // 4),
            verifier_s=max(float(measured_s), elapsed),
        ))
    return SearchReport(rows)
