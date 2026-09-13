"""Tests for deterministic verifier-gated candidate search."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_search import benchmark_search  # noqa: E402


def test_search_counts_failed_branches_and_verifier_cost():
    def generator(prompt, attempts):
        return ["wrong", "also wrong", "answer=42", "unused"]

    def verifier(candidate):
        return candidate == "answer=42", 0.25

    report = benchmark_search("solve", generator, verifier, attempts=3)
    assert len(report.attempts) == 3
    assert report.valid_rate == 1 / 3
    assert report.first_valid is not None
    assert report.first_valid.candidate == "answer=42"
    assert report.generation_tokens == sum(row.generation_tokens for row in report.attempts)
    assert report.verifier_s >= 0.75


def test_search_rejects_zero_attempt_budget():
    try:
        benchmark_search("x", lambda prompt, n: [], lambda candidate: (False, 0.0), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero attempts must fail")
