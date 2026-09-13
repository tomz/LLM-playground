"""Tests for retained reasoning and context compaction."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context import (  # noqa: E402
    ContextState,
    Turn,
    compact_context,
    extractive_summary,
    fidelity,
    retain_reasoning,
    rolling_truncate,
)


def _history() -> ContextState:
    return ContextState([
        Turn("system", "Never modify production data", pinned=True),
        Turn("user", "Project codename is ORCHID"),
        Turn("assistant", "I will inspect logs", "Need preserve codename"),
        Turn("tool", "log batch one " * 20),
        Turn("assistant", "Found timeout in worker seven"),
        Turn("tool", "log batch two " * 20),
    ])


def test_reasoning_retention_is_explicit():
    turn = Turn("assistant", "act", "private plan")
    assert retain_reasoning(turn).reasoning == "private plan"
    assert retain_reasoning(turn, retain=False).reasoning is None


def test_compaction_preserves_pinned_and_old_facts():
    state = compact_context(_history(), 80, extractive_summary, keep_recent=2)
    assert state.tokens <= 80
    assert state.turns[0].pinned
    assert state.compacted_turns > 0
    assert fidelity(["ORCHID", "worker seven", "Never modify production data"], state) == 1.0


def test_rolling_truncation_loses_old_fact_but_compaction_retains_it():
    truncated = rolling_truncate(_history(), 50)
    compacted = compact_context(_history(), 50, extractive_summary, keep_recent=2)
    assert fidelity(["ORCHID"], truncated) == 0.0
    assert fidelity(["ORCHID"], compacted) == 1.0


def test_pathological_summary_still_honors_budget():
    result = compact_context(_history(), 30, lambda _: "x" * 10_000, keep_recent=1)
    assert result.tokens <= 30


def test_invalid_budgets_rejected():
    try:
        compact_context(_history(), 0, extractive_summary)
    except ValueError:
        pass
    else:
        raise AssertionError("zero budget must fail")
