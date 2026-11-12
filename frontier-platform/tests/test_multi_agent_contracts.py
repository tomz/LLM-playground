"""Pre-dispatch admission and honest accounting for all attempted workers."""
from __future__ import annotations

import pytest

from frontier_platform.serving.multi_agent import run_parallel


def test_f7_legacy_overrun_is_not_reported_as_zero():
    calls = []

    def worker(task):
        calls.append(task)
        return "expensive", 20

    result = run_parallel("task", {"a": worker, "b": worker}, token_budget=10)
    assert calls == ["task"]
    assert result.tokens_used == 20
    assert result.measured_tokens == 20
    assert result.stopped_reason == "token_budget"
    assert result.branches[0].tokens == 20
    assert result.branches[0].status == "overrun"
    assert result.answer == ""


def test_reserved_worker_is_admitted_before_dispatch_and_receives_limit():
    from frontier_platform.serving.multi_agent import ReservedWorker

    calls = []

    def worker(task, max_tokens):
        calls.append((task, max_tokens))
        return "ok", max_tokens

    worker = ReservedWorker(max_tokens=6, run=worker)
    denied = run_parallel("task", {"a": worker}, token_budget=5)
    assert calls == []
    assert denied.tokens_used == 0
    assert denied.stopped_reason == "token_budget"
    result = run_parallel("task", {"a": worker, "b": worker}, token_budget=10)
    assert calls == [("task", 6)]
    assert result.tokens_used == 6
    assert result.branches[0].reserved_tokens == 6
    assert result.answer == "ok"


def test_reserved_overrun_is_recorded_even_below_total_budget():
    from frontier_platform.serving.multi_agent import ReservedWorker

    calls = []

    def worker(task, max_tokens):
        calls.append(max_tokens)
        return "too much", max_tokens + 1

    bounded = ReservedWorker(max_tokens=5, run=worker)
    result = run_parallel("task", {"a": bounded, "b": bounded}, token_budget=100)
    assert calls == [5]
    assert result.tokens_used == 6
    assert result.stopped_reason == "reservation_overrun"
    assert result.branches[0].status == "overrun"


@pytest.mark.parametrize("count", [None, -1, True, 1.5, float("nan")])
def test_unmeasured_or_invalid_work_is_unknown_and_stops(count):
    calls = []

    def bad(task):
        calls.append(task)
        return "bad", count

    result = run_parallel("task", {"a": lambda task: ("ok", 3), "b": bad, "c": bad}, token_budget=10)
    assert calls == ["task"]
    assert result.tokens_used is None
    assert result.measured_tokens == 3
    assert result.branches[-1].tokens is None
    assert result.branches[-1].status == "unknown"
    assert result.answer == "ok"


def test_worker_exception_retains_unknown_attempt_and_elapsed_cost():
    calls = []

    def bad(task):
        calls.append(task)
        raise RuntimeError("crashed after dispatch")

    result = run_parallel("task", {"a": bad, "b": bad}, token_budget=10)
    assert calls == ["task"]
    assert result.tokens_used is None
    assert result.branches[0].status == "error"
    assert "crashed after dispatch" in result.branches[0].error
    assert result.branches[0].elapsed_seconds >= 0


@pytest.mark.parametrize("failure", ["raise", "nan"])
def test_scorer_failure_never_erases_execution_cost(failure):
    def scorer(answer):
        if failure == "raise":
            raise RuntimeError("bad score")
        return float("nan")

    result = run_parallel("task", {"a": lambda task: ("answer", 7)}, token_budget=10, scorer=scorer)
    assert result.tokens_used == 7
    assert result.measured_tokens == 7
    assert result.branches[0].status == "error"
    assert result.stopped_reason == "scorer_error"


def test_exhausted_budget_prevents_even_legacy_dispatch():
    calls = []

    def worker(task):
        calls.append(task)
        return "ok", 5

    result = run_parallel("task", {"a": worker, "b": worker}, token_budget=5)
    assert calls == ["task"]
    assert result.tokens_used == 5


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_integer_reservations_are_required(invalid):
    from frontier_platform.serving.multi_agent import ReservedWorker

    with pytest.raises(ValueError):
        ReservedWorker(max_tokens=invalid, run=lambda task, max_tokens: ("", 0))
    with pytest.raises(ValueError):
        run_parallel("task", {}, token_budget=invalid)
