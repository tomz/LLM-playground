"""Offline callback integration and lifetime cost/budget tests."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from distgpt.eval.rollout_queue import RolloutQueue, WorkItem
from distgpt.eval.rollout_worker import (
    Budget, Generation, LocalWorker, StalenessPolicy, toy_checker, toy_generator,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_queue(**kwargs):
    queue = RolloutQueue(**kwargs)
    queue.submit(WorkItem("a", {"length": 3}, 0, domain="short"))
    return queue


def test_real_short_long_callbacks_and_snapshot(tmp_path):
    queue = make_queue()
    queue.submit(WorkItem("b", {"length": 64}, 1, domain="long"))
    clock = Clock()
    seen = []

    def checker(item, generation, meter):
        seen.append((item.domain, generation.text))
        clock.advance(0.25)
        return toy_checker(item, generation, meter)

    summary = LocalWorker(queue, toy_generator, checker, clock=clock, timer=clock).run()
    assert summary["completed"] == 2 and len(seen) == 2
    assert summary["totals"] == {"attempts": 2, "input_tokens": 2, "output_tokens": 67,
                                  "wall_s": 0.5, "grader_calls": 2, "grader_cost": 2.0}
    assert summary["domains"]["short"]["output_tokens"] == 3
    assert summary["domains"]["long"]["output_tokens"] == 64
    assert all(c.result["verdict"]["correct"] for c in queue.completed.values())
    restored = RolloutQueue.restore(queue.snapshot(tmp_path / "state.json"))
    assert LocalWorker(restored, toy_generator, checker).run()["totals"] == summary["totals"]


def test_checker_error_retries_and_keeps_all_costs():
    queue = make_queue(max_attempts=2)
    clock = Clock()

    def checker(item, generation, meter):
        clock.advance(0.5)
        if item.attempt == 0:
            raise RuntimeError("checker unavailable")
        return toy_checker(item, generation, meter)

    summary = LocalWorker(queue, toy_generator, checker, clock=clock, timer=clock,
                          grader_cost=0.25).run()
    records = list(queue.attempts.values())
    assert [r.status for r in records] == ["failed", "completed"]
    assert "checker unavailable" in records[0].reason
    assert summary["totals"]["output_tokens"] == 6
    assert summary["totals"]["grader_calls"] == 2
    assert summary["totals"]["grader_cost"] == 0.5
    assert summary["totals"]["wall_s"] == 1


def test_generator_error_and_permanent_checker_error():
    queue = make_queue(max_attempts=2)

    def generator(item, meter):
        meter.tokens(input_tokens=1, output_tokens=2)
        raise RuntimeError("partial generation")

    LocalWorker(queue, generator, toy_checker).run()
    assert queue.totals()["output_tokens"] == 4
    assert queue.totals()["grader_calls"] == 0
    assert "partial generation" in queue.failed["a"]
    queue = make_queue(max_attempts=2)

    def checker(item, generation, meter):
        raise RuntimeError("broken checker")

    LocalWorker(queue, toy_generator, checker).run()
    assert queue.totals()["grader_calls"] == 2
    assert "broken checker" in queue.failed["a"]


def test_interrupted_worker_reclaims_after_restart_with_partial_cost(tmp_path):
    queue = make_queue()
    clock = Clock()

    def crash(item, meter):
        meter.tokens(input_tokens=1, output_tokens=1)
        clock.advance(1)
        raise KeyboardInterrupt()

    worker = LocalWorker(queue, crash, toy_checker, clock=clock, timer=clock, lease_s=2)
    with pytest.raises(KeyboardInterrupt):
        worker.run_one()
    old = queue.leased["a"]
    queue = RolloutQueue.restore(queue.snapshot(tmp_path / "crash.json"))
    clock.advance(2)
    LocalWorker(queue, toy_generator, toy_checker, clock=clock, timer=clock).run()
    assert queue.completed["a"].attempt == 1
    assert queue.totals()["output_tokens"] == 4
    assert queue.totals()["wall_s"] == 1
    assert queue.attempts[old.token].status == "expired"
    assert not queue.complete("a", "local", {}, clock(), lease_token=old.token)


def test_late_checker_cannot_complete_same_worker_reacquisition():
    queue = make_queue()
    clock = Clock()

    def checker(item, generation, meter):
        clock.advance(2)
        new = queue.acquire("local", clock(), 10)
        assert new.item.attempt == 1
        return {"correct": True}

    token = LocalWorker(queue, toy_generator, checker, clock=clock, timer=clock,
                        lease_s=1).run_one()
    assert queue.attempts[token].status == "stale"
    assert not queue.completed
    assert queue.leased["a"].token != token
    assert queue.totals()["output_tokens"] == 3
    assert queue.totals()["grader_cost"] == 1
    assert queue.totals()["wall_s"] == 2


@pytest.mark.parametrize("policy,logprob,current,lag,accepted", [
    (0, 0, 0, 0, True), (0, 0, 1, 0, False), (0, 0, 1, 1, True),
    (0, 0, 2, 1, False), (1, 1, 1, 0, False), (0, 1, 0, 0, False),
])
def test_provenance_and_staleness(policy, logprob, current, lag, accepted):
    queue = make_queue(max_attempts=1)

    def generator(item, meter):
        result = toy_generator(item, meter)
        return dataclasses.replace(result, policy_version=policy, logprob_version=logprob)

    LocalWorker(queue, generator, toy_checker,
                staleness=StalenessPolicy(current, lag)).run()
    assert bool(queue.completed) == accepted
    assert queue.totals()["output_tokens"] == 3
    assert queue.totals()["grader_calls"] == int(accepted)
    record = next(iter(queue.attempts.values()))
    assert record.result_policy_version == policy
    assert record.result_logprob_version == logprob


def test_future_policy_and_behavior_logprob_mismatch():
    item = WorkItem("a", {}, 0, policy_version=2, logprob_version=2)
    with pytest.raises(ValueError, match="lag"):
        StalenessPolicy(1, 10).validate(item, Generation("x", 2, 2, (0.0,)))
    item = dataclasses.replace(item, logprob_version=1)
    with pytest.raises(ValueError, match="behavior"):
        StalenessPolicy(2, 1).validate(item, Generation("x", 2, 1, (0.0,)))


def test_policy_advance_during_checker_is_rechecked_and_charged():
    queue = make_queue(max_attempts=1)

    def checker(item, generation, meter):
        worker.staleness = StalenessPolicy(1, 0)
        return {"correct": True}

    worker = LocalWorker(queue, toy_generator, checker)
    worker.run()
    assert not queue.completed and "lag" in queue.failed["a"]
    assert queue.totals()["grader_calls"] == 1


@pytest.mark.parametrize("budget,expected", [
    (Budget(max_tokens=2), {"input_tokens": 1, "output_tokens": 1, "grader_calls": 0}),
    (Budget(max_grader_calls=1), {"input_tokens": 1, "output_tokens": 3, "grader_calls": 1}),
    (Budget(max_grader_cost=0.5), {"input_tokens": 3, "output_tokens": 9, "grader_calls": 0}),
    (Budget(max_attempts=1), {"input_tokens": 1, "output_tokens": 3, "grader_calls": 1}),
])
def test_cumulative_budgets_include_failed_work_and_resume(tmp_path, budget, expected):
    queue = make_queue()

    def checker(item, generation, meter):
        raise RuntimeError("retry")

    LocalWorker(queue, toy_generator, checker, budget=budget).run()
    for key, value in expected.items():
        assert queue.totals()[key] == value
    restored = RolloutQueue.restore(queue.snapshot(tmp_path / "budget.json"))
    before = restored.totals()
    LocalWorker(restored, toy_generator, checker, budget=budget).run()
    assert restored.totals() == before


def test_wall_overrun_is_charged_but_not_accepted():
    queue = make_queue()
    clock = Clock()

    def checker(item, generation, meter):
        clock.advance(2)
        return {"correct": True}

    summary = LocalWorker(queue, toy_generator, checker, budget=Budget(max_wall_s=1),
                          clock=clock, timer=clock).run()
    assert not queue.completed
    assert summary["budget_exhausted"]
    assert summary["totals"]["wall_s"] == 2
    assert summary["totals"]["grader_calls"] == 1


@pytest.mark.parametrize("field,value", [
    ("max_attempts", 0), ("max_attempts", 1.5), ("max_tokens", -1),
    ("max_wall_s", float("nan")), ("max_wall_s", float("inf")),
    ("max_grader_calls", True), ("max_grader_cost", float("inf")),
    ("max_grader_cost", -1),
])
def test_invalid_budgets(field, value):
    with pytest.raises(ValueError):
        Budget(**{field: value})


def test_bad_verifier_result_and_logprob_count_are_failures():
    queue = make_queue(max_attempts=1)
    LocalWorker(queue, toy_generator, lambda *args: {"reward": float("nan")}).run()
    assert not queue.completed
    assert queue.totals()["grader_calls"] == 1
    queue = make_queue(max_attempts=1)

    def generator(item, meter):
        meter.tokens(output_tokens=1)
        return Generation("x", 0, 0, ())

    LocalWorker(queue, generator, toy_checker).run()
    assert "log-prob count" in queue.failed["a"]


def test_cli_offline_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "demo.json"
    command = [sys.executable, "-m", "distgpt.eval.rollout_worker", "--snapshot", str(path)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15, check=True)
    summary = json.loads(result.stderr.strip())
    assert summary["completed"] == 2
    assert summary["totals"]["output_tokens"] == 67
    result = subprocess.run(command + ["--resume"], cwd=root, capture_output=True,
                            text=True, timeout=15, check=True)
    assert json.loads(result.stderr.strip())["totals"] == summary["totals"]
