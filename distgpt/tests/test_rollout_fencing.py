"""Lease fencing, migration, validation, and atomic snapshot regressions."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from distgpt.eval.rollout_queue import RolloutQueue, Usage, WorkItem


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("cause", ["expiry", "failure"])
def test_same_worker_old_messages_cannot_mutate_reacquired_lease(tmp_path, restart, cause):
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0))
    old = queue.acquire("w", 0, 1)
    if cause == "failure":
        assert queue.fail("a", "w", "retry", 0.5, lease_token=old.token)
    else:
        queue.reclaim_expired(1)
    new = queue.acquire("w", 2, 10)
    assert new.token != old.token
    if restart:
        queue = RolloutQueue.restore(queue.snapshot(tmp_path / "state.json"))
    for fence in ({}, {"lease_token": old.token}, {"attempt": old.item.attempt},
                  {"lease_token": new.token, "attempt": old.item.attempt}):
        assert not queue.complete("a", "w", {"stale": True}, 3, **fence)
        assert not queue.fail("a", "w", "stale", 3, **fence)
        assert not queue.heartbeat("a", "w", 3, **fence)
        assert queue.leased["a"] == new
    assert queue.heartbeat("a", "w", 3, 20, lease_token=new.token, attempt=1)
    assert queue.leased["a"].expires_at == 23
    assert queue.complete("a", "w", {"ok": True}, 4, lease_token=new.token)
    queue = RolloutQueue.restore(queue.snapshot(tmp_path / "done.json"))
    assert not queue.complete("a", "w", {}, 5)
    assert not queue.complete("a", "w", {}, 5, lease_token=old.token)
    assert queue.complete("a", "w", {}, 5, lease_token=new.token)
    assert queue.completed["a"].result == {"ok": True}


def test_attempt_only_does_not_authorize_heartbeat_or_completion():
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0))
    queue.acquire("w", 0, 1)
    assert not queue.heartbeat("a", "w", 1, attempt=0)
    lease = queue.acquire("w", 1, 1)
    assert not queue.heartbeat("a", "w", 1.5, 2, attempt=1)
    assert queue.heartbeat("a", "w", 1.5, 2, attempt=1, lease_token=lease.token)
    assert not queue.complete("a", "w", {}, 2, attempt=1)
    assert queue.complete("a", "w", {}, 2, attempt=1, lease_token=lease.token)


def test_pending_snapshot_restore_requires_new_acquisition_identity(tmp_path):
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0))
    path = queue.snapshot(tmp_path / "pending.json")
    old = queue.acquire("w", 0, 10)
    restored = RolloutQueue.restore(path)
    new = restored.acquire("w", 1, 10)
    assert old.item.attempt == new.item.attempt == 0
    assert old.token != new.token
    for fence in ({}, {"attempt": 0}, {"lease_token": old.token}):
        assert not restored.complete("a", "w", {"stale": True}, 2, **fence)
        assert not restored.fail("a", "w", "stale", 2, **fence)
        assert not restored.heartbeat("a", "w", 2, **fence)
    assert restored.complete("a", "w", {"fresh": True}, 2, lease_token=new.token)


def test_crashed_workers_eventually_dead_letter():
    queue = RolloutQueue(max_attempts=2)
    queue.submit(WorkItem("a", {}, 0))
    queue.acquire("w", 0, 1)
    queue.acquire("w", 1, 1)
    assert queue.acquire("w", 2, 1) is None
    assert queue.failed == {"a": "lease expired"}
    assert [r.status for r in queue.attempts.values()] == ["expired", "expired"]


def test_v1_migration_invalidates_unfenced_live_leases(tmp_path):
    item = {"item_id": "a", "payload": {}, "created_at": 0, "attempt": 0, "parent_id": "p"}
    state = {"schema_version": 1, "pending": [], "failed": {},
             "leased": {"a": {"item": item, "worker_id": "w", "expires_at": 100}},
             "completed": {"done": {"item_id": "done", "worker_id": "w", "result": {},
                                      "completed_at": 0, "attempt": 0}}}
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(state))
    queue = RolloutQueue.restore(path)
    assert not queue.leased
    new = queue.acquire("w", 1)
    assert new.item.attempt == 1 and new.item.parent_id == "p"
    assert not queue.complete("a", "w", {}, 2)
    assert not queue.complete("a", "w", {}, 2, attempt=0)
    assert not queue.complete("done", "w", {}, 2)
    assert not queue.complete("done", "w", {}, 2, attempt=0)
    assert queue.complete("a", "w", {}, 2, lease_token=new.token)
    restored = RolloutQueue.restore(queue.snapshot(path))
    assert restored.to_dict() == queue.to_dict()
    assert restored.to_dict()["schema_version"] == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_times_do_not_mutate_queue(value):
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0))
    before = queue.to_dict()
    with pytest.raises(ValueError):
        queue.acquire("w", value)
    with pytest.raises(ValueError):
        queue.acquire("w", 0, value)
    assert queue.to_dict() == before
    lease = queue.acquire("w", 0)
    before = queue.to_dict()
    for operation in (lambda: queue.complete("a", "w", {}, value),
                      lambda: queue.fail("a", "w", "bad", value),
                      lambda: queue.heartbeat("a", "w", value),
                      lambda: queue.heartbeat("a", "w", 0, value),
                      lambda: queue.reclaim_expired(value)):
        with pytest.raises(ValueError):
            operation()
        assert queue.to_dict() == before
    assert lease is not None


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_retry_limits(value):
    with pytest.raises(ValueError):
        RolloutQueue(max_attempts=value)
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0))
    queue.acquire("w", 0)
    with pytest.raises(ValueError):
        queue.fail("a", "w", "bad", 1, max_attempts=value)
    assert "a" in queue.leased


def test_atomic_replace_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    queue = RolloutQueue()
    path = queue.snapshot(tmp_path / "state.json")
    original = path.read_bytes()
    queue.submit(WorkItem("a", {}, 0))

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr("distgpt.eval.rollout_queue.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        queue.snapshot(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_usage_is_cumulative_and_survives_late_reports_and_restore(tmp_path):
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0, domain="math"))
    lease = queue.acquire("w", 0, 1)
    usage = Usage(input_tokens=2, output_tokens=3, wall_s=1, grader_calls=1, grader_cost=0.25)
    queue.record_usage(lease.token, usage)
    queue.reclaim_expired(1)
    queue.record_usage(lease.token, usage)
    queue.record_usage(lease.token, dataclasses.replace(usage, wall_s=2))
    with pytest.raises(ValueError, match="decrease"):
        queue.record_usage(lease.token, usage)
    restored = RolloutQueue.restore(queue.snapshot(tmp_path / "state.json"))
    assert restored.totals() == queue.totals("math")
    assert restored.totals()["grader_cost"] == 0.25
    assert restored.totals()["wall_s"] == 2


def test_restore_rejects_nonfinite_deadline_and_unknown_schema(tmp_path):
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, 0))
    queue.acquire("w", 0)
    data = queue.to_dict()
    data["leased"]["a"]["expires_at"] = float("nan")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        RolloutQueue.restore(path)
    data["schema_version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        RolloutQueue.restore(path)
