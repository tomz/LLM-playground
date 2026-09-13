"""Contract tests for the resumable distributed rollout queue."""
from __future__ import annotations

from distgpt.eval.rollout_queue import RolloutQueue, WorkItem


def test_lease_exclusivity_and_idempotent_completion():
    queue = RolloutQueue()
    assert queue.submit(WorkItem("a", {"prompt": "x"}, created_at=1.0))
    assert not queue.submit(WorkItem("a", {}, created_at=2.0))
    lease = queue.acquire("worker-1", now=10.0, lease_s=5.0)
    assert lease is not None and lease.item.item_id == "a"
    assert queue.acquire("worker-2", now=10.0) is None
    assert not queue.complete("a", "worker-2", {}, now=11.0)
    assert queue.complete("a", "worker-1", {"reward": 1.0}, now=11.0)
    assert queue.complete("a", "worker-1", {"reward": 0.0}, now=12.0)
    assert queue.completed["a"].result == {"reward": 1.0}


def test_expired_lease_retries_and_rejects_stale_worker():
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, created_at=1.0))
    queue.acquire("old", now=0.0, lease_s=1.0)
    lease = queue.acquire("new", now=2.0, lease_s=5.0)
    assert lease is not None and lease.item.attempt == 1
    assert not queue.complete("a", "old", {}, now=2.5)
    assert queue.complete("a", "new", {"ok": True}, now=3.0)


def test_fail_requeues_then_dead_letters():
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, created_at=1.0))
    queue.acquire("w", now=0.0)
    assert queue.fail("a", "w", "transient", now=1.0, max_attempts=2)
    lease = queue.acquire("w", now=2.0)
    assert lease is not None and lease.item.attempt == 1
    assert queue.fail("a", "w", "permanent", now=3.0, max_attempts=2)
    assert queue.failed == {"a": "permanent"}


def test_snapshot_roundtrip_preserves_lineage(tmp_path):
    queue = RolloutQueue()
    queue.submit(WorkItem("child", {"stage": "verify"}, 1.0, parent_id="rollout"))
    queue.acquire("verifier-0", now=2.0, lease_s=10.0)
    restored = RolloutQueue.restore(queue.snapshot(tmp_path / "queue.json"))
    assert restored.to_dict() == queue.to_dict()
    assert restored.leased["child"].item.parent_id == "rollout"


def test_lease_is_expired_at_its_deadline():
    queue = RolloutQueue()
    queue.submit(WorkItem("a", {}, created_at=0.0))
    queue.acquire("old", now=0.0, lease_s=1.0)
    assert not queue.complete("a", "old", {}, now=1.0)
    assert not queue.fail("a", "old", "too late", now=1.0)
    renewed = queue.acquire("new", now=1.0, lease_s=1.0)
    assert renewed is not None and renewed.item.attempt == 1
    assert queue.complete("a", "new", {}, now=1.5)
