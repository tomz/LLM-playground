"""Resumable lease queue for distributed rollout and verifier workers.

The in-memory implementation is intentionally small but enforces the production
contract: deterministic ordering, exclusive leases, idempotent completion,
retry after lease expiry, lineage, and JSON snapshots.  A Redis/etcd backend can
implement the same API without changing trainer semantics.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    payload: dict
    created_at: float
    attempt: int = 0
    parent_id: str | None = None


@dataclass(frozen=True)
class Lease:
    item: WorkItem
    worker_id: str
    expires_at: float


@dataclass(frozen=True)
class Completion:
    item_id: str
    worker_id: str
    result: dict
    completed_at: float
    attempt: int


@dataclass
class RolloutQueue:
    pending: list[WorkItem] = field(default_factory=list)
    leased: dict[str, Lease] = field(default_factory=dict)
    completed: dict[str, Completion] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    def submit(self, item: WorkItem) -> bool:
        """Submit once; duplicate IDs are harmless and return ``False``."""
        known = {row.item_id for row in self.pending} | self.leased.keys() \
            | self.completed.keys() | self.failed.keys()
        if item.item_id in known:
            return False
        self.pending.append(item)
        self.pending.sort(key=lambda row: (row.created_at, row.item_id))
        return True

    def reclaim_expired(self, now: float) -> list[str]:
        """Return expired leases to the queue with an incremented attempt."""
        reclaimed: list[str] = []
        for item_id, lease in sorted(list(self.leased.items())):
            if lease.expires_at > now:
                continue
            self.leased.pop(item_id)
            item = dataclasses.replace(lease.item, attempt=lease.item.attempt + 1)
            self.pending.append(item)
            reclaimed.append(item_id)
        self.pending.sort(key=lambda row: (row.created_at, row.item_id))
        return reclaimed

    def acquire(self, worker_id: str, now: float, lease_s: float = 60.0) -> Lease | None:
        if not worker_id or lease_s <= 0:
            raise ValueError("worker_id and positive lease_s are required")
        self.reclaim_expired(now)
        if not self.pending:
            return None
        item = self.pending.pop(0)
        lease = Lease(item, worker_id, now + lease_s)
        self.leased[item.item_id] = lease
        return lease

    def complete(self, item_id: str, worker_id: str, result: dict, now: float) -> bool:
        """Commit a result once; stale or duplicate workers cannot overwrite it."""
        if item_id in self.completed:
            return self.completed[item_id].worker_id == worker_id
        lease = self.leased.get(item_id)
        if lease is None or lease.worker_id != worker_id or lease.expires_at <= now:
            return False
        self.leased.pop(item_id)
        self.completed[item_id] = Completion(
            item_id, worker_id, result, now, lease.item.attempt,
        )
        return True

    def fail(self, item_id: str, worker_id: str, reason: str, now: float,
             max_attempts: int = 3) -> bool:
        lease = self.leased.get(item_id)
        if lease is None or lease.worker_id != worker_id or lease.expires_at <= now:
            return False
        self.leased.pop(item_id)
        if lease.item.attempt + 1 >= max_attempts:
            self.failed[item_id] = reason
        else:
            self.pending.append(dataclasses.replace(
                lease.item, attempt=lease.item.attempt + 1,
            ))
            self.pending.sort(key=lambda row: (row.created_at, row.item_id))
        return True

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "pending": [dataclasses.asdict(row) for row in self.pending],
            "leased": {key: dataclasses.asdict(value) for key, value in self.leased.items()},
            "completed": {key: dataclasses.asdict(value) for key, value in self.completed.items()},
            "failed": dict(self.failed),
        }

    def snapshot(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
        return destination

    @classmethod
    def restore(cls, path: str | Path) -> "RolloutQueue":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("unsupported rollout queue schema")
        queue = cls(pending=[WorkItem(**row) for row in data["pending"]])
        queue.leased = {
            key: Lease(item=WorkItem(**row["item"]), worker_id=row["worker_id"],
                       expires_at=float(row["expires_at"]))
            for key, row in data["leased"].items()
        }
        queue.completed = {
            key: Completion(**row) for key, row in data["completed"].items()
        }
        queue.failed = dict(data["failed"])
        return queue
