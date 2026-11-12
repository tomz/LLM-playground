"""Single-process lease state machine with fenced retries and JSON recovery.

This is not a concurrent queue or a durable distributed database. Snapshots are
atomic checkpoints, not a transaction log; callers serialize all operations.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def finite(value: float, name: str, *, positive: bool = False) -> None:
    """Reject non-finite, negative, and (when requested) zero quantities."""
    if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def count(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    payload: dict
    created_at: float
    attempt: int = 0
    parent_id: str | None = None
    domain: str = "general"
    policy_version: int = 0
    logprob_version: int = 0

    def __post_init__(self) -> None:
        if not self.item_id or not self.domain:
            raise ValueError("item_id and domain are required")
        finite(self.created_at, "created_at")
        for name in ("attempt", "policy_version", "logprob_version"):
            count(getattr(self, name), name)


@dataclass(frozen=True)
class Lease:
    item: WorkItem
    worker_id: str
    expires_at: float
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    legacy_safe: bool = False

    def __post_init__(self) -> None:
        finite(self.expires_at, "expires_at")
        if not self.worker_id or not self.token:
            raise ValueError("worker_id and token are required")


@dataclass(frozen=True)
class Completion:
    item_id: str
    worker_id: str
    result: dict
    completed_at: float
    attempt: int
    token: str = ""
    legacy_safe: bool = False

    def __post_init__(self) -> None:
        finite(self.completed_at, "completed_at")
        count(self.attempt, "attempt")


@dataclass(frozen=True)
class Usage:
    """Cumulative measured/reported usage for one acquisition, never a delta."""

    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    grader_calls: int = 0
    grader_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "grader_calls"):
            count(getattr(self, name), name)
        finite(self.wall_s, "wall_s")
        finite(self.grader_cost, "grader_cost")


@dataclass
class AttemptRecord:
    item_id: str
    worker_id: str
    attempt: int
    token: str
    domain: str
    policy_version: int
    logprob_version: int
    usage: Usage = field(default_factory=Usage)
    status: str = "leased"
    reason: str = ""
    result_policy_version: int | None = None
    result_logprob_version: int | None = None


@dataclass
class RolloutQueue:
    pending: list[WorkItem] = field(default_factory=list)
    leased: dict[str, Lease] = field(default_factory=dict)
    completed: dict[str, Completion] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    attempts: dict[str, AttemptRecord] = field(default_factory=dict)
    max_attempts: int = 3

    def __post_init__(self) -> None:
        count(self.max_attempts, "max_attempts", positive=True)

    def _sort(self) -> None:
        self.pending.sort(key=lambda row: (row.created_at, row.item_id))

    def submit(self, item: WorkItem) -> bool:
        """Submit once; duplicate IDs are harmless and return ``False``."""
        known = {row.item_id for row in self.pending} | self.leased.keys() \
            | self.completed.keys() | self.failed.keys()
        if item.item_id in known:
            return False
        self.pending.append(item)
        self._sort()
        return True

    def _retry(self, lease: Lease, reason: str, max_attempts: int) -> None:
        if lease.item.attempt + 1 >= max_attempts:
            self.failed[lease.item.item_id] = reason
        else:
            self.pending.append(dataclasses.replace(lease.item, attempt=lease.item.attempt + 1))
            self._sort()

    def reclaim_expired(self, now: float) -> list[str]:
        """Requeue expired work, or dead-letter it at the retry limit."""
        finite(now, "now")
        reclaimed: list[str] = []
        for item_id, lease in sorted(list(self.leased.items())):
            if lease.expires_at > now:
                continue
            self.leased.pop(item_id)
            self.attempts[lease.token].status = "expired"
            self.attempts[lease.token].reason = "lease expired"
            self._retry(lease, "lease expired", self.max_attempts)
            reclaimed.append(item_id)
        return reclaimed

    def acquire(self, worker_id: str, now: float, lease_s: float = 60.0) -> Lease | None:
        finite(now, "now")
        finite(lease_s, "lease_s", positive=True)
        finite(now + lease_s, "lease deadline")
        if not worker_id:
            raise ValueError("worker_id is required")
        self.reclaim_expired(now)
        if not self.pending:
            return None
        item = self.pending.pop(0)
        lease = Lease(item, worker_id, now + lease_s, legacy_safe=item.attempt == 0)
        self.leased[item.item_id] = lease
        self.attempts[lease.token] = AttemptRecord(
            item.item_id, worker_id, item.attempt, lease.token, item.domain,
            item.policy_version, item.logprob_version,
        )
        return lease

    @staticmethod
    def _matches(owner: Lease | Completion, worker_id: str,
                 lease_token: str | None, attempt: int | None) -> bool:
        expected_attempt = owner.item.attempt if isinstance(owner, Lease) else owner.attempt
        if owner.worker_id != worker_id:
            return False
        if lease_token is not None and lease_token != owner.token:
            return False
        if attempt is not None:
            count(attempt, "attempt")
            if attempt != expected_attempt:
                return False
        # An attempt number is not an acquisition identity: restoring a pending
        # checkpoint can issue a different lease at the same attempt number.
        # Keep legacy fields readable, but never authorize mutations with them.
        return bool(owner.token) and lease_token is not None

    def _active(self, item_id: str, worker_id: str, now: float,
                lease_token: str | None, attempt: int | None) -> Lease | None:
        finite(now, "now")
        lease = self.leased.get(item_id)
        if lease and lease.expires_at > now and self._matches(lease, worker_id, lease_token, attempt):
            return lease
        return None

    def heartbeat(self, item_id: str, worker_id: str, now: float, lease_s: float = 60.0,
                  *, lease_token: str | None = None, attempt: int | None = None) -> bool:
        """Extend a live lease without reviving or shortening it."""
        finite(lease_s, "lease_s", positive=True)
        finite(now + lease_s, "lease deadline")
        lease = self._active(item_id, worker_id, now, lease_token, attempt)
        if lease is None:
            return False
        self.leased[item_id] = dataclasses.replace(lease, expires_at=max(lease.expires_at, now + lease_s))
        return True

    def complete(self, item_id: str, worker_id: str, result: dict, now: float,
                 *, lease_token: str | None = None, attempt: int | None = None) -> bool:
        """Commit once; duplicate acknowledgments require the original fence."""
        finite(now, "now")
        if item_id in self.completed:
            return self._matches(self.completed[item_id], worker_id, lease_token, attempt)
        lease = self._active(item_id, worker_id, now, lease_token, attempt)
        if lease is None:
            return False
        self.leased.pop(item_id)
        self.completed[item_id] = Completion(
            item_id, worker_id, result, now, lease.item.attempt, lease.token, lease.legacy_safe,
        )
        self.attempts[lease.token].status = "completed"
        return True

    def fail(self, item_id: str, worker_id: str, reason: str, now: float,
             max_attempts: int | None = None, *, lease_token: str | None = None,
             attempt: int | None = None) -> bool:
        limit = self.max_attempts if max_attempts is None else max_attempts
        count(limit, "max_attempts", positive=True)
        lease = self._active(item_id, worker_id, now, lease_token, attempt)
        if lease is None:
            return False
        self.leased.pop(item_id)
        self.attempts[lease.token].status = "failed"
        self.attempts[lease.token].reason = reason
        self._retry(lease, reason, limit)
        return True

    def record_usage(self, lease_token: str, usage: Usage) -> None:
        """Keep costs even for late attempts; repeated reports do not double count."""
        record = self.attempts[lease_token]
        if any(getattr(usage, f.name) < getattr(record.usage, f.name) for f in dataclasses.fields(Usage)):
            raise ValueError("cumulative usage cannot decrease")
        record.usage = usage

    def totals(self, domain: str | None = None) -> dict:
        records = [r for r in self.attempts.values() if domain is None or r.domain == domain]
        return {
            "attempts": len(records),
            **{f.name: sum(getattr(r.usage, f.name) for r in records) for f in dataclasses.fields(Usage)},
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": 2,
            "max_attempts": self.max_attempts,
            "pending": [dataclasses.asdict(row) for row in self.pending],
            "leased": {key: dataclasses.asdict(value) for key, value in self.leased.items()},
            "completed": {key: dataclasses.asdict(value) for key, value in self.completed.items()},
            "failed": dict(self.failed),
            "attempts": {key: dataclasses.asdict(value) for key, value in self.attempts.items()},
        }

    def snapshot(self, path: str | Path) -> Path:
        """Fsync and replace in the same directory; failed writes keep the old file."""
        serialized = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             prefix=f".{destination.name}.", delete=False) as handle:
                temporary = handle.name
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return destination

    @classmethod
    def restore(cls, path: str | Path) -> RolloutQueue:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version not in (1, 2):
            raise ValueError("unsupported rollout queue schema")
        queue = cls(pending=[WorkItem(**row) for row in data["pending"]],
                    max_attempts=data.get("max_attempts", 3))
        queue.failed = dict(data["failed"])
        queue.completed = {key: Completion(**row) for key, row in data["completed"].items()}
        if version == 2:
            queue.attempts = {
                key: AttemptRecord(**{**row, "usage": Usage(**row["usage"])})
                for key, row in data["attempts"].items()
            }
        for key, row in data["leased"].items():
            lease = Lease(**{**row, "item": WorkItem(**row["item"])})
            if key != lease.item.item_id:
                raise ValueError("lease item ID does not match snapshot key")
            if version == 1:
                # There is no persisted token to authenticate old workers. Force
                # a new attempt; old worker-id and attempt=0 messages stay stale.
                queue._retry(lease, "v1 lease invalidated on migration", queue.max_attempts)
            else:
                if lease.token not in queue.attempts:
                    raise ValueError("missing lease accounting record")
                queue.leased[key] = lease
        queue._sort()
        ids = [row.item_id for row in queue.pending] + list(queue.leased) + list(queue.completed) + list(queue.failed)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate item IDs in snapshot")
        return queue
