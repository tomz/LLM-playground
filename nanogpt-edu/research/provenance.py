"""Schema-versioned provenance for every autonomous-research attempt.

The compact ``ledger.tsv`` remains the chart-friendly summary.  This JSONL
sidecar is the audit record: it preserves failed attempts, exact candidate
identity, budget and measured cost, gate verdict, and any human intervention.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttemptRecord:
    """One attempted experiment, whether kept, discarded, or crashed."""

    experiment: int
    candidate_sha256: str
    git_revision: str
    description: str
    seed: int
    budget_kind: str
    budget_value: float
    status: str
    gate_ok: bool
    gate_reason: str
    tokens: int = 0
    wall_s: float = 0.0
    val_bpb: float | None = None
    verifier_cost_s: float = 0.0
    human_interventions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.experiment <= 0:
            raise ValueError("experiment must be positive")
        if self.budget_kind not in {"tokens", "minutes"}:
            raise ValueError("budget_kind must be 'tokens' or 'minutes'")
        if self.budget_value <= 0:
            raise ValueError("budget_value must be positive")
        if self.status not in {"keep", "discard", "crash"}:
            raise ValueError("invalid attempt status")
        if self.tokens < 0 or self.wall_s < 0 or self.verifier_cost_s < 0:
            raise ValueError("cost fields cannot be negative")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def sha256_file(path: str | Path) -> str:
    """Return the content identity of a candidate file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_attempt(path: str | Path, record: AttemptRecord) -> Path:
    """Append one record as JSONL and return the destination path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record.to_dict(), sort_keys=True))
        stream.write("\n")
    return destination


def read_attempts(path: str | Path) -> Iterator[AttemptRecord]:
    """Stream attempt records, rejecting unsupported future schemas."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            data = json.loads(line)
            version = int(data.get("schema_version", 1))
            if version != SCHEMA_VERSION:
                raise ValueError(f"unsupported attempt schema version {version}")
            yield AttemptRecord(**data)


def total_cost(records: Iterable[AttemptRecord]) -> dict[str, float]:
    """Aggregate attempted work, including failed and discarded branches."""
    rows = list(records)
    return {
        "attempts": float(len(rows)),
        "tokens": float(sum(row.tokens for row in rows)),
        "wall_s": sum(row.wall_s for row in rows),
        "verifier_cost_s": sum(row.verifier_cost_s for row in rows),
    }
