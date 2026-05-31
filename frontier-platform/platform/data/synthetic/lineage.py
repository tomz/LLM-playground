"""Per-sample lineage records for synthetic data.

Every sample the factory considers — accepted *or* rejected — becomes a
:class:`SampleRecord`. Records carry enough metadata (teacher, policy, seed,
verifier score, rejection reason) to fully reproduce the run and to audit the
dataset for contamination, license issues, or distribution drift downstream.

The on-disk format is JSONL: one record per line, schema-versioned in the
first field. ``read_lineage_jsonl`` is a cheap streaming reader so a hundred
million-record file doesn't have to be held in memory.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SampleRecord:
    """One synthetic sample, accepted or rejected."""

    sample_id: str
    teacher: str
    policy: str
    seed: int
    prompt: str
    response: str
    accepted: bool
    verifier_score: float = 0.0
    rejection_reason: str | None = None
    meta: dict = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_lineage_jsonl(
    records: Iterable[SampleRecord],
    path: str | Path,
    *,
    only_accepted: bool = False,
) -> Path:
    """Stream ``records`` to ``path`` as JSONL; return the path.

    Set ``only_accepted=True`` to emit a clean training file (the rejected
    records still belong in a *full* lineage file for audit).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            if only_accepted and not r.accepted:
                continue
            f.write(json.dumps(r.to_dict(), ensure_ascii=False))
            f.write("\n")
    return p


def read_lineage_jsonl(path: str | Path) -> Iterator[SampleRecord]:
    """Stream :class:`SampleRecord`s out of a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield SampleRecord(
                sample_id=data["sample_id"],
                teacher=data["teacher"],
                policy=data["policy"],
                seed=int(data["seed"]),
                prompt=data["prompt"],
                response=data["response"],
                accepted=bool(data["accepted"]),
                verifier_score=float(data.get("verifier_score", 0.0)),
                rejection_reason=data.get("rejection_reason"),
                meta=data.get("meta", {}),
                schema_version=int(data.get("schema_version", _SCHEMA_VERSION)),
            )
