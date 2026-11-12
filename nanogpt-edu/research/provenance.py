"""Versioned research provenance, including failed attempts and unknown work.

Schema 3 permits unknown training-token/verifier costs. V1/v2 remain readable;
unknown values are not interpreted as measured zero or used in cost ratios.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = 3
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "grader_tokens")
USD_FIELDS = ("model_cost_usd", "grader_cost_usd", "other_cost_usd")


def _nonnegative(name: str, value: float, *, integer: bool = False) -> None:
    """Reject non-numbers, booleans, nonfinite values and negative costs."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or (integer and not isinstance(value, int))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative {'integer' if integer else 'number'}")


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt with incremental, not parent-cumulative, costs.

    ``tokens`` denotes reported training tokens, separately from inference work.
    Cache/reasoning counters are subsets, not additive. None means unknown;
    zero means measured zero. Default zero core costs preserve the old constructor
    contract: an instrumented caller must explicitly supply None when unavailable.
    Wall time includes verifier time and must not be added to it again.
    """

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
    tokens: int | None = 0
    wall_s: float = 0.0
    val_bpb: float | None = None
    verifier_cost_s: float | None = 0.0
    human_interventions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    domain: str | None = None
    model_id: str | None = None
    policy_id: str | None = None
    grader_id: str | None = None
    effort: str | None = None
    attempt_id: str | None = None
    parent_attempt_id: str | None = None
    retry_index: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    grader_tokens: int | None = None
    model_cost_usd: float | None = None
    grader_cost_usd: float | None = None
    other_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2, SCHEMA_VERSION}:
            raise ValueError(f"unsupported attempt schema version {self.schema_version}")
        _nonnegative("experiment", self.experiment, integer=True)
        if self.experiment == 0:
            raise ValueError("experiment must be positive")
        if self.budget_kind not in {"tokens", "minutes"}:
            raise ValueError("budget_kind must be 'tokens' or 'minutes'")
        _nonnegative("budget_value", self.budget_value)
        if self.budget_value == 0:
            raise ValueError("budget_value must be positive")
        if self.budget_kind == "tokens" and self.budget_value != int(self.budget_value):
            raise ValueError("token budget must be integral")
        if self.status not in {"keep", "discard", "crash"}:
            raise ValueError("invalid attempt status")
        if type(self.gate_ok) is not bool:
            raise ValueError("gate_ok must be a boolean")
        if self.tokens is not None:
            _nonnegative("tokens", self.tokens, integer=True)
        elif self.schema_version < 3:
            raise ValueError("unknown training tokens require schema 3")
        _nonnegative("wall_s", self.wall_s)
        _nonnegative("retry_index", self.retry_index, integer=True)
        for name in ("verifier_cost_s", *TOKEN_FIELDS, *USD_FIELDS):
            value = getattr(self, name)
            if value is not None:
                _nonnegative(name, value, integer=name in TOKEN_FIELDS)
            elif name == "verifier_cost_s" and self.schema_version < 3:
                raise ValueError("unknown verifier time requires schema 3")
        if self.val_bpb is not None:
            _nonnegative("val_bpb", self.val_bpb)
        for name in ("domain", "model_id", "policy_id", "grader_id", "effort", "attempt_id", "parent_attempt_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or None")
        if self.parent_attempt_id is not None and self.parent_attempt_id == self.attempt_id:
            raise ValueError("an attempt cannot be its own parent")
        if self.retry_index and (self.attempt_id is None or self.parent_attempt_id is None):
            raise ValueError("retries require attempt_id and parent_attempt_id")
        for subset, total in (("cached_input_tokens", "input_tokens"), ("reasoning_tokens", "output_tokens")):
            part, whole = getattr(self, subset), getattr(self, total)
            if part is not None and (whole is None or part > whole):
                raise ValueError(f"{subset} requires {total} and cannot exceed it")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def sha256_file(path: str | Path) -> str:
    """Return the content identity of a candidate file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_attempt(path: str | Path, record: AttemptRecord) -> Path:
    """Append one strictly JSON-serializable attempt."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    return destination


def read_attempts(path: str | Path) -> Iterator[AttemptRecord]:
    """Stream v1/v2/v3 records; an absent version still means v1."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                data = json.loads(line)
                data.setdefault("schema_version", 1)
                yield AttemptRecord(**data)


def total_cost(records: Iterable[AttemptRecord]) -> dict[str, float | None]:
    """Legacy four keys, with null for any unmeasured aggregate in schema 3."""
    rows = list(records)
    return {
        "attempts": float(len(rows)),
        "tokens": float(sum(row.tokens for row in rows)) if all(row.tokens is not None for row in rows) else None,
        "wall_s": sum(row.wall_s for row in rows),
        "verifier_cost_s": (sum(row.verifier_cost_s for row in rows)
                            if all(row.verifier_cost_s is not None for row in rows) else None),
    }


def cost_summary(records: Iterable[AttemptRecord]) -> dict:
    """Amortize all supplied branches over accepted results, without selection.

    Caller must supply the complete campaign. Unknown costs and zero-denominator
    ratios serialize as null. Parent links are not recursively charged.
    """
    rows = list(records)
    ids = [row.attempt_id for row in rows if row.attempt_id is not None]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate attempt_id would double-count costs")
    accepted = sum(row.status == "keep" and row.gate_ok for row in rows)
    totals = total_cost(rows)
    measured = {name: sum(getattr(row, name) is not None for row in rows)
                for name in ("tokens", "wall_s", "verifier_cost_s")}
    for name in (*TOKEN_FIELDS, *USD_FIELDS):
        values = [getattr(row, name) for row in rows]
        measured[name] = sum(value is not None for value in values)
        totals[name] = sum(values) if all(value is not None for value in values) else None
    totals["inference_tokens"] = (
        sum(totals[name] for name in ("input_tokens", "output_tokens", "grader_tokens"))
        if all(totals[name] is not None for name in ("input_tokens", "output_tokens", "grader_tokens")) else None
    )
    totals["cost_usd"] = (
        sum(totals[name] for name in USD_FIELDS) if all(totals[name] is not None for name in USD_FIELDS) else None
    )
    return {
        "accepted_results": accepted, "totals": totals, "measured_attempts": measured,
        "per_accepted_result": {
            name: value / accepted if accepted and value is not None else None
            for name, value in totals.items()
        },
    }
