"""Exact task-cost accounting for bounded routing experiments, not billing software.

Token buckets are disjoint. Output includes reasoning tokens; reasoning is a
subset for attribution, never an additional charge. Unknown usage stays unknown
and retains its admission reservation instead of being reported as free work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
import json
from pathlib import Path


ZERO = Decimal("0")
MILLION = Decimal("1000000")


def amount(value: Decimal | str | int) -> Decimal:
    """Normalize a finite nonnegative monetary amount without binary rounding."""
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("money must be finite and nonnegative")
    return result


def count(value: int) -> None:
    """Reject bool, fractional, and negative token counts."""
    if type(value) is not int or value < 0:
        raise ValueError("token counts must be nonnegative integers")


@dataclass(frozen=True)
class Usage:
    """Disjoint billed buckets plus a non-additive reasoning subset."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    tool_usd: Decimal = ZERO

    def __post_init__(self) -> None:
        for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens",
                     "output_tokens", "reasoning_tokens"):
            count(getattr(self, name))
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("reasoning tokens must be included in output tokens")
        object.__setattr__(self, "tool_usd", amount(self.tool_usd))

    @property
    def tokens(self) -> int:
        return (self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
                + self.output_tokens)

    def fits(self, ceiling: "Usage") -> bool:
        """Check every cost-bearing bucket, not only total tokens."""
        return all(getattr(self, key) <= getattr(ceiling, key) for key in (
            "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "tool_usd",
        ))


@dataclass(frozen=True)
class Tariff:
    """Caller-pinned rate revision; input/output prices are USD per million."""

    tariff_id: str
    model: str
    source: str
    valid_from: str
    input_usd: Decimal
    output_usd: Decimal
    cache_read_usd: Decimal = ZERO
    cache_write_usd: Decimal = ZERO
    valid_through: str | None = None

    def __post_init__(self) -> None:
        if not all((self.tariff_id, self.model, self.source)):
            raise ValueError("tariff identity, model, and provenance are required")
        start = date.fromisoformat(self.valid_from)
        if self.valid_through and date.fromisoformat(self.valid_through) < start:
            raise ValueError("tariff validity interval is reversed")
        for name in ("input_usd", "output_usd", "cache_read_usd", "cache_write_usd"):
            object.__setattr__(self, name, amount(getattr(self, name)))

    def check_date(self, observed_on: str) -> None:
        day = date.fromisoformat(observed_on)
        if day < date.fromisoformat(self.valid_from) or (
            self.valid_through and day > date.fromisoformat(self.valid_through)
        ):
            raise ValueError("tariff is not valid on the request date")

    def cost(self, usage: Usage) -> Decimal:
        return (self.input_usd * usage.input_tokens
                + self.cache_read_usd * usage.cache_read_tokens
                + self.cache_write_usd * usage.cache_write_tokens
                + self.output_usd * usage.output_tokens) / MILLION + usage.tool_usd


@dataclass(frozen=True)
class TaskAttempt:
    """One dispatched or denied branch; identity and unknown cost are explicit."""

    attempt_id: str
    task_id: str
    domain: str
    requested_model: str
    model_revision: str
    actual_model: str | None
    actual_revision: str | None
    effort: str
    safeguard_mode: str
    actual_safeguard_mode: str | None
    grader_revision: str
    tariff: Tariff
    outcome: str
    usage: Usage | None
    reserved_usage: Usage
    wall_s: float
    verifier_s: float = 0.0
    parent_attempt_id: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        import math
        if not all((self.attempt_id, self.task_id, self.domain, self.requested_model,
                    self.model_revision, self.effort, self.safeguard_mode, self.grader_revision)):
            raise ValueError("attempt identities and provenance are required")
        if self.requested_model != self.tariff.model:
            raise ValueError("tariff model does not match requested model")
        if self.outcome not in {
            "accepted", "rejected", "backend_error", "checker_error", "cancelled",
            "denied", "budget", "refused", "incomplete", "contract_error",
        }:
            raise ValueError("unknown attempt outcome")
        if any(not math.isfinite(x) or x < 0 for x in (self.wall_s, self.verifier_s)):
            raise ValueError("durations must be finite and nonnegative")
        if self.outcome == "accepted" and self.usage is None:
            raise ValueError("accepted result requires observed usage")

    @property
    def known_cost_usd(self) -> Decimal | None:
        # A substituted model cannot be priced with the requested model's tariff.
        if self.usage is None or self.actual_model not in (None, self.requested_model):
            return None
        return self.tariff.cost(self.usage)

    @property
    def accounted_usd(self) -> Decimal:
        cost = self.known_cost_usd
        return self.tariff.cost(self.reserved_usage) if cost is None else cost

    @property
    def accounted_tokens(self) -> int:
        return (self.usage or self.reserved_usage).tokens


@dataclass
class TaskLedger:
    """Append-only in-process ledger; production durability remains external."""

    attempts: list[TaskAttempt] = field(default_factory=list)

    def append(self, attempt: TaskAttempt) -> None:
        if any(row.attempt_id == attempt.attempt_id for row in self.attempts):
            raise ValueError("duplicate attempt id")
        self.attempts.append(attempt)

    @property
    def accounted_usd(self) -> Decimal:
        return sum((row.accounted_usd for row in self.attempts), ZERO)

    @property
    def accounted_tokens(self) -> int:
        return sum(row.accounted_tokens for row in self.attempts)

    def summary(self, domain: str | None = None) -> dict:
        rows = [row for row in self.attempts if domain is None or row.domain == domain]
        known = sum((row.known_cost_usd for row in rows if row.known_cost_usd is not None), ZERO)
        accepted = len({row.task_id for row in rows if row.outcome == "accepted"})
        unknown = sum(row.known_cost_usd is None for row in rows)
        return {
            "attempts": len(rows), "accepted_tasks": accepted,
            "known_cost_usd": str(known), "unknown_cost_attempts": unknown,
            "accounted_usd": str(sum((row.accounted_usd for row in rows), ZERO)),
            "accounted_tokens": sum(row.accounted_tokens for row in rows),
            "wall_s": sum(row.wall_s for row in rows),
            "verifier_s": sum(row.verifier_s for row in rows),
            "cost_per_accepted_task_usd": str(known / accepted) if accepted and not unknown else None,
        }

    def write_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            for row in self.attempts:
                stream.write(json.dumps({"schema_version": 1, **asdict(row)},
                                        default=str, sort_keys=True, allow_nan=False) + "\n")
        return destination

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "TaskLedger":
        result = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if data.pop("schema_version", None) != 1:
                raise ValueError("unsupported task ledger schema")
            data["tariff"] = Tariff(**data["tariff"])
            data["reserved_usage"] = Usage(**data["reserved_usage"])
            if data["usage"] is not None:
                data["usage"] = Usage(**data["usage"])
            result.append(TaskAttempt(**data))
        return result
