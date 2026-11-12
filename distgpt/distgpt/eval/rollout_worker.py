"""Bounded synchronous rollout/checker callbacks and a dependency-free CPU demo.

Callbacks must cooperate with the meter and return promptly. Wall limits are
checked between operations, not enforced by preempting Python callbacks.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .rollout_queue import Lease, RolloutQueue, Usage, WorkItem, count, finite

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """The next operation would exceed a run-wide allowance."""


@dataclass(frozen=True)
class Budget:
    """Lifetime queue allowances, including failed/stale attempts after restore."""

    max_attempts: int = 20
    max_tokens: int = 1000
    max_wall_s: float = 10.0
    max_grader_calls: int = 20
    max_grader_cost: float = 20.0

    def __post_init__(self) -> None:
        for name in ("max_attempts", "max_tokens", "max_grader_calls"):
            count(getattr(self, name), name, positive=True)
        for name in ("max_wall_s", "max_grader_cost"):
            finite(getattr(self, name), name, positive=True)

    def exceeded(self, totals: dict, *, admission: bool = False) -> bool:
        pairs = (
            (totals["attempts"], self.max_attempts),
            (totals["input_tokens"] + totals["output_tokens"], self.max_tokens),
            (totals["wall_s"], self.max_wall_s),
            (totals["grader_calls"], self.max_grader_calls),
            (totals["grader_cost"], self.max_grader_cost),
        )
        return any(value >= limit if admission else value > limit for value, limit in pairs)


@dataclass(frozen=True)
class Generation:
    text: str
    policy_version: int
    logprob_version: int
    log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        count(self.policy_version, "policy_version")
        count(self.logprob_version, "logprob_version")
        if any(not math.isfinite(p) or p > 0 for p in self.log_probs):
            raise ValueError("log probabilities must be finite and nonpositive")


@dataclass(frozen=True)
class StalenessPolicy:
    current_version: int = 0
    max_lag: int = 0

    def __post_init__(self) -> None:
        count(self.current_version, "current_version")
        count(self.max_lag, "max_lag")

    def validate(self, item: WorkItem, generation: Generation) -> None:
        """Require exact requested provenance and bounded, nonnegative policy lag."""
        if generation.policy_version != item.policy_version:
            raise ValueError("generator policy differs from requested policy")
        if generation.logprob_version != item.logprob_version:
            raise ValueError("log-prob version differs from requested version")
        if generation.logprob_version != generation.policy_version:
            raise ValueError("log-probs do not belong to the behavior policy")
        lag = self.current_version - generation.policy_version
        if lag < 0 or lag > self.max_lag:
            raise ValueError("policy lag outside accepted range")


class Meter:
    """Cooperative accounting: reserve tokens before generating/processing them."""

    def __init__(self, queue: RolloutQueue, lease: Lease, budget: Budget,
                 timer: Callable[[], float]) -> None:
        self.queue = queue
        self.lease = lease
        self.budget = budget
        self.timer = timer
        self.started = timer()

    @property
    def usage(self) -> Usage:
        return self.queue.attempts[self.lease.token].usage

    def checkpoint(self, *, enforce: bool = True) -> None:
        elapsed = self.timer() - self.started
        finite(elapsed, "elapsed wall time")
        self.queue.record_usage(self.lease.token, dataclasses.replace(
            self.usage, wall_s=max(self.usage.wall_s, elapsed),
        ))
        if enforce and self.budget.exceeded(self.queue.totals()):
            raise BudgetExceeded("wall or cumulative budget exceeded")

    def _charge(self, **increments: int | float) -> None:
        self.checkpoint()
        updated = dataclasses.replace(self.usage, **{
            key: getattr(self.usage, key) + value for key, value in increments.items()
        })
        totals = self.queue.totals()
        for key, value in increments.items():
            totals[key] += value
        if self.budget.exceeded(totals):
            raise BudgetExceeded("operation would exceed cumulative budget")
        self.queue.record_usage(self.lease.token, updated)

    def tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        count(input_tokens, "input_tokens")
        count(output_tokens, "output_tokens")
        self._charge(input_tokens=input_tokens, output_tokens=output_tokens)

    def grader(self, cost: float) -> None:
        finite(cost, "grader cost")
        self._charge(grader_calls=1, grader_cost=cost)


Generator = Callable[[WorkItem, Meter], Generation]
Verifier = Callable[[WorkItem, Generation, Meter], dict]


class LocalWorker:
    """Execute real callbacks serially; never share this queue between processes."""

    def __init__(self, queue: RolloutQueue, generator: Generator, verifier: Verifier,
                 *, worker_id: str = "local", budget: Budget | None = None,
                 staleness: StalenessPolicy | None = None, lease_s: float = 60.0,
                 grader_cost: float = 1.0, clock: Callable[[], float] = time.time,
                 timer: Callable[[], float] = time.monotonic) -> None:
        finite(lease_s, "lease_s", positive=True)
        finite(grader_cost, "grader_cost")
        if not worker_id:
            raise ValueError("worker_id is required")
        self.queue = queue
        self.generator = generator
        self.verifier = verifier
        self.worker_id = worker_id
        self.budget = budget or Budget()
        self.staleness = staleness or StalenessPolicy()
        self.lease_s = lease_s
        self.grader_cost = grader_cost
        self.clock = clock
        self.timer = timer

    def run_one(self) -> str | None:
        """Return the attempt token, or None when idle or budget-limited."""
        if self.budget.exceeded(self.queue.totals(), admission=True):
            return None
        lease = self.queue.acquire(self.worker_id, self.clock(), self.lease_s)
        if lease is None:
            return None
        meter = Meter(self.queue, lease, self.budget, self.timer)
        record = self.queue.attempts[lease.token]
        try:
            meter.checkpoint()
            generation = self.generator(lease.item, meter)
            record.result_policy_version = generation.policy_version
            record.result_logprob_version = generation.logprob_version
            meter.checkpoint()
            self.staleness.validate(lease.item, generation)
            if len(generation.log_probs) != meter.usage.output_tokens:
                raise ValueError("output token accounting must match log-prob count")
            meter.grader(self.grader_cost)
            verdict = self.verifier(lease.item, generation, meter)
            if not isinstance(verdict, dict):
                raise ValueError("verifier must return a JSON object")
            meter.checkpoint()
            # Recheck at commit, since a callback may advance the learner policy.
            self.staleness.validate(lease.item, generation)
            result = {"generation": dataclasses.asdict(generation), "verdict": verdict}
            json.dumps(result, allow_nan=False)
            accepted = self.queue.complete(lease.item.item_id, self.worker_id, result,
                                           self.clock(), lease_token=lease.token,
                                           attempt=lease.item.attempt)
            if not accepted:
                record.status = "stale"
                record.reason = "lease no longer owns result"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            accepted = self.queue.fail(lease.item.item_id, self.worker_id, reason,
                                       self.clock(), lease_token=lease.token,
                                       attempt=lease.item.attempt)
            if not accepted:
                record.status = "stale"
                record.reason = reason
        finally:
            # Includes checker/generator exceptions and cooperative interruptions.
            # A process killed outright can only retain its last saved checkpoint.
            meter.checkpoint(enforce=False)
        return lease.token

    def run(self) -> dict:
        """Drain available jobs without waiting, bounded by lifetime allowances."""
        while self.run_one() is not None:
            pass
        return {
            "completed": len(self.queue.completed), "failed": len(self.queue.failed),
            "pending": len(self.queue.pending), "leased": len(self.queue.leased),
            "budget_exhausted": self.budget.exceeded(self.queue.totals(), admission=True),
            "totals": self.queue.totals(),
            "domains": {domain: self.queue.totals(domain) for domain in sorted(
                {r.domain for r in self.queue.attempts.values()}
            )},
        }


def toy_generator(item: WorkItem, meter: Meter) -> Generation:
    """Emit a deterministic integer sequence; its selected-token log-probs are 0."""
    length = item.payload["length"]
    count(length, "length", positive=True)
    meter.tokens(input_tokens=1)
    tokens: list[str] = []
    for value in range(length):
        meter.tokens(output_tokens=1)
        tokens.append(str(value))
    return Generation(" ".join(tokens), item.policy_version, item.logprob_version,
                      (0.0,) * length)


def toy_checker(item: WorkItem, generation: Generation, meter: Meter) -> dict:
    """Check every emitted integer, rather than returning a canned reward."""
    meter.checkpoint()
    expected = [str(value) for value in range(item.payload["length"])]
    return {"correct": generation.text.split() == expected, "checked_tokens": len(expected)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("out/rollout-queue.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-wall-s", type=float, default=10.0)
    parser.add_argument("--max-grader-calls", type=int, default=20)
    parser.add_argument("--max-grader-cost", type=float, default=20.0)
    parser.add_argument("--lease-s", type=float, default=60.0)
    parser.add_argument("--current-policy-version", type=int, default=0)
    parser.add_argument("--max-policy-lag", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        budget = Budget(args.max_attempts, args.max_tokens, args.max_wall_s,
                        args.max_grader_calls, args.max_grader_cost)
        staleness = StalenessPolicy(args.current_policy_version, args.max_policy_lag)
        queue = RolloutQueue.restore(args.snapshot) if args.resume else RolloutQueue()
        if not args.resume:
            queue.submit(WorkItem("short", {"length": 3}, 0.0, domain="short"))
            queue.submit(WorkItem("long", {"length": 64}, 1.0, domain="long"))
        worker = LocalWorker(queue, toy_generator, toy_checker, budget=budget,
                             staleness=staleness, lease_s=args.lease_s)
        summary = worker.run()
        queue.snapshot(args.snapshot)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    logger.info(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
