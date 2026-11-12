"""Sequential reference fan-out with admission and honest attempted-work costs.

Reserved workers receive an explicit ceiling they must enforce themselves.
Legacy callbacks have no pre-dispatch bound: their overruns remain visible.
Neither form is a process sandbox, preemptive timeout, or concurrent budget bank.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable

Worker = Callable[[str], tuple[str, int]]
Scorer = Callable[[str], float]


def _positive_count(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("budgets must be positive integers")


@dataclass(frozen=True)
class ReservedWorker:
    """Caller-declared upper bound, passed to a cooperative execution adapter."""

    max_tokens: int
    run: Callable[[str, int], tuple[str, int]]

    def __post_init__(self) -> None:
        _positive_count(self.max_tokens)
        if not callable(self.run):
            raise ValueError("worker must be callable")


@dataclass(frozen=True)
class AgentResult:
    worker: str
    answer: str
    tokens: int | None
    score: float
    status: str = "completed"
    error: str | None = None
    elapsed_seconds: float = 0.0
    reserved_tokens: int | None = None


@dataclass(frozen=True)
class MultiAgentResult:
    answer: str
    branches: list[AgentResult]
    tokens_used: int | None
    stopped_reason: str
    measured_tokens: int = 0


def run_parallel(
    task: str,
    workers: dict[str, Worker | ReservedWorker],
    *,
    token_budget: int,
    scorer: Scorer | None = None,
    max_agents: int = 4,
    cancelled: Callable[[], bool] = lambda: False,
) -> MultiAgentResult:
    """Reconcile completed branches, retaining every attempted branch's cost.

    The historical name does not imply concurrent execution. Unknown usage makes
    ``tokens_used`` unavailable while ``measured_tokens`` keeps the known subtotal.
    Cancellation is cooperative and cannot interrupt a hung callback.
    """
    _positive_count(token_budget)
    _positive_count(max_agents)
    if any(not isinstance(name, str) or not name for name in workers):
        raise ValueError("worker names must be nonempty strings")
    branches: list[AgentResult] = []
    used = 0
    unknown = False
    reason = "complete"
    for name in sorted(workers)[:max_agents]:
        worker = workers[name]
        reservation = worker.max_tokens if isinstance(worker, ReservedWorker) else None
        if cancelled():
            reason = "cancelled"
            break
        if used >= token_budget or (reservation is not None and used + reservation > token_budget):
            reason = "token_budget"
            break
        started = perf_counter()
        tokens: int | None = None
        answer = ""
        score = 0.0
        error = None
        status = "completed"
        try:
            output = worker.run(task, reservation) if isinstance(worker, ReservedWorker) else worker(task)
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("worker must return (answer, measured_tokens)")
            candidate, count = output
            if type(count) is not int or count < 0:
                unknown = True
                status, reason = "unknown", "unknown_usage"
                error = "worker usage is missing or invalid"
            else:
                tokens = count
                used += count  # Charge observed work BEFORE any acceptance/scoring checks.
                if not isinstance(candidate, str):
                    raise ValueError("worker answer must be text")
                answer = candidate
                if used > token_budget:
                    status, reason = "overrun", "token_budget"
                elif reservation is not None and count > reservation:
                    status, reason = "overrun", "reservation_overrun"
                elif cancelled():
                    status, reason = "cancelled", "cancelled"
                elif scorer is not None:
                    try:
                        score = float(scorer(answer))
                        if not math.isfinite(score):
                            raise ValueError("score must be finite")
                    except Exception as exc:
                        score = 0.0
                        status, reason = "error", "scorer_error"
                        error = f"{type(exc).__name__}: {exc}"[:1000]
        except Exception as exc:
            unknown = unknown or tokens is None
            status, reason = "error", "worker_error"
            error = f"{type(exc).__name__}: {exc}"[:1000]
        branches.append(AgentResult(
            name, answer, tokens, score, status, error,
            perf_counter() - started, reservation,
        ))
        if status != "completed":
            break
    candidates = [row for row in branches if row.status == "completed"]
    answer = ""
    if candidates:
        if scorer:
            winner = max(candidates, key=lambda row: (row.score, -row.tokens, row.worker))
        else:
            counts = Counter(row.answer for row in candidates)
            winner = max(candidates, key=lambda row: (counts[row.answer], -row.tokens, row.worker))
        answer = winner.answer
    return MultiAgentResult(answer, branches, None if unknown else used, reason, used)
