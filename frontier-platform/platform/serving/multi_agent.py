"""Budgeted multi-agent fan-out and deterministic reconciliation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

Worker = Callable[[str], tuple[str, int]]
Scorer = Callable[[str], float]


@dataclass(frozen=True)
class AgentResult:
    worker: str
    answer: str
    tokens: int
    score: float


@dataclass(frozen=True)
class MultiAgentResult:
    answer: str
    branches: list[AgentResult]
    tokens_used: int
    stopped_reason: str


def run_parallel(
    task: str,
    workers: dict[str, Worker],
    *,
    token_budget: int,
    scorer: Scorer | None = None,
    max_agents: int = 4,
) -> MultiAgentResult:
    """Run branches within a shared budget and reconcile by score or majority."""
    if token_budget <= 0 or max_agents <= 0:
        raise ValueError("positive budgets are required")
    branches: list[AgentResult] = []
    used = 0
    reason = "complete"
    for name in sorted(workers)[:max_agents]:
        answer, tokens = workers[name](task)
        if tokens < 0:
            raise ValueError("worker token count cannot be negative")
        if used + tokens > token_budget:
            reason = "token_budget"
            break
        score = float(scorer(answer)) if scorer else 0.0
        branches.append(AgentResult(name, answer, tokens, score))
        used += tokens
    if not branches:
        return MultiAgentResult("", [], used, reason)
    if scorer:
        winner = max(branches, key=lambda row: (row.score, -row.tokens, row.worker))
    else:
        counts = Counter(row.answer for row in branches)
        winner = max(branches, key=lambda row: (counts[row.answer], -row.tokens, row.worker))
    return MultiAgentResult(winner.answer, branches, used, reason)
