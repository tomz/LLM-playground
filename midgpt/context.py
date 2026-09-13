"""Retained agent state and deterministic context-compaction policies.

This is backend-agnostic: production can replace ``summarize`` with an LLM,
while tests and evaluations use a deterministic summarizer.  Unlike rolling
truncation, compaction preserves explicit facts, decisions, and open constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

Summarizer = Callable[[list["Turn"]], str]


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    reasoning: str | None = None
    pinned: bool = False

    @property
    def tokens(self) -> int:
        # Stable dependency-free estimate for policy comparisons.
        return max(1, (len(self.content) + len(self.reasoning or "") + 3) // 4)


@dataclass
class ContextState:
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    compacted_turns: int = 0

    @property
    def tokens(self) -> int:
        return max(0, (len(self.summary) + 3) // 4) + sum(t.tokens for t in self.turns)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)


def retain_reasoning(turn: Turn, *, retain: bool = True) -> Turn:
    """Return a turn with private reasoning retained or explicitly discarded."""
    return turn if retain else Turn(turn.role, turn.content, None, turn.pinned)


def rolling_truncate(state: ContextState, max_tokens: int) -> ContextState:
    """Drop oldest unpinned turns until the estimated budget is met."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    kept = list(state.turns)
    while kept and ContextState(kept, state.summary, state.compacted_turns).tokens > max_tokens:
        index = next((i for i, turn in enumerate(kept) if not turn.pinned), None)
        if index is None:
            break
        kept.pop(index)
    return ContextState(kept, state.summary, state.compacted_turns)


def compact_context(
    state: ContextState,
    max_tokens: int,
    summarize: Summarizer,
    *,
    keep_recent: int = 4,
) -> ContextState:
    """Summarize old unpinned turns and retain recent/pinned turns verbatim."""
    if max_tokens <= 0 or keep_recent < 0:
        raise ValueError("invalid compaction budget")
    if state.tokens <= max_tokens:
        return ContextState(list(state.turns), state.summary, state.compacted_turns)
    recent_start = max(0, len(state.turns) - keep_recent)
    old = [t for i, t in enumerate(state.turns) if i < recent_start and not t.pinned]
    kept = [t for i, t in enumerate(state.turns) if i >= recent_start or t.pinned]
    if not old:
        return rolling_truncate(state, max_tokens)
    new_summary = summarize(old).strip()
    if state.summary:
        new_summary = f"{state.summary}\n{new_summary}".strip()
    result = ContextState(kept, new_summary, state.compacted_turns + len(old))
    # Raw tool output is cheaper to evict than the compacted state. Drop the
    # largest unpinned retained turns first, while preserving pinned policy and
    # recent concise decisions. Only then shorten a pathological summary.
    while result.tokens > max_tokens:
        candidates = [(turn.tokens, index) for index, turn in enumerate(result.turns)
                      if not turn.pinned]
        if not candidates:
            break
        _, index = max(candidates)
        evicted = result.turns.pop(index)
        evicted_summary = extractive_summary([evicted])
        result.summary = f"{result.summary}\n{evicted_summary}".strip()
    if result.tokens > max_tokens:
        summary_chars = max(0, max_tokens * 4 - sum(t.tokens for t in result.turns) * 4)
        result.summary = result.summary[:summary_chars] if summary_chars else ""
    return result


def extractive_summary(turns: list[Turn]) -> str:
    """Deterministically preserve one normalized line per old turn."""
    lines: list[str] = []
    for turn in turns:
        # Tool payloads are usually the noisiest part of an agent transcript;
        # decisions and constraints deserve more of the compacted budget.
        content_cap = 48 if turn.role == "tool" else 160
        text = " ".join(turn.content.split())[:content_cap]
        reasoning = " ".join((turn.reasoning or "").split())[:120]
        line = f"{turn.role}: {text}"
        if reasoning:
            line += f" [reasoning: {reasoning}]"
        lines.append(line)
    return "\n".join(lines)


def fidelity(required_facts: list[str], state: ContextState) -> float:
    """Fraction of required facts still present in summary or retained turns."""
    if not required_facts:
        return 1.0
    haystack = "\n".join([state.summary, *(t.content for t in state.turns),
                           *((t.reasoning or "") for t in state.turns)]).lower()
    return sum(fact.lower() in haystack for fact in required_facts) / len(required_facts)
