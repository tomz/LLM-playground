"""Minimal rolling-truncation versus compaction A/B for agent transcripts."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    text: str
    pinned: bool = False


@dataclass(frozen=True)
class ContextABResult:
    rolling_fidelity: float
    compacted_fidelity: float
    rolling_chars: int
    compacted_chars: int


def _fidelity(required: list[str], text: str) -> float:
    if not required:
        return 1.0
    lower = text.lower()
    return sum(fact.lower() in lower for fact in required) / len(required)


def rolling_context(messages: list[Message], char_budget: int) -> str:
    """Keep newest messages plus pinned constraints under a character budget."""
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")
    pinned = [message.text for message in messages if message.pinned]
    pinned_text = "\n".join(pinned)
    remaining = max(0, char_budget - len(pinned_text) - bool(pinned_text))
    history = "\n".join(message.text for message in messages if not message.pinned)
    recent = history[-remaining:] if remaining else ""
    return "\n".join(part for part in (pinned_text, recent) if part)


def compacted_context(messages: list[Message], char_budget: int) -> str:
    """Retain one bounded fact line per message, prioritizing pinned content."""
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")
    lines = [message.text.strip()[:80] for message in messages]
    pinned = [line for line, message in zip(lines, messages) if message.pinned]
    ordinary = [line for line, message in zip(lines, messages) if not message.pinned]
    # Preserve the beginning of each old decision before spending remaining
    # budget on newer observations.
    selected = pinned + ordinary
    return "\n".join(selected)[:char_budget]


def run_context_ab(messages: list[Message], required_facts: list[str],
                   char_budget: int) -> ContextABResult:
    rolling = rolling_context(messages, char_budget)
    compacted = compacted_context(messages, char_budget)
    return ContextABResult(
        _fidelity(required_facts, rolling),
        _fidelity(required_facts, compacted),
        len(rolling),
        len(compacted),
    )
