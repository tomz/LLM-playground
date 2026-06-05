"""Reasoning-trace archetype rubric (harvest of MAI-Thinking-1 Appendix C).

The paper's most transferable qualitative finding is a *taxonomy of how reasoning
improves* as a model climbs — observed by training from scratch and watching the
traces evolve (docs/research/mai-thinking-1-deep-dive.md §6):

* **"Weak models guess, strong models work hard."** Strong traces derive *all*
  candidates and **filter by the domain condition** rather than fabricating one.
* **"Weak models brute force, strong models find invariants."** Strong traces
  name structure instead of grinding.
* **"Strong models are skeptics."** They pause ("*Wait, let's re-examine*") and
  **test their own converse on a small case**.
* **Agentic:** strong checkpoints **write and run unit tests** and do "evidence
  archaeology" (read before patching).

This module turns that taxonomy into a **deterministic, CPU-runnable rubric**: a
set of named *archetype detectors*, each scoring a trace in ``[0, 1]`` on
linguistic / structural signals of the strong-reasoning behavior. It is a
**heuristic proxy**, not an LLM judge — but (a) it gives a stable, introspectable
contract for "measure qualitative reasoning", (b) it runs in CI with no model,
and (c) on the paper's own weak-vs-strong examples the strong trace scores
higher. A real LLM judge slots in via the :class:`TraceJudge` protocol exactly
the way the red-team harness swaps :class:`platform.safety.redteam`'s judges.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol


# ---------------------------------------------------------------------------
# Signal detectors — each maps a trace to a strength in [0, 1].
# ---------------------------------------------------------------------------
#
# Detectors are intentionally simple, well-commented regex/structure heuristics.
# They reward the *presence and density* of a behavior without saturating on a
# single hit, using a smooth `1 - 0.5**k` curve over the match count `k` (one
# hit -> 0.5, two -> 0.75, three -> 0.875, ...).

def _saturating(count: int) -> float:
    """Map a non-negative match count to [0, 1): 0->0, 1->0.5, 2->0.75, ..."""
    if count <= 0:
        return 0.0
    return 1.0 - 0.5 ** count


_BACKTRACK_RE = re.compile(
    r"\b(wait|hold on|actually|on second thought|let me reconsider|"
    r"re-?examine|re-?think|that'?s wrong|scratch that|i made (?:a|an) (?:error|mistake)|"
    r"let me redo|going back)\b",
    re.IGNORECASE,
)
_VERIFY_RE = re.compile(
    r"\b(let me (?:check|verify|confirm)|double-?check|sanity check|"
    r"substitut\w+ back|plug(?:ging)? (?:it |this )?back|verify(?:ing)?|"
    r"to confirm|check(?:ing)? (?:the|our|my) (?:answer|work|result))\b",
    re.IGNORECASE,
)
_CASE_RE = re.compile(
    r"\b(case\s*\d|case\s+(?:one|two|three|i+\b)|"
    r"(?:first|second|third) case|by cases|"
    r"if .{1,40}? (?:then|else)|otherwise|"
    r"satisf\w+ the (?:condition|constraint)|"
    r"(?:filter|discard|reject|rule out|eliminat\w+|exclud\w+) "
    r"(?:the |those |these )?(?:candidate|solution|root|value|option))\b",
    re.IGNORECASE,
)
_INVARIANT_RE = re.compile(
    r"\b(invariant|symmetr\w+|conserv\w+|parity|monovariant|"
    r"without loss of generality|wlog|by induction|"
    r"structur\w+|group(?:-| )theor\w+|modul(?:o|ar)|"
    r"general (?:pattern|formula|rule)|closed form)\b",
    re.IGNORECASE,
)
_SKEPTIC_RE = re.compile(
    r"\b(is (?:that|this) (?:right|correct)\??|does (?:that|this) (?:hold|work)\??|"
    r"counter-?example|small case|test(?:ing)? (?:the |my )?converse|"
    r"but what if|let me test|try a (?:small|simple|specific) (?:example|case))\b",
    re.IGNORECASE,
)
_ENUMERATE_RE = re.compile(
    r"\b(all (?:\w+\s+){0,2}(?:candidate|root|value|solution|possibilit)|"
    r"enumerat\w+|list (?:all|the) (?:candidate|possibilit)|"
    r"there are (?:\d+|two|three|four|five|several) (?:candidate|case|root|possibilit))",
    re.IGNORECASE,
)
# Agentic archetypes.
_UNIT_TEST_RE = re.compile(
    r"(assert\s|def test_|unittest|pytest|"
    r"\bwrite (?:a |some )?(?:unit )?tests?\b|run (?:the )?tests?|"
    r"\b(?:the )?tests? (?:pass|fail|now pass))",
    re.IGNORECASE,
)
_EVIDENCE_RE = re.compile(
    r"\b(read (?:the|through) (?:repo|code|file|source)|"
    r"before (?:patch|edit|chang)\w+|"
    r"trace (?:the|through)|grep|search the (?:repo|codebase)|"
    r"source of truth|root cause|where (?:is|does) .{1,30}? (?:defined|called))\b",
    re.IGNORECASE,
)


def _detect(regex: re.Pattern) -> Callable[[str], float]:
    return lambda trace: _saturating(len(regex.findall(trace or "")))


@dataclass(frozen=True)
class ReasoningSignal:
    """A single named archetype detector."""

    name: str
    description: str
    detector: Callable[[str], float]
    agentic: bool = False

    def score(self, trace: str) -> float:
        return float(self.detector(trace))


# The default signal set, one per paper archetype (math + agentic).
DEFAULT_SIGNALS: tuple[ReasoningSignal, ...] = (
    ReasoningSignal("backtracking",
                    "pauses and self-corrects ('wait', 'let me reconsider')",
                    _detect(_BACKTRACK_RE)),
    ReasoningSignal("verification",
                    "checks its own work (substitute back, double-check)",
                    _detect(_VERIFY_RE)),
    ReasoningSignal("case_analysis",
                    "splits into cases and filters by the domain condition",
                    _detect(_CASE_RE)),
    ReasoningSignal("invariant_seeking",
                    "names structure/invariants instead of brute force",
                    _detect(_INVARIANT_RE)),
    ReasoningSignal("self_skepticism",
                    "questions its conclusion, tests a converse on a small case",
                    _detect(_SKEPTIC_RE)),
    ReasoningSignal("enumerate_then_filter",
                    "derives all candidates before selecting (not guessing one)",
                    _detect(_ENUMERATE_RE)),
    ReasoningSignal("unit_testing",
                    "writes and runs tests (agentic)",
                    _detect(_UNIT_TEST_RE), agentic=True),
    ReasoningSignal("evidence_first",
                    "reads the repo / finds the source of truth before editing (agentic)",
                    _detect(_EVIDENCE_RE), agentic=True),
)


class TraceJudge(Protocol):
    """Pluggable judge: score a trace's reasoning quality in ``[0, 1]``.

    The heuristic rubric implements this; a real deployment can supply an
    LLM-as-judge with the same signature (mirrors :mod:`platform.safety.redteam`
    judge swapping)."""

    def __call__(self, trace: str) -> float: ...


@dataclass
class RubricResult:
    """Per-signal scores plus aggregates."""

    signals: dict[str, float]
    overall: float
    math_score: float
    agentic_score: float

    def __getitem__(self, key: str) -> float:
        return self.signals[key]


@dataclass
class ReasoningRubric:
    """Score a reasoning trace against the archetype signal set.

    ``weights`` optionally re-weights individual signals in the aggregate (default
    uniform). ``overall`` is the weighted mean over all signals; ``math_score`` /
    ``agentic_score`` split by the ``agentic`` flag so a math trace isn't punished
    for lacking unit-test behavior and vice versa.
    """

    signals: tuple[ReasoningSignal, ...] = DEFAULT_SIGNALS
    weights: dict[str, float] = field(default_factory=dict)

    def _w(self, name: str) -> float:
        return float(self.weights.get(name, 1.0))

    def score(self, trace: str) -> RubricResult:
        per = {s.name: s.score(trace) for s in self.signals}

        def _mean(subset: list[ReasoningSignal]) -> float:
            wsum = sum(self._w(s.name) for s in subset)
            if wsum <= 0:
                return 0.0
            return sum(self._w(s.name) * per[s.name] for s in subset) / wsum

        math_signals = [s for s in self.signals if not s.agentic]
        agentic_signals = [s for s in self.signals if s.agentic]
        return RubricResult(
            signals=per,
            overall=_mean(list(self.signals)),
            math_score=_mean(math_signals),
            agentic_score=_mean(agentic_signals),
        )

    # Convenience so a ReasoningRubric *is* a TraceJudge.
    def __call__(self, trace: str) -> float:
        return self.score(trace).overall

    def rank(self, traces: list[str]) -> list[int]:
        """Return indices of ``traces`` ordered strongest→weakest by overall."""
        scored = sorted(range(len(traces)), key=lambda i: self.score(traces[i]).overall,
                        reverse=True)
        return scored

    def compare(self, weak: str, strong: str) -> dict[str, float]:
        """Score a (weak, strong) pair and return the per-axis deltas. Useful for
        regression-testing that a checkpoint's traces are climbing the rubric."""
        sw, ss = self.score(weak), self.score(strong)
        out = {f"delta:{k}": ss.signals[k] - sw.signals[k] for k in sw.signals}
        out["delta:overall"] = ss.overall - sw.overall
        return out


__all__ = [
    "ReasoningSignal",
    "ReasoningRubric",
    "RubricResult",
    "TraceJudge",
    "DEFAULT_SIGNALS",
]
