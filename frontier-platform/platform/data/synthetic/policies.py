"""Generation policies: what *kind* of synthetic data to make.

A :class:`GenerationPolicy` emits prompts (the teacher will answer them) and
optionally an *acceptance verifier* (re-using :mod:`platform.rl.verifiers`) so
the factory can reject teacher outputs that don't meet the policy's spec.

The shipped policies cover the four flavours of synthetic data that move the
needle in 2025-era frontier programs:

* :class:`TemplatePolicy`  — fill-the-blank prompts (the cheap baseline).
* :class:`RephrasePolicy`  — "rewrite this in the style of" (the WizardLM trick).
* :class:`TextbookPolicy`  — "explain X to a beginner" (Phi-style).
* :class:`MathProblemPolicy`  — stamped math problems with a known answer, so the
  acceptance verifier can be a :class:`MathExactVerifier` and only correct
  teacher outputs land in the dataset.
* :class:`QAPolicy`  — paragraph -> Q&A pair (RAG-style training data).
* :class:`ReasoningTracePolicy` — a math problem paired with a verifier; the
  factory's rejection sampler keeps only chains-of-thought that arrive at the
  correct boxed answer (R1-style cold-start data).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol


Verifier = Callable[[str, str], float]


class GenerationPolicy(Protocol):
    """Protocol every policy implements.

    ``prompts(n, rng)`` yields up to ``n`` prompts using the provided
    :class:`random.Random` so generation is reproducible. ``acceptance_verifier``
    returns a callable that scores ``(prompt, response) -> [0, 1]`` reward; the
    factory keeps samples with score above ``accept_threshold``. Return ``None``
    to skip verification entirely.
    """

    name: str

    def prompts(self, n: int, *, rng: random.Random) -> Iterable[str]: ...

    def acceptance_verifier(self) -> Verifier | None: ...


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_DEFAULT_TOPICS = (
    "photosynthesis", "gradient descent", "the Roman Republic", "black holes",
    "the Krebs cycle", "compound interest", "Bayes theorem", "tectonic plates",
    "transformer attention", "the French Revolution", "genetic drift", "RAFT consensus",
    "the Doppler effect", "the Carnot cycle", "chemical equilibrium", "the Magna Carta",
)

_DEFAULT_STYLES = (
    "a concise encyclopedia entry", "an undergraduate textbook paragraph",
    "a friendly explainer for a curious teenager", "a five-bullet study guide",
    "a one-paragraph Wikipedia lead",
)


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------

@dataclass
class TemplatePolicy:
    """Fill ``templates`` with random ``slot_values``. Deterministic given seed."""

    templates: list[str]
    slot_values: dict[str, list[str]] = field(default_factory=dict)
    name: str = "template"

    def prompts(self, n: int, *, rng: random.Random) -> Iterator[str]:
        for _ in range(n):
            tmpl = rng.choice(self.templates)
            fields = {k: rng.choice(v) for k, v in self.slot_values.items()}
            yield tmpl.format(**fields)

    def acceptance_verifier(self) -> Verifier | None:
        return None


@dataclass
class RephrasePolicy:
    """WizardLM-style: rephrase / re-style passages with controlled variation."""

    seed_passages: list[str]
    styles: tuple[str, ...] = _DEFAULT_STYLES
    name: str = "rephrase"

    def prompts(self, n: int, *, rng: random.Random) -> Iterator[str]:
        for _ in range(n):
            passage = rng.choice(self.seed_passages)
            style = rng.choice(self.styles)
            yield (
                f"Rewrite the following passage as {style}. "
                f"Preserve every fact; change only the phrasing.\n\n{passage}"
            )

    def acceptance_verifier(self) -> Verifier | None:
        return None


@dataclass
class TextbookPolicy:
    """Phi-style 'textbook' prompts: explain topic at a target depth."""

    topics: tuple[str, ...] = _DEFAULT_TOPICS
    audiences: tuple[str, ...] = (
        "a curious high-schooler", "a first-year undergraduate",
        "a beginner programmer", "a non-technical reader",
    )
    name: str = "textbook"

    def prompts(self, n: int, *, rng: random.Random) -> Iterator[str]:
        for _ in range(n):
            topic = rng.choice(self.topics)
            audience = rng.choice(self.audiences)
            yield f"Explain {topic} to {audience} in three short paragraphs."

    def acceptance_verifier(self) -> Verifier | None:
        return None


@dataclass
class QAPolicy:
    """Turn ``passages`` into question-answer pairs."""

    passages: list[str]
    n_questions: int = 1
    name: str = "qa"

    def prompts(self, n: int, *, rng: random.Random) -> Iterator[str]:
        for _ in range(n):
            passage = rng.choice(self.passages)
            yield (
                f"Read the passage below and write {self.n_questions} question(s) "
                f"with their answers. Use the format 'Q: ...\\nA: ...'.\n\n{passage}"
            )

    def acceptance_verifier(self) -> Verifier | None:
        return None


@dataclass
class MathProblemPolicy:
    """Stamped arithmetic problems with a known integer answer.

    The acceptance verifier is a :class:`MathExactVerifier`, so the factory
    only keeps teacher outputs whose final boxed (or trailing) number matches
    the stamped answer. The trick: each prompt is rendered by
    :meth:`prompts`, and the answer is recovered by
    :meth:`_answer_for_prompt` (the prompt encodes the operands).

    For unit tests we hand ``MathProblemPolicy`` an explicit prompt + answer
    list via ``fixed`` so the verifier can be checked deterministically.
    """

    n_max: int = 20
    op: str = "+"
    fixed: list[tuple[str, int]] | None = None
    accept_threshold: float = 0.5
    name: str = "math_problem"

    def __post_init__(self) -> None:
        self._answers: dict[str, int] = {}
        if self.fixed:
            for p, a in self.fixed:
                self._answers[p] = a

    def prompts(self, n: int, *, rng: random.Random) -> Iterator[str]:
        if self.fixed:
            for i in range(n):
                yield self.fixed[i % len(self.fixed)][0]
            return
        for _ in range(n):
            a, b = rng.randint(1, self.n_max), rng.randint(1, self.n_max)
            if self.op == "+":
                ans = a + b
            elif self.op == "-":
                ans = a - b
            elif self.op == "*":
                ans = a * b
            else:
                raise ValueError(f"unsupported op: {self.op}")
            prompt = f"What is {a} {self.op} {b}? Reply with the number inside \\boxed{{}}."
            self._answers[prompt] = ans
            yield prompt

    def acceptance_verifier(self) -> Verifier | None:
        from platform.rl.verifiers import MathExactVerifier

        answers = self._answers

        def _verifier(prompt: str, response: str) -> float:
            expected = answers.get(prompt)
            if expected is None:
                return 0.0
            return MathExactVerifier(expected)(prompt, response)

        return _verifier


@dataclass
class ReasoningTracePolicy:
    """R1-style cold-start: ask for a chain-of-thought ending in \\boxed{answer}.

    Pairs each math problem with a :class:`MathExactVerifier` over the known
    answer. The factory's rejection sampler keeps only traces whose final
    boxed (or trailing) number matches — these become high-quality cold-start
    reasoning data.
    """

    problems: list[tuple[str, int]]
    name: str = "reasoning_trace"

    def __post_init__(self) -> None:
        self._answers: dict[str, int] = dict(self.problems)

    def prompts(self, n: int, *, rng: random.Random) -> Iterator[str]:
        items = list(self.problems)
        for i in range(n):
            problem, _ = items[i % len(items)]
            yield (
                "Solve the problem step by step. Show your reasoning, then put "
                f"the final answer inside \\boxed{{}}.\n\n{problem}"
            )

    def acceptance_verifier(self) -> Verifier | None:
        from platform.rl.verifiers import MathExactVerifier

        answers = self._answers

        def _verifier(prompt: str, response: str) -> float:
            # Find the original problem text embedded in the prompt suffix.
            for problem, expected in answers.items():
                if problem in prompt:
                    return MathExactVerifier(expected)(prompt, response)
            return 0.0

        return _verifier
