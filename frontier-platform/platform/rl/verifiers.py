"""Verifiable reward functions for RLVR.

A *verifier* maps (prompt, response_text) -> scalar reward. Unlike a learned
reward model, a verifier is deterministic and (ideally) cheap, so it can score
thousands of rollouts per training step without drift or reward-model hacking of
the usual kind.

The frontier uses three big families:
  - math:   exact-answer / symbolic-equivalence checks
  - code:   unit tests run in a sandbox (gVisor/Firecracker)
  - formal: proof checkers / constraint solvers / schema validators

This module ships the cheap, dependency-free ones (string/regex/math) plus a
**sandboxed code verifier** (:class:`CodeUnitTestVerifier`, subprocess + POSIX
rlimits via :mod:`platform.rl.sandbox`). The math verifier uses sympy for
symbolic equivalence when available and falls back to numeric last-number
matching otherwise. For a public deployment, wrap the code sandbox in
gVisor/Firecracker/nsjail — see docs/09-safety-redteam.md.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Protocol

# A Verifier scores a single (prompt, response) pair.
Verifier = Callable[[str, str], float]


class _VerifierProto(Protocol):  # documentation aid only
    def __call__(self, prompt: str, response: str) -> float: ...


# ---------- cheap string/regex verifiers ----------

def reward_contains(target: str, *, reward: float = 1.0) -> Verifier:
    """Reward ``reward`` iff ``target`` appears (case-insensitively) in response."""
    t = target.lower()

    def _v(prompt: str, response: str) -> float:
        return reward if t in response.lower() else 0.0

    return _v


def reward_regex(pattern: str, *, reward: float = 1.0) -> Verifier:
    """Reward ``reward`` iff ``pattern`` matches anywhere in the response."""
    rx = re.compile(pattern)

    def _v(prompt: str, response: str) -> float:
        return reward if rx.search(response) is not None else 0.0

    return _v


def length_penalty(max_tokens: int, *, coef: float = 0.001) -> Verifier:
    """Negative shaped reward for over-long responses (curbs CoT explosion).

    Length is measured in whitespace-delimited words here for simplicity; in
    production use the tokenizer's token count.
    """

    def _v(prompt: str, response: str) -> float:
        n = len(response.split())
        return -coef * max(0, n - max_tokens)

    return _v


# ---------- math exact-answer ----------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


class MathExactVerifier:
    """Verify a math answer against an expected value.

    The canonical RLVR math reward: extract the final answer and check
    equivalence. Extraction prefers a ``\\boxed{...}`` answer (the convention the
    reasoning-SFT format teaches), then falls back to the last number in the
    text. Equivalence is checked three ways, in order:

      1. numeric close (``abs(got - expected) <= atol``),
      2. **symbolic** equality via sympy when installed (so ``1/2``, ``0.5`` and
         ``\\frac{1}{2}`` all match), and
      3. exact string match of the normalized answer.

    When ``expected`` is a string (e.g. ``"\\frac{1}{2}"``) only the symbolic /
    string paths apply. sympy is optional — without it the verifier degrades to
    numeric + string matching.
    """

    def __init__(self, expected, *, atol: float = 1e-6, reward: float = 1.0):
        self.expected_raw = expected
        try:
            self.expected = float(expected)
            self._expected_is_num = True
        except (TypeError, ValueError):
            self.expected = None
            self._expected_is_num = False
        self.atol = atol
        self.reward = reward

    def __call__(self, prompt: str, response: str) -> float:
        ans = self._extract_answer(response)
        if ans is None:
            return 0.0
        # 1. numeric close
        if self._expected_is_num:
            got = self._to_float(ans)
            if got is not None and abs(got - self.expected) <= self.atol:
                return self.reward
        # 2. symbolic equivalence (optional sympy)
        if self._symbolic_equal(ans, self.expected_raw):
            return self.reward
        # 3. exact normalized string match
        if self._normalize(ans) == self._normalize(str(self.expected_raw)):
            return self.reward
        return 0.0

    # ---- extraction ----
    _BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

    def _extract_answer(self, response: str) -> str | None:
        boxed = self._BOXED_RE.findall(response)
        if boxed:
            return boxed[-1].strip()
        nums = _NUM_RE.findall(response)
        return nums[-1] if nums else None

    @staticmethod
    def _to_float(s: str):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().rstrip("."))

    @staticmethod
    def _latex_to_sympy(s: str) -> str:
        r"""Best-effort \frac{a}{b} -> (a)/(b) and strip common LaTeX wrappers so
        plain sympy.sympify can parse simple competition answers."""
        s = s.strip().strip("$")
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = s.replace("^", "**")
        return s

    def _symbolic_equal(self, a: str, b) -> bool:
        try:
            import sympy
        except Exception:
            return False
        try:
            ea = sympy.sympify(self._latex_to_sympy(str(a)))
            eb = sympy.sympify(self._latex_to_sympy(str(b)))
            diff = sympy.simplify(ea - eb)
            return diff == 0
        except Exception:
            return False


# ---------- IFEval-style objective constraint verifiers ----------
#
# RLVR doesn't only verify *answers*; instruction-following is verifiable too.
# IFEval (Zhou et al. 2023) scores a response against a set of *objective,
# programmatically-checkable* constraints ("write at least 3 paragraphs",
# "include the keyword 'photosynthesis' twice", "respond in valid JSON", "no
# commas"). Each constraint is a deterministic predicate over the response, so
# the whole family slots straight into the Verifier protocol as a dense,
# unhackable reward — the model can't game a regex when the regex *is* the task.
#
# A constraint is ``(name, **params)``; the checker returns True/False. The
# verifier reward is the satisfied fraction (instruction-level accuracy) or
# all-or-nothing (prompt-level strict accuracy, IFEval's headline metric).

_WORD_RE = re.compile(r"\b\w+\b")
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
_TITLE_RE = re.compile(r"<<(.+?)>>")
_HIGHLIGHT_RE = re.compile(r"\*[^*\n]+\*")
# A markdown bullet line: -, *, + or "1." / "1)" style enumerators.
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S", re.MULTILINE)
# A fenced code block with an optional language tag (```json / ```python / ```).
_ANY_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\s*(.*?)```", re.DOTALL)


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_sentences(text: str) -> int:
    parts = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return len(parts)


def _count_paragraphs(text: str) -> int:
    # IFEval separates paragraphs with a line containing only '***'; fall back to
    # blank-line separation when that marker is absent.
    if "***" in text:
        parts = [p for p in re.split(r"\n?\s*\*\*\*\s*\n?", text) if p.strip()]
    else:
        parts = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(parts)


def _relation_ok(value: int, *, at_least: int | None, at_most: int | None,
                 exactly: int | None) -> bool:
    if exactly is not None:
        return value == exactly
    ok = True
    if at_least is not None:
        ok = ok and value >= at_least
    if at_most is not None:
        ok = ok and value <= at_most
    return ok


# Each checker: (response, params) -> bool. Kept module-level (not closures) so
# the registry is introspectable and individually unit-testable.

def _c_keyword_exists(resp: str, p: dict) -> bool:
    kw = p["keyword"]
    hay = resp if p.get("case_sensitive") else resp.lower()
    needle = kw if p.get("case_sensitive") else kw.lower()
    return needle in hay


def _c_keyword_forbidden(resp: str, p: dict) -> bool:
    return not _c_keyword_exists(resp, p)


def _c_keyword_frequency(resp: str, p: dict) -> bool:
    kw = p["keyword"]
    hay = resp if p.get("case_sensitive") else resp.lower()
    needle = kw if p.get("case_sensitive") else kw.lower()
    count = hay.count(needle)
    return _relation_ok(count, at_least=p.get("at_least"), at_most=p.get("at_most"),
                        exactly=p.get("exactly"))


def _c_word_count(resp: str, p: dict) -> bool:
    return _relation_ok(_count_words(resp), at_least=p.get("at_least"),
                        at_most=p.get("at_most"), exactly=p.get("exactly"))


def _c_sentence_count(resp: str, p: dict) -> bool:
    return _relation_ok(_count_sentences(resp), at_least=p.get("at_least"),
                        at_most=p.get("at_most"), exactly=p.get("exactly"))


def _c_paragraph_count(resp: str, p: dict) -> bool:
    return _relation_ok(_count_paragraphs(resp), at_least=p.get("at_least"),
                        at_most=p.get("at_most"), exactly=p.get("exactly"))


def _c_bullets(resp: str, p: dict) -> bool:
    n = len(_BULLET_LINE_RE.findall(resp))
    return _relation_ok(n, at_least=p.get("at_least"), at_most=p.get("at_most"),
                        exactly=p.get("exactly", p.get("count")))


def _c_highlights(resp: str, p: dict) -> bool:
    n = len(_HIGHLIGHT_RE.findall(resp))
    return _relation_ok(n, at_least=p.get("at_least", p.get("count")),
                        at_most=p.get("at_most"), exactly=p.get("exactly"))


def _c_title(resp: str, p: dict) -> bool:
    m = _TITLE_RE.findall(resp)
    return len(m) >= 1 and any(t.strip() for t in m)


def _c_json(resp: str, p: dict) -> bool:
    text = resp.strip()
    # Tolerate a ```json ... ``` (or any-language) fence around the payload.
    m = _ANY_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def _c_case_upper(resp: str, p: dict) -> bool:
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _c_case_lower(resp: str, p: dict) -> bool:
    letters = [c for c in resp if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _c_startswith(resp: str, p: dict) -> bool:
    return resp.strip().startswith(p["prefix"])


def _c_endswith(resp: str, p: dict) -> bool:
    return resp.strip().endswith(p["suffix"])


def _c_quotation(resp: str, p: dict) -> bool:
    t = resp.strip()
    return len(t) >= 2 and t[0] == '"' and t[-1] == '"'


def _c_no_commas(resp: str, p: dict) -> bool:
    return "," not in resp


CONSTRAINT_CHECKERS: dict[str, Callable[[str, dict], bool]] = {
    "keywords:existence": _c_keyword_exists,
    "keywords:forbidden": _c_keyword_forbidden,
    "keywords:frequency": _c_keyword_frequency,
    "length:words": _c_word_count,
    "length:sentences": _c_sentence_count,
    "length:paragraphs": _c_paragraph_count,
    "format:bullets": _c_bullets,
    "format:highlights": _c_highlights,
    "format:title": _c_title,
    "format:json": _c_json,
    "case:upper": _c_case_upper,
    "case:lower": _c_case_lower,
    "startend:startswith": _c_startswith,
    "startend:endswith": _c_endswith,
    "startend:quotation": _c_quotation,
    "punctuation:no_commas": _c_no_commas,
}


def check_constraint(name: str, response: str, **params) -> bool:
    """Evaluate a single IFEval-style constraint against ``response``.

    ``name`` must be a key of :data:`CONSTRAINT_CHECKERS`. Raises ``ValueError``
    for unknown constraints (a typo'd constraint should fail loudly, not be
    silently scored as satisfied)."""
    try:
        checker = CONSTRAINT_CHECKERS[name]
    except KeyError:
        raise ValueError(f"unknown constraint: {name!r}; "
                         f"known: {sorted(CONSTRAINT_CHECKERS)}") from None
    return checker(response, params)


class ConstraintFollowingVerifier:
    """Score instruction-following against a list of objective constraints.

    Each constraint is a ``{"name": <key>, ...params}`` dict (or an equivalent
    ``(name, params)`` tuple). Reward is:

      * **all-or-nothing** (default, IFEval *prompt-level strict*): ``reward`` iff
        *every* constraint holds, else 0.
      * **fractional** (``all_or_nothing=False``, IFEval *instruction-level*):
        ``reward * satisfied / total``.

    The prompt argument is ignored (constraints are self-contained), so this is a
    drop-in ``Verifier``. Use :meth:`breakdown` for per-constraint logging — it
    plugs into ``CompositeReward``-style monitoring."""

    def __init__(self, constraints: list, *, all_or_nothing: bool = True,
                 reward: float = 1.0):
        self.constraints = [self._normalize(c) for c in constraints]
        self.all_or_nothing = all_or_nothing
        self.reward = reward

    @staticmethod
    def _normalize(c) -> tuple[str, dict]:
        if isinstance(c, dict):
            params = dict(c)
            name = params.pop("name", None) or params.pop("type", None)
            if name is None:
                raise ValueError(f"constraint dict needs a 'name': {c!r}")
            return (name, params)
        if isinstance(c, (tuple, list)) and len(c) == 2:
            return (c[0], dict(c[1]))
        raise ValueError(f"bad constraint spec: {c!r}")

    def results(self, response: str) -> list[bool]:
        return [check_constraint(name, response, **params)
                for name, params in self.constraints]

    def breakdown(self, prompt: str, response: str) -> dict[str, float]:
        res = self.results(response)
        out = {f"{name}#{i}": float(ok)
               for i, ((name, _), ok) in enumerate(zip(self.constraints, res))}
        out["satisfied_frac"] = (sum(res) / len(res)) if res else 0.0
        return out

    def __call__(self, prompt: str, response: str) -> float:
        res = self.results(response)
        if not res:
            return 0.0
        if self.all_or_nothing:
            return self.reward if all(res) else 0.0
        return self.reward * sum(res) / len(res)


# ---------- RLPR: verifier-free probability reward ----------

class ProbabilityRewardVerifier:
    """RLVR **without a verifier** — the reward *is* the policy's own probability
    of the reference answer (RLPR; Yu et al., OpenBMB, arXiv 2506.18254).

    RLVR's reach is gated by "is there an executable checker?" — it needs a
    rule-based verifier (math equality, unit tests, a constraint regex). RLPR
    removes that gate: it scores a reasoning trace by how probable it makes the
    **known reference answer**, under the policy itself, conditioned on the
    prompt *plus the generated reasoning*:

        reward(prompt, response) = mean_t  P_policy( a_t | prompt, response, a_<t )

    i.e. the **mean decoding probability** of the reference answer tokens. A
    reasoning trace that actually leads to the answer makes those tokens likely;
    a wrong or empty one doesn't. Because it's just a forward pass over the
    reference, it works in **general domains** (chat, open-ended QA) where no
    executable checker exists — the concrete, buildable instance of the
    "process / explanation-scoring rewards" Tier-3 bet.

    Design notes:
      * **Mean of per-token probabilities** (arithmetic), not the sequence
        product — RLPR found the mean far more stable than the length-sensitive
        product (which collapses toward 0 for any multi-token answer).
      * The prompt argument carries the question; ``references[prompt]`` is the
        answer key. Unknown prompts score 0.0 (no reference to score against).
      * Lazy torch import (like :class:`CodeUnitTestVerifier`) keeps this module
        import-light; the forward runs under ``no_grad`` — it's a reward, not a
        loss term, so no gradient flows through the reward itself.
      * Drop-in ``Verifier``: ``(prompt, response) -> float`` in ``[0, reward]``,
        so it composes with ``CompositeReward`` and feeds ``grpo_step`` unchanged.
    """

    def __init__(self, model, tokenizer, references: dict[str, str], *,
                 reward: float = 1.0, include_response_context: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.references = references
        self.reward = reward
        # If True, the reference answer is scored conditioned on prompt+response
        # (the trace's reasoning is part of the context — the RLPR signal). If
        # False, it's scored on the prompt alone (an ablation: pure answer
        # likelihood, ignoring the reasoning).
        self.include_response_context = include_response_context

    def mean_answer_probability(self, prompt: str, response: str) -> float:
        """Mean decoding probability of ``references[prompt]`` under the policy.

        Returns 0.0 if there's no reference for this prompt or the reference is
        empty. The raw signal behind the reward (exposed for logging/tests)."""
        import torch

        from ..alignment._common import compute_token_logps

        ref = self.references.get(prompt)
        if not ref:
            return 0.0
        context = (prompt + response) if self.include_response_context else prompt
        prefix_ids = self.tokenizer.encode(context)
        answer_ids = self.tokenizer.encode(ref)
        if not answer_ids:
            return 0.0
        # Score the answer tokens as a continuation of the prefix. We need the
        # next-token distribution *at* each answer position, so the model sees
        # prefix + answer[:-1] and predicts answer.
        full = prefix_ids + answer_ids
        if len(full) < 2:
            return 0.0
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                ids = torch.tensor([full], dtype=torch.long,
                                   device=next(self.model.parameters()).device)
                inp, tgt = ids[:, :-1], ids[:, 1:]
                logp = compute_token_logps(self.model, inp, tgt)[0]   # [T-1]
            # The last len(answer_ids) target positions are the answer tokens.
            ans_logp = logp[-len(answer_ids):]
            mean_prob = float(ans_logp.exp().mean())
        finally:
            if was_training:
                self.model.train()
        return mean_prob

    def __call__(self, prompt: str, response: str) -> float:
        return self.reward * self.mean_answer_probability(prompt, response)


# ---------- sandboxed code (subprocess + rlimits) ----------

class CodeUnitTestVerifier:
    """Run candidate code against hidden unit tests in an isolated subprocess.

    Executes the model's code + the hidden tests in a separate OS process with
    POSIX rlimits (CPU/memory/output) and a wall-clock timeout via
    :mod:`platform.rl.sandbox`. Reward is the fraction of test groups passing
    (or all-or-nothing when ``all_or_nothing=True``).

    SECURITY: the subprocess sandbox stops runaway/OOM code and isolates crashes,
    but is not a complete jail (no FS namespacing / syscall filter). For a public
    deployment wrap `platform.rl.sandbox` in gVisor/Firecracker/nsjail + a network
    namespace — the interface is unchanged. See docs/09-safety-redteam.md.
    """

    def __init__(self, tests: list[str], *, timeout_s: float = 5.0,
                 all_or_nothing: bool = True, cpu_seconds: int = 5,
                 address_space_mb: int = 512):
        self.tests = tests
        self.timeout_s = timeout_s
        self.all_or_nothing = all_or_nothing
        self.cpu_seconds = cpu_seconds
        self.address_space_mb = address_space_mb

    def __call__(self, prompt: str, response: str) -> float:
        from .sandbox import SandboxLimits, run_in_sandbox

        code = _extract_code(response)
        limits = SandboxLimits(
            cpu_seconds=self.cpu_seconds,
            wall_seconds=self.timeout_s,
            address_space_mb=self.address_space_mb,
        )
        passed = 0
        for test in self.tests:
            res = run_in_sandbox(code, test, limits)
            passed += int(res.ok)
        if not self.tests:
            return 0.0
        if self.all_or_nothing:
            return 1.0 if passed == len(self.tests) else 0.0
        return passed / len(self.tests)


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(response: str) -> str:
    """Pull code out of a model response: prefer the first ```python``` fence,
    else use the whole response (models often emit bare code)."""
    m = _CODE_FENCE_RE.search(response)
    return (m.group(1) if m else response).strip()


# ---------- registry ----------

def make_verifier(kind: str, **kwargs) -> Verifier:
    """Factory for the built-in verifiers.

    kind ∈ {'contains', 'regex', 'math_exact', 'length_penalty', 'code_tests',
    'constraints', 'probability'}.
    """
    if kind == "contains":
        return reward_contains(**kwargs)
    if kind == "regex":
        return reward_regex(**kwargs)
    if kind == "math_exact":
        return MathExactVerifier(**kwargs)
    if kind == "length_penalty":
        return length_penalty(**kwargs)
    if kind == "code_tests":
        return CodeUnitTestVerifier(**kwargs)
    if kind == "constraints":
        return ConstraintFollowingVerifier(**kwargs)
    if kind == "probability":
        # RLPR: verifier-free reward = policy's mean decoding probability of the
        # reference answer. Needs model + tokenizer + references in kwargs.
        return ProbabilityRewardVerifier(**kwargs)
    raise ValueError(f"unknown verifier kind: {kind}")
