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


# ---------- sandboxed code (intentional stub) ----------

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

    kind ∈ {'contains', 'regex', 'math_exact', 'length_penalty', 'code_tests'}.
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
    raise ValueError(f"unknown verifier kind: {kind}")
