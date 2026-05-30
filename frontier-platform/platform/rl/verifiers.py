"""Verifiable reward functions for RLVR.

A *verifier* maps (prompt, response_text) -> scalar reward. Unlike a learned
reward model, a verifier is deterministic and (ideally) cheap, so it can score
thousands of rollouts per training step without drift or reward-model hacking of
the usual kind.

The frontier uses three big families:
  - math:   exact-answer / symbolic-equivalence checks
  - code:   unit tests run in a sandbox (gVisor/Firecracker)
  - formal: proof checkers / constraint solvers / schema validators

This module ships the cheap, dependency-free ones (string/regex/math) that make
the GRPO loop runnable in tests, and a NotImplementedError stub for the
sandboxed code verifier (which needs real isolation — see docs/09-safety-redteam.md).
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
    """Compare the *last* number in the response against an expected value.

    This is the canonical RLVR math reward: parse the final answer and check
    exact (or float-close) equality. Real systems also accept symbolic
    equivalence (sympy) and boxed-answer extraction.
    """

    def __init__(self, expected: float, *, atol: float = 1e-6, reward: float = 1.0):
        self.expected = float(expected)
        self.atol = atol
        self.reward = reward

    def __call__(self, prompt: str, response: str) -> float:
        matches = _NUM_RE.findall(response)
        if not matches:
            return 0.0
        try:
            got = float(matches[-1])
        except ValueError:
            return 0.0
        return self.reward if abs(got - self.expected) <= self.atol else 0.0


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
