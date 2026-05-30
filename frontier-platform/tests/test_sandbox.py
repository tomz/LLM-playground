"""Sandboxed code-execution + code verifier tests (platform/rl/sandbox.py)."""
from __future__ import annotations

import sys

import pytest

from platform.rl.sandbox import SandboxLimits, run_in_sandbox
from platform.rl.verifiers import CodeUnitTestVerifier

_POSIX = sys.platform != "win32"


def test_correct_solution_passes():
    res = run_in_sandbox("def add(a, b):\n    return a + b",
                         "assert add(2, 3) == 5")
    assert res.ok and res.returncode == 0


def test_wrong_solution_fails():
    res = run_in_sandbox("def add(a, b):\n    return a - b",
                         "assert add(2, 3) == 5")
    assert not res.ok


def test_syntax_error_is_caught_not_raised():
    res = run_in_sandbox("def broken(:\n    pass", "assert True")
    assert not res.ok and res.returncode != 0


def test_infinite_loop_times_out():
    res = run_in_sandbox("def f():\n    while True:\n        pass\nf()",
                         "assert True",
                         SandboxLimits(cpu_seconds=1, wall_seconds=2.0))
    assert not res.ok
    # Either the wall-clock timeout or the CPU rlimit fired.
    assert res.timed_out or res.returncode != 0


@pytest.mark.skipif(not _POSIX, reason="rlimits are POSIX-only")
def test_memory_bomb_is_capped():
    # Try to allocate ~1GB under a 256MB cap -> MemoryError (nonzero exit).
    res = run_in_sandbox(
        "x = bytearray(1024 * 1024 * 1024)",
        "assert True",
        SandboxLimits(cpu_seconds=3, wall_seconds=5.0, address_space_mb=256),
    )
    assert not res.ok


def test_sandbox_cannot_crash_host_on_exception():
    # An exception in the candidate must be contained as a failed result.
    res = run_in_sandbox("raise RuntimeError('boom')", "assert True")
    assert not res.ok and "boom" in res.stderr


# ---------- CodeUnitTestVerifier ----------

def test_code_verifier_extracts_fenced_code():
    v = CodeUnitTestVerifier(tests=["assert solve(10) == 20"])
    response = "Here is my solution:\n```python\ndef solve(n):\n    return n * 2\n```\nDone."
    assert v("double it", response) == 1.0


def test_code_verifier_partial_credit():
    v = CodeUnitTestVerifier(
        tests=["assert f(1) == 1", "assert f(2) == 4", "assert f(3) == 9"],
        all_or_nothing=False,
    )
    # f(x)=x*x passes all three; f(x)=x passes only the first.
    assert v("square", "def f(x):\n    return x * x") == pytest.approx(1.0)
    score = v("square", "def f(x):\n    return x")
    assert 0.0 < score < 1.0


def test_code_verifier_all_or_nothing_default():
    v = CodeUnitTestVerifier(tests=["assert f(1) == 1", "assert f(2) == 4"])
    assert v("square", "def f(x):\n    return x") == 0.0   # only 1/2 -> 0
    assert v("square", "def f(x):\n    return x * x") == 1.0
