"""Code verification defaults to no host execution; fixtures opt in explicitly."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier_platform.rl.sandbox import SandboxResult
from frontier_platform.rl.verifiers import CodeUnitTestVerifier, make_verifier


def test_model_code_requires_explicit_executor():
    with pytest.raises(ValueError, match="host execution is disabled"):
        CodeUnitTestVerifier(["assert f() == 1"])
    with pytest.raises(ValueError, match="isolated code executor"):
        make_verifier("code_tests", tests=["assert f() == 1"])


def test_benchmark_denies_before_any_generation():
    from frontier_platform.eval.benchmarks_2026 import LiveCodeBenchAdapter

    calls = []
    with pytest.raises(ValueError, match="before generation"):
        LiveCodeBenchAdapter().score(None, [], generate=lambda prompt: calls.append(prompt))
    assert calls == []


def test_injected_executor_gets_exact_code_tests_and_limits():
    calls = []

    def execute(code, test, limits):
        calls.append((code, test, limits.wall_seconds))
        return SandboxResult(test == "first", 0, "", "")

    verifier = CodeUnitTestVerifier(["first", "second"], executor=execute,
                                    timeout_s=2.0, all_or_nothing=False)
    assert verifier("prompt", "```python\ndef f(): return 1\n```") == 0.5
    assert calls == [("def f(): return 1", "first", 2.0),
                     ("def f(): return 1", "second", 2.0)]


def test_execution_infrastructure_failure_is_not_zero_reward():
    def unavailable(*args):
        raise RuntimeError("isolated backend unavailable")

    verifier = CodeUnitTestVerifier(["assert True"], executor=unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        verifier("prompt", "candidate")


@pytest.mark.parametrize("value", [None, 1, "passed"])
def test_ambiguous_executor_verdict_is_rejected(value):
    verifier = CodeUnitTestVerifier(["assert True"],
                                    executor=lambda *args: SimpleNamespace(ok=value))
    with pytest.raises(TypeError, match="Boolean"):
        verifier("prompt", "candidate")


@pytest.mark.parametrize("tests", [[], [""], [None]])
def test_empty_or_invalid_tests_are_not_a_reward_signal(tests):
    with pytest.raises(ValueError, match="test groups"):
        CodeUnitTestVerifier(tests, trusted_test_only=True)
