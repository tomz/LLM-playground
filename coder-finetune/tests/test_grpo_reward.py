"""Reward plumbing with non-executing doubles; reviewed fixtures are explicit."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.python_jail import ExecutorUnavailable
from eval.run_humaneval import build_program
from eval.trusted_subprocess import trusted_run_one
from cf_rl.reward import (
    code_unit_test_reward,
    dynamic_sampling_mask,
    format_reward,
    overlong_reward_shaping,
    soft_length_penalty,
    _completion_text,
)
from cf_rl import prompts as grpo_prompts


# ---------- reward: explicit verifier verdicts ----------

ADD_TEST = "def check(fn):\n    assert fn(1, 2) == 3\n    assert fn(0, 0) == 0\n"
ADD_GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
ADD_BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_code_reward_maps_passing_verdict_and_submits_exact_program(fake_executor):
    executor = fake_executor([(True, "passed")])
    rewards = code_unit_test_reward(
        completions=[ADD_GOOD], test=[ADD_TEST], entry_point=["add"], executor=executor,
    )
    assert rewards == [1.0]
    assert executor.programs == ["def add(a, b):\n    return a + b\n\n\n" + ADD_TEST + "\n\ncheck(add)\n"]


@pytest.mark.parametrize("outcome", ["failed", "timeout", "output_limit"])
def test_code_reward_maps_assessed_failure_to_zero(fake_executor, outcome):
    executor = fake_executor([(False, outcome)])
    rewards = code_unit_test_reward(
        completions=[ADD_BAD], test=[ADD_TEST], entry_point=["add"], executor=executor,
    )
    assert rewards == [0.0]
    assert len(executor.programs) == 1


def test_code_reward_uses_stub_when_completion_has_no_def(fake_executor):
    executor = fake_executor([(True, "passed")])
    body = "```python\n    return a + b\n```"
    rewards = code_unit_test_reward(
        completions=[body], test=[ADD_TEST], entry_point=["add"], prompt_code=["def add(a, b):\n"],
        executor=executor,
    )
    assert rewards == [1.0]
    assert executor.programs[0].startswith("def add(a, b):\n\n    return a + b\n")
    assert executor.programs[0].endswith("check(add)\n")


def test_code_reward_handles_conversational_completion(fake_executor):
    executor = fake_executor([(True, "passed")])
    chat = [{"role": "assistant", "content": ADD_GOOD}]
    rewards = code_unit_test_reward(
        completions=[chat], test=[ADD_TEST], entry_point=["add"], prompt_code=["def add(a, b):\n"],
        executor=executor,
    )
    assert rewards == [1.0]
    assert executor.programs == [build_program("", ADD_GOOD, ADD_TEST, "add")]


def test_code_reward_batch_preserves_order(fake_executor):
    executor = fake_executor([(True, "passed"), (False, "failed"), (True, "passed")])
    completions = [ADD_GOOD, ADD_BAD, ADD_GOOD]
    rewards = code_unit_test_reward(
        completions=completions, test=[ADD_TEST] * 3, entry_point=["add"] * 3,
        prompt_code=[""] * 3, executor=executor,
    )
    assert rewards == [1.0, 0.0, 1.0]
    assert executor.programs == [build_program("", code, ADD_TEST, "add") for code in completions]


@pytest.mark.parametrize("outcome", ["image_unavailable", "protocol_error", "runner_error", "cleanup_failed"])
def test_code_reward_audits_unknown_verdict_before_aborting(fake_executor, outcome):
    executor = fake_executor([(True, "passed"), (None, outcome), (False, "failed")])
    records = []
    with pytest.raises(ExecutorUnavailable, match=outcome):
        code_unit_test_reward(completions=[ADD_GOOD, ADD_GOOD, ADD_BAD], test=[ADD_TEST] * 3,
                              entry_point=["add"] * 3, executor=executor, verdict_sink=records.append)
    assert [row["sample_index"] for row in records] == [0, 1, 2]
    assert [row["passed"] for row in records] == [True, None, False]
    assert records[1]["completion"] == ADD_GOOD and records[1]["outcome"] == outcome
    assert all(row["test"] == ADD_TEST and row["evaluator"] == executor.identity() for row in records)


@pytest.mark.parametrize("column,values", [
    ("test", [ADD_TEST, "", ADD_TEST]), ("entry_point", ["add", "", "add"]),
    ("test", [ADD_TEST]), ("entry_point", ["add", "not an identifier", "add"]),
    ("prompt_code", ["", None, ""]),
])
def test_code_reward_rejects_invalid_slots_before_any_execution(fake_executor, column, values):
    executor = fake_executor([])
    columns = {"test": [ADD_TEST] * 3, "entry_point": ["add"] * 3, "prompt_code": [""] * 3}
    columns[column] = values
    with pytest.raises(ValueError):
        code_unit_test_reward(completions=[ADD_GOOD] * 3, executor=executor, **columns)
    assert executor.programs == []


def test_code_reward_requires_executor_and_completions():
    with pytest.raises(ExecutorUnavailable, match="no host fallback"):
        code_unit_test_reward(completions=[ADD_GOOD], test=[ADD_TEST], entry_point=["add"])
    with pytest.raises(ValueError, match="completions"):
        code_unit_test_reward()
    assert code_unit_test_reward(completions=[]) == []


@pytest.mark.parametrize("kwargs", [{"timeout": 1}, {"limits": False}, {"mp_mode": "fork"},
                                     {"silence_output": False}, {"max_workers": 2}, {"all_or_nothing": False}])
def test_code_reward_rejects_legacy_or_unimplemented_options(fake_executor, kwargs):
    executor = fake_executor([])
    with pytest.raises(ValueError):
        code_unit_test_reward(completions=[ADD_GOOD], test=[ADD_TEST], entry_point=["add"],
                              executor=executor, **kwargs)
    assert executor.programs == []


def test_code_reward_rejects_incomplete_verifier_batch(fake_executor, monkeypatch):
    executor = fake_executor([])
    monkeypatch.setattr(executor, "run_many", lambda programs: [])
    with pytest.raises(ExecutorUnavailable, match="incomplete reward batch"):
        code_unit_test_reward(completions=[ADD_GOOD], test=[ADD_TEST], entry_point=["add"], executor=executor)


def test_code_reward_propagates_executor_exception(fake_executor, monkeypatch):
    executor = fake_executor([])

    def unavailable(programs):
        raise ExecutorUnavailable("test backend unavailable")

    monkeypatch.setattr(executor, "run_many", unavailable)
    with pytest.raises(ExecutorUnavailable, match="test backend unavailable"):
        code_unit_test_reward(completions=[ADD_GOOD], test=[ADD_TEST], entry_point=["add"], executor=executor)


# ---------- reward: format shaping ----------

def test_format_reward_prefers_single_clean_block():
    clean = "```python\ndef f():\n    return 1\n```"
    no_block = "def f(): return 1"
    two_blocks = "```python\ndef f():\n    return 1\n```\n```python\nprint(f())\n```"
    rc = format_reward(completions=[clean, no_block, two_blocks], coef=0.2)
    assert rc[0] > rc[2]      # one block beats two
    assert rc[0] > rc[1]      # fenced+def beats raw


def test_length_penalty_zero_under_target_then_ramps():
    short = "word " * 10
    long = "word " * 5000
    r = soft_length_penalty(completions=[short, long], target_tokens=384, max_tokens=1024, coef=0.1)
    assert r[0] == 0.0
    assert r[1] < 0.0


def test_overlong_reward_shaping_ramps_near_hard_budget():
    short = "word " * 4
    mid = "word " * 9
    long = "word " * 20
    r = overlong_reward_shaping(
        completions=[short, mid, long], max_tokens=10, soft_window=4, coef=0.2,
    )
    assert r[0] == 0.0
    assert 0.0 > r[1] > -0.2
    assert r[2] == -0.2


def test_dynamic_sampling_mask_drops_zero_signal_groups():
    # All-wrong and all-correct groups have no relative advantage signal.
    mask = dynamic_sampling_mask([0, 0, 1, 1, 0, 1], group_size=2)
    assert mask == [False, False, False, False, True, True]


def test_completion_text_normalizes_shapes():
    assert _completion_text("hi") == "hi"
    assert _completion_text([{"role": "assistant", "content": "yo"}]) == "yo"


# ---------- prompt dataset ----------

def test_builtin_grpo_prompts_carry_verifier_columns():
    ds = grpo_prompts.load_builtin(repeat=1, use_chat=True)
    assert len(ds) == len(grpo_prompts.BUILTIN_TASKS)
    ex = ds[0]
    assert {"prompt", "test", "entry_point", "prompt_code"} <= set(ex)
    assert "check" in ex["test"]
    assert isinstance(ex["prompt"], list)
    assert ex["prompt"][0]["role"] == "system"


def test_builtin_grpo_prompts_text_format():
    ds = grpo_prompts.load_builtin(repeat=1, use_chat=False)
    assert isinstance(ds[0]["prompt"], str)


def test_trusted_builtin_reference_fixtures_pass_their_own_tests():
    """Only these static, reviewed repository fixtures use host execution."""
    reference = {
        "add": "def add(a, b):\n    return a + b\n",
        "is_even": "def is_even(n):\n    return n % 2 == 0\n",
        "factorial": "def factorial(n):\n    r = 1\n    for i in range(2, n+1):\n        r *= i\n    return r\n",
        "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
        "max_of": "def max_of(xs):\n    return max(xs)\n",
        "count_words": "def count_words(s):\n    return len(s.split())\n",
        "gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        "is_palindrome": "def is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n",
    }
    for _, entry, stub, test in grpo_prompts.BUILTIN_TASKS:
        program = build_program(stub, reference[entry], test, entry)
        passed, message = trusted_run_one(program, mp_mode="spawn")
        assert passed, f"trusted reference for {entry} failed: {message}"
