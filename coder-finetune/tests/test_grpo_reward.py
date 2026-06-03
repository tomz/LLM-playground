import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cf_rl.reward import (
    code_unit_test_reward,
    dynamic_sampling_mask,
    format_reward,
    overlong_reward_shaping,
    soft_length_penalty,
    _completion_text,
)
from cf_rl import prompts as grpo_prompts


# ---------- reward: verifiable unit tests ----------

ADD_TEST = "def check(fn):\n    assert fn(1, 2) == 3\n    assert fn(0, 0) == 0\n"


def test_code_reward_passes_correct_solution():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    r = code_unit_test_reward(
        completions=[good], test=[ADD_TEST], entry_point=["add"], prompt_code=["def add(a, b):\n"],
    )
    assert r == [1.0]


def test_code_reward_fails_wrong_solution():
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    r = code_unit_test_reward(
        completions=[bad], test=[ADD_TEST], entry_point=["add"], prompt_code=["def add(a, b):\n"],
    )
    assert r == [0.0]


def test_code_reward_uses_stub_when_completion_has_no_def():
    # Completion is just the body; the stub provides the signature.
    body = "```python\n    return a + b\n```"
    r = code_unit_test_reward(
        completions=[body], test=[ADD_TEST], entry_point=["add"], prompt_code=["def add(a, b):\n"],
    )
    assert r == [1.0]


def test_code_reward_handles_conversational_completion():
    # TRL conversational env passes a list of message dicts.
    chat = [{"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}]
    r = code_unit_test_reward(
        completions=[chat], test=[ADD_TEST], entry_point=["add"], prompt_code=["def add(a, b):\n"],
    )
    assert r == [1.0]


def test_code_reward_times_out_on_infinite_loop():
    loop = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    r = code_unit_test_reward(
        completions=[loop], test=[ADD_TEST], entry_point=["add"],
        prompt_code=["def add(a, b):\n"], timeout=1.0,
    )
    assert r == [0.0]


def test_code_reward_batch_mixed():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return 0\n```"
    r = code_unit_test_reward(
        completions=[good, bad],
        test=[ADD_TEST, ADD_TEST],
        entry_point=["add", "add"],
        prompt_code=["", ""],
    )
    assert r == [1.0, 0.0]


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
    # all-wrong and all-correct groups have no relative advantage signal; mixed
    # groups are useful for GRPO/DAPO.
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
    # chat format -> messages list
    assert isinstance(ex["prompt"], list)
    assert ex["prompt"][0]["role"] == "system"


def test_builtin_grpo_prompts_text_format():
    ds = grpo_prompts.load_builtin(repeat=1, use_chat=False)
    assert isinstance(ds[0]["prompt"], str)


def test_builtin_tasks_reference_solutions_pass_their_own_tests():
    # Sanity: the canonical answers must satisfy the verifier (no broken tasks).
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
    completions, tests, entries, stubs = [], [], [], []
    for instr, entry, stub, test in grpo_prompts.BUILTIN_TASKS:
        completions.append("```python\n" + reference[entry] + "```")
        tests.append(test)
        entries.append(entry)
        stubs.append(stub)
    r = code_unit_test_reward(completions=completions, test=tests, entry_point=entries, prompt_code=stubs)
    assert r == [1.0] * len(grpo_prompts.BUILTIN_TASKS)
