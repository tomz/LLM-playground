"""Prompt datasets for GRPO/RLVR — each row carries a *verifier* (unit tests).

Unlike the SFT datasets in `cf_data` (which carry gold ``assistant`` answers),
an RLVR dataset carries no answer — only a prompt and the hidden tests used to
*score* whatever the model generates. Each row has:

  - ``prompt``       : chat-formatted messages (or raw text) fed to the policy
  - ``test``         : a ``check(fn)`` harness asserting correctness
  - ``entry_point``  : the function name the tests call
  - ``prompt_code``  : the function stub to prepend if the model omits the def

The columns line up with `cf_rl.reward.code_unit_test_reward`'s keyword args, so
TRL's GRPOTrainer broadcasts them straight into the reward function.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write a single correct Python function "
    "for the request. Return only a python code block."
)

# A small dependency-free verifiable set for smoke runs (no download). Each entry
# is (instruction, entry_point, stub, test).
BUILTIN_TASKS = [
    (
        "Write a Python function `add(a, b)` that returns the sum of a and b.",
        "add",
        "def add(a, b):\n",
        "def check(fn):\n    assert fn(1, 2) == 3\n    assert fn(-5, 5) == 0\n    assert fn(0, 0) == 0\n",
    ),
    (
        "Write a Python function `is_even(n)` that returns True iff n is even.",
        "is_even",
        "def is_even(n):\n",
        "def check(fn):\n    assert fn(2) is True\n    assert fn(3) is False\n    assert fn(0) is True\n",
    ),
    (
        "Write a Python function `factorial(n)` that returns n! for n >= 0.",
        "factorial",
        "def factorial(n):\n",
        "def check(fn):\n    assert fn(0) == 1\n    assert fn(5) == 120\n    assert fn(1) == 1\n",
    ),
    (
        "Write a Python function `reverse_string(s)` that returns s reversed.",
        "reverse_string",
        "def reverse_string(s):\n",
        "def check(fn):\n    assert fn('abc') == 'cba'\n    assert fn('') == ''\n    assert fn('a') == 'a'\n",
    ),
    (
        "Write a Python function `max_of(xs)` that returns the maximum of a non-empty list.",
        "max_of",
        "def max_of(xs):\n",
        "def check(fn):\n    assert fn([1, 3, 2]) == 3\n    assert fn([-1, -5]) == -1\n    assert fn([7]) == 7\n",
    ),
    (
        "Write a Python function `count_words(s)` that returns the number of whitespace-separated words.",
        "count_words",
        "def count_words(s):\n",
        "def check(fn):\n    assert fn('a b c') == 3\n    assert fn('') == 0\n    assert fn('  hi   there ') == 2\n",
    ),
    (
        "Write a Python function `gcd(a, b)` using the Euclidean algorithm.",
        "gcd",
        "def gcd(a, b):\n",
        "def check(fn):\n    assert fn(12, 8) == 4\n    assert fn(17, 5) == 1\n    assert fn(100, 10) == 10\n",
    ),
    (
        "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumerics.",
        "is_palindrome",
        "def is_palindrome(s):\n",
        "def check(fn):\n    assert fn('A man a plan a canal Panama') is True\n    assert fn('abc') is False\n",
    ),
]


def _to_prompt(instruction: str, use_chat: bool):
    if use_chat:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
    return f"{SYSTEM_PROMPT}\n\n{instruction}\n"


def load_builtin(repeat: int = 8, use_chat: bool = True):
    """Repeat the built-in verifiable tasks into a GRPO prompt dataset."""
    from datasets import Dataset

    rows = []
    for _ in range(repeat):
        for instr, entry, stub, test in BUILTIN_TASKS:
            rows.append({
                "prompt": _to_prompt(instr, use_chat),
                "test": test,
                "entry_point": entry,
                "prompt_code": stub,
            })
    return Dataset.from_list(rows)


def load_mbpp(max_examples: int | None = None, use_chat: bool = True):
    """MBPP (sanitized): real Python tasks with `test_list` assertions.

    Each MBPP row has ``text`` (the task), ``code`` (a reference solution), and
    ``test_list`` (assert statements). We synthesize a ``check(fn)`` harness from
    the asserts. The reference ``code`` is *not* shown to the policy — only used
    to recover the entry-point name.
    """
    import re

    from datasets import load_dataset

    ds = load_dataset("mbpp", "sanitized", split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    def _convert(ex):
        code = ex.get("code", "")
        m = re.search(r"def\s+(\w+)\s*\(", code)
        entry = m.group(1) if m else ""
        asserts = ex.get("test_list", [])
        # The MBPP asserts call the function by name directly; wrap them so the
        # reward harness can `check(entry)` uniformly.
        body = "\n".join("    " + a for a in asserts) if asserts else "    pass"
        test = f"def check(fn):\n{body}\n"
        return {
            "prompt": _to_prompt(ex["text"], use_chat),
            "test": test,
            "entry_point": entry,
            "prompt_code": "",
        }

    return ds.map(_convert, remove_columns=ds.column_names).filter(lambda e: bool(e["entry_point"]))


def load(cfg: dict):
    src = cfg.get("source", "builtin")
    use_chat = cfg.get("format", "chatml") == "chatml"
    if src == "builtin":
        return load_builtin(repeat=cfg.get("repeat", 8), use_chat=use_chat)
    if src == "mbpp":
        return load_mbpp(cfg.get("max_examples"), use_chat=use_chat)
    raise ValueError(f"unknown GRPO dataset source: {src}")
