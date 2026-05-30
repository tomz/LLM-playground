"""Preference-pair datasets for DPO/ORPO — each row is (prompt, chosen, rejected).

Unlike the SFT datasets in `cf_data` (one gold ``assistant`` answer per prompt)
and the RLVR datasets in `cf_rl.prompts` (a prompt + hidden unit tests), an
*offline preference* dataset carries two candidate answers per prompt and a
label saying which is preferred. DPO/ORPO optimize the policy to raise the
log-prob margin of ``chosen`` over ``rejected`` — the cheap, stable preference
path you run *before* reaching for online RL (GRPO).

TRL's DPOTrainer accepts the "standard" text triple form directly:

    {"prompt": str, "chosen": str, "rejected": str}

We build a dependency-free built-in set by pairing each task's correct reference
solution (``chosen``) against a plausible-but-wrong variant (``rejected``) — the
kind of subtle bug a base model actually emits — so smoke runs see a real
preference signal without any download. ``hf`` loads a real preference dataset
(e.g. an UltraFeedback-style set) and normalizes common column schemas.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write a single correct Python function "
    "for the request. Return only a python code block."
)

# (instruction, chosen_code, rejected_code). The rejected variant is a realistic
# near-miss (off-by-one, wrong operator, missing edge case) rather than garbage,
# so the margin the model must learn is meaningful.
BUILTIN_PREFERENCES = [
    (
        "Write a Python function `add(a, b)` that returns the sum of a and b.",
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a - b\n",
    ),
    (
        "Write a Python function `is_even(n)` that returns True iff n is even.",
        "def is_even(n):\n    return n % 2 == 0\n",
        "def is_even(n):\n    return n % 2 == 1\n",
    ),
    (
        "Write a Python function `factorial(n)` that returns n! for n >= 0.",
        "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
        "def factorial(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
    ),
    (
        "Write a Python function `reverse_string(s)` that returns s reversed.",
        "def reverse_string(s):\n    return s[::-1]\n",
        "def reverse_string(s):\n    return s\n",
    ),
    (
        "Write a Python function `max_of(xs)` that returns the maximum of a non-empty list.",
        "def max_of(xs):\n    return max(xs)\n",
        "def max_of(xs):\n    return xs[0]\n",
    ),
    (
        "Write a Python function `count_words(s)` that returns the number of whitespace-separated words.",
        "def count_words(s):\n    return len(s.split())\n",
        "def count_words(s):\n    return len(s)\n",
    ),
    (
        "Write a Python function `gcd(a, b)` using the Euclidean algorithm.",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
        "def gcd(a, b):\n    return a % b\n",
    ),
    (
        "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumerics.",
        "def is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n",
        "def is_palindrome(s):\n    return s == s[::-1]\n",
    ),
]


def _fmt(code: str, use_chat: bool):
    """Wrap a code string as either a chat message list or a fenced text block."""
    block = f"```python\n{code}```"
    if use_chat:
        return [{"role": "assistant", "content": block}]
    return block


def _prompt(instruction: str, use_chat: bool):
    if use_chat:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
    return f"{SYSTEM_PROMPT}\n\n{instruction}\n"


def load_builtin(repeat: int = 50, use_chat: bool = False):
    """Repeat the built-in preference set so smoke-runs see a real margin curve."""
    from datasets import Dataset

    rows = []
    for _ in range(repeat):
        for instr, chosen, rejected in BUILTIN_PREFERENCES:
            rows.append({
                "prompt": _prompt(instr, use_chat),
                "chosen": _fmt(chosen, use_chat),
                "rejected": _fmt(rejected, use_chat),
            })
    return Dataset.from_list(rows)


def _normalize(ds, source_name: str):
    """Coerce common preference schemas to {prompt, chosen, rejected}."""
    cols = set(ds.column_names)
    if {"prompt", "chosen", "rejected"}.issubset(cols):
        # Already in DPO form (text or conversational) — keep extra cols off.
        keep = ["prompt", "chosen", "rejected"]
        return ds.remove_columns([c for c in ds.column_names if c not in keep])
    if {"chosen", "rejected"}.issubset(cols):
        # Some sets (e.g. Anthropic HH) embed the prompt in both; TRL can train
        # reference-free on the pair, but we require an explicit prompt column
        # for the implicit-reward math to be well-posed.
        raise ValueError(
            f"{source_name}: has chosen/rejected but no 'prompt' column; "
            "provide a prompt-carrying preference set."
        )
    raise ValueError(f"don't know how to normalize columns {cols} from {source_name}")


def load_hf(name: str, max_examples: int | None = None):
    from datasets import load_dataset

    ds = load_dataset(name, split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return _normalize(ds, name)


def load(cfg: dict):
    src = cfg.get("source", "builtin")
    use_chat = cfg.get("format", "text") == "chatml"
    if src == "builtin":
        return load_builtin(repeat=cfg.get("repeat", 50), use_chat=use_chat)
    if src == "hf":
        return load_hf(cfg["hf_name"], cfg.get("max_examples"))
    raise ValueError(f"unknown preference dataset source: {src}")
