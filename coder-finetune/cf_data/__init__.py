"""Dataset loaders. Each returns a HF Dataset with a single 'messages' column
in OpenAI/ChatML format: list[{'role': 'user'|'assistant'|'system', 'content': str}].
"""
from __future__ import annotations


# A tiny built-in instruction set (Python-focused, MIT-licensed style).
# Useful for smoke-runs without any download.
BUILTIN_PAIRS = [
    ("Write a Python function `fib(n)` that returns the n-th Fibonacci number iteratively.",
     "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"),
    ("Write a Python function `is_prime(n)` that returns True iff n is prime.",
     "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    if n % 2 == 0:\n        return n == 2\n    i = 3\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 2\n    return True\n```"),
    ("Write a Python function `reverse_words(s)` that reverses the order of words in a string.",
     "```python\ndef reverse_words(s):\n    return ' '.join(s.split()[::-1])\n```"),
    ("Write a Python function `count_vowels(s)` that returns the number of vowels in s.",
     "```python\ndef count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n```"),
    ("Write a Python function `flatten(xs)` that flattens a list of lists into one list.",
     "```python\ndef flatten(xs):\n    return [x for sub in xs for x in sub]\n```"),
    ("Write a Python function `gcd(a, b)` using the Euclidean algorithm.",
     "```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```"),
    ("Write a Python function `unique(xs)` that returns the list of unique items preserving order.",
     "```python\ndef unique(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n```"),
    ("Write a Python function `chunks(xs, n)` that yields successive n-sized chunks from xs.",
     "```python\ndef chunks(xs, n):\n    for i in range(0, len(xs), n):\n        yield xs[i:i+n]\n```"),
    ("Write a Python function `transpose(m)` that transposes a 2-D list (list of equal-length rows).",
     "```python\ndef transpose(m):\n    return [list(row) for row in zip(*m)]\n```"),
    ("Write a Python function `running_sum(xs)` that returns the prefix sums of xs.",
     "```python\ndef running_sum(xs):\n    out, total = [], 0\n    for x in xs:\n        total += x\n        out.append(total)\n    return out\n```"),
    ("Write a Python function `fizzbuzz(n)` that returns a list of Fizz/Buzz/FizzBuzz/number for 1..n.",
     "```python\ndef fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        s = ''\n        if i % 3 == 0: s += 'Fizz'\n        if i % 5 == 0: s += 'Buzz'\n        out.append(s or str(i))\n    return out\n```"),
    ("Write a Python function `caesar(s, k)` that applies a Caesar cipher with shift k to letters in s.",
     "```python\ndef caesar(s, k):\n    out = []\n    for c in s:\n        if c.isupper():\n            out.append(chr((ord(c) - 65 + k) % 26 + 65))\n        elif c.islower():\n            out.append(chr((ord(c) - 97 + k) % 26 + 97))\n        else:\n            out.append(c)\n    return ''.join(out)\n```"),
    ("Write a Python function `binary_search(xs, target)` returning the index or -1.",
     "```python\ndef binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```"),
    ("Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list.",
     "```python\ndef merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n```"),
    ("Write a Python function `factorial(n)` recursively.",
     "```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n```"),
    ("Write a Python function `is_palindrome(s)` ignoring case and non-alphanumeric chars.",
     "```python\ndef is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n```"),
]


SYSTEM_PROMPT = "You are a helpful coding assistant. Write clean, correct Python."


def _to_messages(prompt: str, response: str, system: str = SYSTEM_PROMPT) -> dict:
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]}


def load_builtin(repeat: int = 50):
    """Repeat the small built-in set so smoke-runs see real loss curves."""
    from datasets import Dataset
    rows = []
    for _ in range(repeat):
        for p, r in BUILTIN_PAIRS:
            rows.append(_to_messages(p, r))
    return Dataset.from_list(rows)


def load_hf(name: str, max_examples: int | None = None, lang_filter: str | None = None):
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    if lang_filter and "lang" in ds.column_names:
        ds = ds.filter(lambda ex: ex["lang"] == lang_filter)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return _normalize(ds, name)


def load_jsonl(path: str, max_examples: int | None = None):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=path, split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return _normalize(ds, path)


def _normalize(ds, source_name: str):
    """Coerce common dataset schemas to {messages: [...]}"""
    cols = set(ds.column_names)
    if "messages" in cols:
        return ds
    if {"problem", "solution"}.issubset(cols):     # Magicoder-OSS-Instruct
        return ds.map(lambda ex: _to_messages(ex["problem"], ex["solution"]),
                      remove_columns=ds.column_names)
    if {"instruction", "output"}.issubset(cols):   # alpaca-style
        def fn(ex):
            user = ex["instruction"]
            if ex.get("input"):
                user = f"{user}\n\n{ex['input']}"
            return _to_messages(user, ex["output"])
        return ds.map(fn, remove_columns=ds.column_names)
    if {"prompt", "response"}.issubset(cols):
        return ds.map(lambda ex: _to_messages(ex["prompt"], ex["response"]),
                      remove_columns=ds.column_names)
    raise ValueError(f"don't know how to normalize columns {cols} from {source_name}")


def load(cfg: dict):
    src = cfg["source"]
    if src == "builtin":
        return load_builtin()
    if src == "hf":
        return load_hf(cfg["hf_name"], cfg.get("max_examples"), cfg.get("lang_filter"))
    if src == "jsonl":
        return load_jsonl(cfg["jsonl_path"], cfg.get("max_examples"))
    raise ValueError(f"unknown dataset source: {src}")
