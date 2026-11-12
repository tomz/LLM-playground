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
    """Repeat fixtures with stable task IDs so copies cannot leak across splits."""
    from datasets import Dataset
    rows = []
    for _ in range(repeat):
        for index, (p, r) in enumerate(BUILTIN_PAIRS):
            rows.append({**_to_messages(p, r), "source": "builtin",
                         "task_id": f"builtin/{index}", "license": "MIT"})
    return _normalize(Dataset.from_list(rows), "builtin")


def load_hf(name: str, max_examples: int | None = None, lang_filter: str | None = None,
            shuffle: bool = False, seed: int = 1337, revision: str | None = None):
    """Load a pinned source, retaining metadata through normalization."""
    from datasets import load_dataset
    from cf_provenance import require_revision
    revision = require_revision(revision, label="SFT dataset")
    ds = load_dataset(name, split="train", revision=revision)
    if lang_filter:
        if "lang" not in ds.column_names:
            raise ValueError("lang_filter requested but the dataset has no lang column")
        ds = ds.filter(lambda ex: ex["lang"] == lang_filter)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return _normalize(ds, name, revision)


def load_jsonl(path: str, max_examples: int | None = None):
    """Identify a local source by its bytes, not its mutable filename alone."""
    from datasets import load_dataset
    from cf_provenance import file_sha256
    ds = load_dataset("json", data_files=path, split="train")
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return _normalize(ds, path, file_sha256(path))


def _normalize(ds, source_name: str, revision: str | None = None):
    """Add messages and provenance without discarding license/source/group fields."""
    cols = set(ds.column_names)
    if not ("messages" in cols or {"problem", "solution"} <= cols
            or {"instruction", "output"} <= cols or {"prompt", "response"} <= cols):
        raise ValueError(f"don't know how to normalize columns {cols} from {source_name}")

    def normalize(ex: dict) -> dict:
        if "messages" in ex:
            result = {"messages": ex["messages"]}
        elif "problem" in ex:
            result = _to_messages(ex["problem"], ex["solution"])
        elif "instruction" in ex:
            user = ex["instruction"]
            if ex.get("input"):
                user += f"\n\n{ex['input']}"
            result = _to_messages(user, ex["output"])
        else:
            result = _to_messages(ex["prompt"], ex["response"])
        result.update(_cf_source=source_name, _cf_revision=revision)
        return result
    return ds.map(normalize)


def split_dataset(ds, *, validation_fraction: float, seed: int,
                  group_by: str | None = None) -> tuple:
    """Stable source-stratified group holdout; identical prompts stay together.

    Explicit group_by (for example repository) is global across sources. Without
    it use group_id/repository/repo/task_id, then the non-assistant conversation.
    Exact prompt duplicates join groups transitively. This is not a semantic or
    benchmark contamination detector. Singleton source strata stay in training.
    """
    import math
    from collections import defaultdict
    from cf_provenance import digest

    if isinstance(validation_fraction, bool) or not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    rows = list(ds)
    parents: dict[str, str] = {}
    prompts: dict[str, str] = {}
    entries = []

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("each SFT row needs nonempty messages")
        prompt_hash = digest([m for m in messages if m.get("role") != "assistant"])
        field = group_by or next((k for k in ("group_id", "repository", "repo", "task_id") if row.get(k)), None)
        if field and not row.get(field):
            raise ValueError(f"missing split group field: {field}")
        group = digest([field, row[field]]) if field else prompt_hash
        parents.setdefault(group, group)
        if prompt_hash in prompts:
            a, b = root(group), root(prompts[prompt_hash])
            parents[max(a, b)] = min(a, b)
        prompts[prompt_hash] = group
        entries.append({"row_sha256": digest(row), "prompt_sha256": prompt_hash,
                        "group": group, "source": row.get("source") or row.get("_cf_source") or "unspecified",
                        "metadata": {k: v for k, v in row.items() if k != "messages"}})
    groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        entry["group"] = root(entry["group"])
        groups[entry["group"]].append(index)
    strata: dict[str, list[str]] = defaultdict(list)
    for group, indices in groups.items():
        strata[min(str(entries[i]["source"]) for i in indices)].append(group)
    held_out = set()
    for source, keys in sorted(strata.items()):
        keys.sort(key=lambda key: digest([seed, source, key]))
        if len(keys) > 1:
            count = max(1, min(len(keys) - 1, round(len(keys) * validation_fraction)))
            held_out.update(keys[:count])
    if not held_out:
        raise ValueError("validation needs at least two independent groups within a source")
    membership: dict[str, list[int]] = {"train": [], "validation": []}
    for index, entry in enumerate(entries):
        entry["split"] = "validation" if entry["group"] in held_out else "train"
        membership[entry["split"]].append(index)
    manifest = {"schema": "coder-finetune.sft-split/v1", "seed": seed,
                "validation_fraction": validation_fraction, "group_by": group_by,
                "policy": "source-stratified group holdout; exact prompt groups joined; singleton strata train-only",
                "rows": entries, "counts": {k: len(v) for k, v in membership.items()},
                "membership_sha256": digest(sorted((e["row_sha256"], e["group"], e["split"]) for e in entries))}
    return ds.select(membership["train"]), ds.select(membership["validation"]), manifest


def load(cfg: dict):
    """Load and cap examples before the independent validation split."""
    src = cfg["source"]
    if src == "builtin":
        ds = load_builtin(cfg.get("repeat", 50))
        if cfg.get("max_examples"):
            ds = ds.select(range(min(cfg["max_examples"], len(ds))))
        return ds
    if src == "hf":
        return load_hf(cfg["hf_name"], cfg.get("max_examples"), cfg.get("lang_filter"),
                       shuffle=cfg.get("shuffle", False), seed=cfg.get("shuffle_seed", 1337),
                       revision=cfg.get("revision"))
    if src == "jsonl":
        return load_jsonl(cfg["jsonl_path"], cfg.get("max_examples"))
    raise ValueError(f"unknown dataset source: {src}")
