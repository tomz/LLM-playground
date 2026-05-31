"""Tier 8.1 regression tests: bugs in extract_code / format reward / eval CLI /
GRPO config validator that ship without surfacing until a real run hits them.

The bugs these pin:
  * ``extract_code`` returned the whole prose preamble when the model didn't
    fence its output — verifier then SyntaxError'd and scored 0 (counted as
    wrong) instead of recovering the code body. Real failure mode on any base
    model that says "Sure! Here you go:" before its def.
  * Format reward's ``_DEF_RE`` didn't match ``async def`` — async code lost
    the format bonus, biasing GRPO against valid async outputs.
  * ``eval/run_humaneval.py --n-samples`` was accepted by argparse but never
    used by ``quick_eval``; pass@k was silently pass@1.
  * Eval didn't pass ``eos_token_id`` to ``model.generate`` — completions ran
    to ``max_new_tokens`` after ``<|im_end|>``.
  * The shipped ``grpo_3050.yaml`` had num_generations=6 against an effective
    batch of 8 — TRL would later raise an opaque tensor-shape error deep in
    training; this tier surfaces the error upfront in ``build_grpo_trainer``.
"""
from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.run_humaneval import extract_code, build_program
from cf_rl.reward import format_reward


# ---------------------------------------------------------------------------
# extract_code recovers code from prose-prefixed outputs
# ---------------------------------------------------------------------------


def test_extract_code_strips_prose_preamble_without_fence():
    """The headline failure: a base model emits prose then code, no fence. The
    old version returned the whole string; the verifier's ``exec()`` then died
    with SyntaxError on the prose. Now we recover the code suffix."""
    msg = "Sure! Here you go:\ndef add(a, b):\n    return a + b\n"
    recovered = extract_code(msg)
    assert recovered.lstrip().startswith("def add"), recovered
    # The recovered code must actually be valid Python (no SyntaxError):
    ns: dict = {}
    exec(recovered, ns)
    assert ns["add"](2, 3) == 5


def test_extract_code_recovers_async_def_too():
    """The recovery regex must also match ``async def`` (a coder model emitting
    an async helper after prose was previously unrecoverable)."""
    msg = "Of course:\nasync def fetch(u):\n    return u\n"
    recovered = extract_code(msg)
    assert recovered.lstrip().startswith("async def")


def test_extract_code_recovers_import_or_class_start():
    """Code can start with ``import``, ``from``, or ``class`` — not just ``def``."""
    for prefix in ("import os\n", "from re import compile\n", "class C:\n    pass\n"):
        msg = "Here is the answer:\n" + prefix
        out = extract_code(msg)
        assert out.startswith(prefix.split()[0]), (prefix, out)


def test_extract_code_fenced_block_still_wins():
    """Fence-based extraction is the priority path and must keep working."""
    msg = "Sure!\n```python\ndef f():\n    return 1\n```\nThat's it."
    assert "def f()" in extract_code(msg)
    # The trailing prose must be stripped (was already the case; pinned here).
    assert "That's it" not in extract_code(msg)


def test_extract_code_returns_text_when_nothing_codelike():
    """If there's no fence and no code-looking line, return verbatim — the
    verifier will fail honestly rather than silently dropping content."""
    msg = "I don't know how to do that, sorry."
    assert extract_code(msg) == msg


def test_build_program_uses_recovered_code_after_prose():
    """End-to-end: prose-prefixed completion now produces a runnable program."""
    prompt = "def add(a, b):\n"
    completion = "Here's the implementation:\ndef add(a, b):\n    return a + b\n"
    test = "def check(fn):\n    assert fn(2, 3) == 5\n"
    prog = build_program(prompt, completion, test, "add")
    # No SyntaxError when we exec the result:
    ns: dict = {}
    exec(prog, ns)  # would have raised on the old code


# ---------------------------------------------------------------------------
# Format reward accepts async def
# ---------------------------------------------------------------------------


def test_format_reward_credits_async_def():
    """Async functions are legitimate Python code; the format-bonus regex must
    accept them. Previously a model emitting an async function lost half the
    format reward even with a clean fenced block."""
    async_block = "```python\nasync def fetch(url):\n    return url\n```"
    sync_block = "```python\ndef fetch(url):\n    return url\n```"
    rs = format_reward(completions=[sync_block, async_block], coef=0.2)
    # Both must score the same — having `async` in front shouldn't penalize.
    assert rs[0] == rs[1] == 0.2


def test_format_reward_no_def_scores_strictly_less():
    """Sanity: not having a def at all is still worse than having one (this
    half of the reward kept working — pin it so the async fix didn't regress
    the def detection)."""
    fenced_no_def = "```python\nx = 1 + 2\nprint(x)\n```"
    fenced_def = "```python\ndef f(): return 1\n```"
    rs = format_reward(completions=[fenced_no_def, fenced_def], coef=0.2)
    assert rs[0] < rs[1]


# ---------------------------------------------------------------------------
# eval CLI flags
# ---------------------------------------------------------------------------


def test_eval_main_module_accepts_seed_and_n_samples():
    """``--n-samples`` and ``--seed`` flags must be plumbed through. We don't
    actually run eval (it would download a model + dataset); just check the
    function signature accepts both new kwargs without TypeError."""
    import inspect
    from eval.run_humaneval import quick_eval

    sig = inspect.signature(quick_eval)
    assert "n_samples" in sig.parameters
    assert "seed" in sig.parameters
    # ``seed`` must default to None so existing callers don't break.
    assert sig.parameters["seed"].default is None


def test_eval_eos_ids_helper_collects_chat_end_tokens():
    """``_eos_ids`` must collect the chat-template turn-end tokens so generation
    actually stops at ``<|im_end|>`` — pinning behavior identical to
    infer/generate.py. We synthesise a minimal tokenizer-shaped object to avoid
    a download.
    """
    from eval.run_humaneval import _eos_ids

    class FakeTok:
        eos_token_id = 7
        unk_token_id = 0

        def convert_tokens_to_ids(self, t):
            return {"<|im_end|>": 42, "<|endoftext|>": 43}.get(t, 0)

    ids = _eos_ids(FakeTok())
    assert 7 in ids
    assert 42 in ids  # <|im_end|>
    assert 43 in ids  # <|endoftext|>
    assert 0 not in ids  # unk_token_id must be filtered


# ---------------------------------------------------------------------------
# GRPO divisibility validator
# ---------------------------------------------------------------------------


def _minimal_grpo_cfg(batch_size, grad_accum, num_generations):
    """Build the smallest cfg dict that exercises build_grpo_trainer's validator
    without going anywhere near a real model load."""
    return {
        "out_dir": "/tmp/x",
        "seed": 0,
        "method": "lora",
        "model": {"dtype": "bfloat16"},
        "train": {
            "batch_size": batch_size, "grad_accum": grad_accum, "epochs": 1,
            "lr": 1e-5, "warmup_ratio": 0.0, "weight_decay": 0.0,
            "grad_clip": 1.0, "log_every": 1, "save_every": 1,
            "max_seq_len": 64, "gradient_checkpointing": True,
        },
        "grpo": {"num_generations": num_generations},
    }


def test_grpo_divisibility_validator_raises_on_mismatch():
    """The headline GRPO config footgun: effective batch (bs * accum * world)
    not divisible by num_generations. Previously you'd discover this minutes
    into a run; now it's caught before the model is even loaded."""
    from cf_rl.grpo_train import build_grpo_trainer

    # bs=2, accum=4, G=6 → 8 mod 6 = 2 (the old grpo_3050.yaml was broken!)
    bad = _minimal_grpo_cfg(batch_size=2, grad_accum=4, num_generations=6)
    with pytest.raises(SystemExit) as exc:
        build_grpo_trainer(model=None, tok=None, train_ds=None,
                           reward_funcs=[], cfg=bad)
    msg = str(exc.value)
    # The message must name the offending fields so the user can fix it.
    assert "num_generations" in msg
    assert "batch_size" in msg or "grad_accum" in msg


def test_grpo_validator_accepts_aligned_config():
    """Sanity: a divisible config must reach (and fail on) the *next* step
    (GRPOConfig instantiation needing real torch ops) — not the validator."""
    from cf_rl.grpo_train import build_grpo_trainer

    # bs=2, accum=4, G=8 → 8 mod 8 = 0; passes the validator. We expect the
    # call to fail *later* (None model can't be wrapped), but NOT at the
    # divisibility check.
    ok = _minimal_grpo_cfg(batch_size=2, grad_accum=4, num_generations=8)
    with pytest.raises(Exception) as exc:
        build_grpo_trainer(model=None, tok=None, train_ds=None,
                           reward_funcs=[], cfg=ok)
    # Must not be the divisibility SystemExit.
    if isinstance(exc.value, SystemExit):
        assert "num_generations" not in str(exc.value), \
            "divisibility validator should not have fired on an aligned config"


def test_shipped_grpo_3050_config_passes_validator():
    """Regression-pin: the shipped configs/grpo_3050.yaml must not silently
    break the divisibility rule again. We don't run the trainer — just parse
    the YAML and check the arithmetic."""
    import yaml

    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "grpo_3050.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    bs = cfg["train"]["batch_size"]
    accum = cfg["train"]["grad_accum"]
    G = cfg["grpo"]["num_generations"]
    assert (bs * accum) % G == 0, \
        f"grpo_3050.yaml: effective batch {bs}*{accum}={bs*accum} not divisible by G={G}"
