"""Verifiable rewards for RLVR / GRPO on code (see frontier-platform docs/15).

The headline post-2024 post-training shift is **RL against verifiable rewards**:
instead of a learned reward model (which gets hacked), score each sampled
completion with a *deterministic verifier* — for code, run it against unit tests.
DeepSeek-R1 / DeepSeekMath did this with GRPO (Group-Relative Policy
Optimization), which drops PPO's value network and instead standardizes rewards
within a group of G samples per prompt.

This module provides the **reward half** for `cf_rl/grpo_train.py`. It reuses the
exact subprocess sandbox from `eval/run_humaneval.py` (timeout + separate
process), so a reward is the fraction of unit tests a completion passes. The
reward functions follow TRL's `GRPOTrainer` contract:

    reward_fn(prompts: list, completions: list, **kwargs) -> list[float]

where extra dataset columns (here ``test`` and ``entry_point``) are passed as
keyword lists. Format/length shaping mirrors `platform.rl.reward.CompositeReward`
in frontier-platform.

SECURITY: this executes model-generated code locally. The subprocess guard is a
safety floor, not a jail — wrap in Docker/gVisor/Firecracker for untrusted models
(same caveat as `eval/run_humaneval.py`).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_humaneval import build_program, run_one  # noqa: E402


def _completion_text(completion) -> str:
    """TRL passes completions as plain strings (text env) or as chat message
    lists (conversational env). Normalize to the assistant text either way."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)
    return str(completion)


def code_unit_test_reward(
    prompts=None,
    completions=None,
    *,
    test=None,
    entry_point=None,
    prompt_code=None,
    timeout: float = 5.0,
    all_or_nothing: bool = True,
    **kwargs,
) -> list[float]:
    """GRPO reward: 1.0 if the completion passes its hidden unit test(s), else 0.0.

    Args follow TRL's reward-function contract: ``completions`` is the batch of
    sampled generations; ``test`` / ``entry_point`` / ``prompt_code`` are the
    per-example dataset columns broadcast as parallel lists. Each completion is
    assembled into a runnable program (prompt stub + completion + test harness)
    and executed in the sandbox.
    """
    assert completions is not None, "GRPO passes completions=..."
    n = len(completions)
    tests = test if test is not None else [""] * n
    entries = entry_point if entry_point is not None else [""] * n
    stubs = prompt_code if prompt_code is not None else [""] * n

    rewards: list[float] = []
    for comp, tst, entry, stub in zip(completions, tests, entries, stubs):
        text = _completion_text(comp)
        if not tst or not entry:
            rewards.append(0.0)
            continue
        program = build_program(stub or "", text, tst, entry)
        ok, _msg = run_one(program, timeout=timeout)
        rewards.append(1.0 if ok else 0.0)
    return rewards


_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)


def format_reward(
    prompts=None,
    completions=None,
    *,
    coef: float = 0.2,
    **kwargs,
) -> list[float]:
    """Small shaping bonus for emitting a single well-formed python code block
    containing a function definition. Keeps generations parseable so the unit-
    test verifier can find the function (mirrors the format reward in
    frontier-platform's CompositeReward)."""
    assert completions is not None
    out: list[float] = []
    for comp in completions:
        text = _completion_text(comp)
        fenced = _FENCE_RE.findall(text)
        has_one_block = len(fenced) == 1
        body = fenced[0] if fenced else text
        has_def = _DEF_RE.search(body) is not None
        out.append(coef * (0.5 * has_one_block + 0.5 * bool(has_def)))
    return out


def soft_length_penalty(
    prompts=None,
    completions=None,
    *,
    tokenizer=None,
    target_tokens: int = 384,
    max_tokens: int = 1024,
    coef: float = 0.1,
    **kwargs,
) -> list[float]:
    """Zero up to ``target_tokens`` then ramps to ``-coef`` at ``max_tokens``.
    Curbs runaway chain-of-thought (which costs serving $ and often signals
    flailing). Uses real token counts when a ``tokenizer`` is supplied, else a
    whitespace-word approximation."""
    assert completions is not None
    out: list[float] = []
    for comp in completions:
        text = _completion_text(comp)
        n = len(tokenizer(text)["input_ids"]) if tokenizer is not None else len(text.split())
        if n <= target_tokens:
            out.append(0.0)
        else:
            frac = min(1.0, (n - target_tokens) / max(1, max_tokens - target_tokens))
            out.append(-coef * frac)
    return out
