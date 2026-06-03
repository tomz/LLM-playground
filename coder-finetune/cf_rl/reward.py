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

from eval.run_humaneval import build_program, run_many  # noqa: E402


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
    and executed in the rlimit-bounded sandbox via ``run_many`` (currently
    sequential — see eval/run_humaneval.py:run_many for why).
    """
    assert completions is not None, "GRPO passes completions=..."
    n = len(completions)
    tests = test if test is not None else [""] * n
    entries = entry_point if entry_point is not None else [""] * n
    stubs = prompt_code if prompt_code is not None else [""] * n

    # Two-pass: first build all programs (and mark which slots are invalid),
    # then dispatch the runnable ones in one batch, then merge the results
    # back. Keeps result order identical to the per-row loop and makes the
    # batch path the single integration point should run_many ever go parallel.
    programs: list[str] = []
    runnable_idx: list[int] = []
    for i, (comp, tst, entry, stub) in enumerate(zip(completions, tests, entries, stubs)):
        if not tst or not entry:
            continue
        text = _completion_text(comp)
        programs.append(build_program(stub or "", text, tst, entry))
        runnable_idx.append(i)

    results = run_many(programs, timeout=timeout)

    rewards: list[float] = [0.0] * n
    for i, (ok, _msg) in zip(runnable_idx, results):
        rewards[i] = 1.0 if ok else 0.0
    return rewards


_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
# Must accept ``async def`` too — coder models legitimately emit async helpers
# (HTTP clients, fetchers, asyncio examples) and they used to get zero credit
# from the format bonus, dragging valid code below scoring noise.
_DEF_RE = re.compile(r"^\s*(?:async\s+def|def)\s+\w+\s*\(", re.MULTILINE)


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


def overlong_reward_shaping(
    prompts=None,
    completions=None,
    *,
    tokenizer=None,
    max_tokens: int = 1024,
    soft_window: int = 128,
    coef: float = 0.2,
    **kwargs,
) -> list[float]:
    """DAPO-style overlong shaping for responses that exceed the hard budget.

    It is zero under ``max_tokens - soft_window``, then ramps linearly to
    ``-coef`` at ``max_tokens`` and stays there. This keeps the signal smooth
    near the boundary instead of turning every over-budget sample into the same
    cliff penalty.

    COST: this re-tokenizes each completion (``tokenizer(text)``) once per call.
    In a GRPO loop that is O(batch * group) extra tokenizer passes per step. If
    the trainer already has per-sample token counts, prefer passing those in;
    the whitespace-word fallback (no ``tokenizer``) avoids the cost entirely for
    smoke runs.
    """
    assert completions is not None
    start = max(0, max_tokens - soft_window)
    width = max(1, max_tokens - start)
    out: list[float] = []
    for comp in completions:
        text = _completion_text(comp)
        n = len(tokenizer(text)["input_ids"]) if tokenizer is not None else len(text.split())
        if n <= start:
            out.append(0.0)
        else:
            out.append(-coef * min(1.0, (n - start) / width))
    return out


def dynamic_sampling_mask(rewards, group_size: int) -> list[bool]:
    """Return False for all-correct/all-wrong groups, True for useful groups.

    DAPO's dynamic sampling avoids spending policy-gradient work on groups with
    zero relative signal. This pure helper is intentionally bounded and side
    effect free; any trainer integration must cap re-sampling attempts.
    """
    n = len(rewards)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if n % group_size != 0:
        raise ValueError(f"len(rewards)={n} is not a multiple of group_size={group_size}")
    mask: list[bool] = []
    for start in range(0, n, group_size):
        group = [float(r) for r in rewards[start:start + group_size]]
        # Boundary: an exact 0.0 counts as "wrong" (r <= 0.0), so a group of
        # all-zeros is dropped, but a group mixing 0.0 with positives is kept —
        # the 0.0 is a distinct (negative) signal that still yields advantage.
        keep = not (all(r <= 0.0 for r in group) or all(r > 0.0 for r in group))
        mask.extend([keep] * group_size)
    return mask


def group_standardize_advantages(
    rewards,
    group_size: int,
    *,
    eps: float = 1e-8,
    scale_by_std: bool = True,
) -> list[float]:
    """Standardize rewards *within each group of ``group_size``* — the core of
    GRPO's value-network-free advantage estimate.

    GRPO (DeepSeekMath / DeepSeek-R1) drops PPO's learned value baseline. For
    each prompt it samples a group of G completions, then uses the *group* mean
    as the baseline and (optionally) the group std as the scale:

        A_i = (r_i - mean(group)) / (std(group) + eps)

    Two properties this pins, both load-bearing for stable GRPO:
      * **Baseline-invariance.** Adding a constant to every reward in a group
        leaves the advantages unchanged — so an absolute reward offset (e.g.
        a format bonus applied uniformly) can't bias the policy gradient.
      * **Sign structure.** With binary rewards ``[1, 0, 0, 1]`` and
        ``scale_by_std=True`` the advantages come out symmetric
        ``[+1, -1, -1, +1]`` (to float tolerance), so passing completions are
        reinforced and failing ones suppressed by equal magnitude regardless
        of the group's overall pass rate.

    This is a *reference implementation for teaching/testing* — TRL computes the
    same quantity internally on tensors. Reused here so the invariant can be
    pinned without spinning up a trainer. ``len(rewards)`` must be a multiple of
    ``group_size``.
    """
    n = len(rewards)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if n % group_size != 0:
        raise ValueError(
            f"len(rewards)={n} is not a multiple of group_size={group_size} — "
            "GRPO groups can't straddle the batch boundary."
        )
    out: list[float] = []
    for start in range(0, n, group_size):
        group = [float(r) for r in rewards[start:start + group_size]]
        mean = sum(group) / group_size
        centered = [r - mean for r in group]
        if scale_by_std:
            var = sum(c * c for c in centered) / group_size
            std = var ** 0.5
            out.extend(c / (std + eps) for c in centered)
        else:
            out.extend(centered)
    return out
