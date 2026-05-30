"""RLVR / reasoning-RL subsystem (see docs/15-reasoning-rl-rlvr.md).

This is the post-2024 post-training regime missing from `platform.alignment`:
RL against *verifiable* rewards (GRPO), where the reward is a deterministic
verifier (math/code/schema), not a learned reward model.

The implementation here is deliberately *toy-functional* — it runs end-to-end on
CPU with the byte tokenizer and the tiny test Transformer, mirroring the style of
`platform.alignment.{dpo,ppo}`. Production hooks (async rollout via vLLM/SGLang,
sandboxed code execution) are marked with NotImplementedError stubs.
"""
from .verifiers import (
    Verifier,
    reward_contains,
    reward_regex,
    length_penalty,
    MathExactVerifier,
    CodeUnitTestVerifier,
    make_verifier,
)
from .reward import (
    RewardConfig,
    CompositeReward,
    format_reward,
    soft_length_penalty,
    repetition_penalty,
    answer_spam_guard,
)
from .coldstart import ColdStartConfig, ColdStartResult, run_coldstart, format_trace
from .rollout import sample_group, GroupRollout
from .grpo import GRPOConfig, group_advantages, grpo_step, run_grpo

__all__ = [
    "Verifier",
    "reward_contains",
    "reward_regex",
    "length_penalty",
    "MathExactVerifier",
    "CodeUnitTestVerifier",
    "make_verifier",
    "RewardConfig",
    "CompositeReward",
    "format_reward",
    "soft_length_penalty",
    "repetition_penalty",
    "answer_spam_guard",
    "ColdStartConfig",
    "ColdStartResult",
    "run_coldstart",
    "format_trace",
    "sample_group",
    "GroupRollout",
    "GRPOConfig",
    "group_advantages",
    "grpo_step",
    "run_grpo",
]
