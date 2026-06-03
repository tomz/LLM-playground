"""RLVR / reasoning-RL subsystem (see docs/15-reasoning-rl-rlvr.md).

This is the post-2024 post-training regime missing from `platform.alignment`:
RL against *verifiable* rewards (GRPO), where the reward is a deterministic
verifier (math/code/schema), not a learned reward model.

The learner math is the production GRPO objective (per-token clipped importance
ratio against the behavior policy + k3 KL to a reference), and the code verifier
runs candidate code in a real subprocess sandbox with POSIX rlimits. Everything
runs end-to-end on CPU with the byte tokenizer and the tiny test Transformer.
The remaining swap-the-backend boundaries are external systems: an out-of-process
vLLM/SGLang generation actor (the in-process async actor–learner loop is wired)
and a gVisor/Firecracker jail around the code sandbox.
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
from .sandbox import SandboxLimits, SandboxResult, run_in_sandbox
from .async_rollout import (
    AsyncRolloutConfig,
    AsyncRolloutEngine,
    RolloutBuffer,
)
from .agentic import (
    ToolSpec,
    ToolEnv,
    Trajectory,
    Transition,
    rollout_episode,
    parse_action,
    make_calculator,
    make_keyvalue_store,
)
from .selfplay import Candidate, Generation, evaluate_candidate, run_selfplay, scripted_policy
from .rollout import sample_group, GroupRollout
from .grpo import GRPOConfig, group_advantages, grpo_step, run_grpo, run_grpo_async

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
    "SandboxLimits",
    "SandboxResult",
    "run_in_sandbox",
    "AsyncRolloutConfig",
    "AsyncRolloutEngine",
    "RolloutBuffer",
    "ToolSpec",
    "ToolEnv",
    "Trajectory",
    "Transition",
    "rollout_episode",
    "parse_action",
    "make_calculator",
    "make_keyvalue_store",
    "Candidate",
    "Generation",
    "evaluate_candidate",
    "run_selfplay",
    "scripted_policy",
    "sample_group",
    "GroupRollout",
    "GRPOConfig",
    "group_advantages",
    "grpo_step",
    "run_grpo",
    "run_grpo_async",
]
