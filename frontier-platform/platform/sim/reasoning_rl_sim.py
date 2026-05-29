"""Simulate a reasoning-RL (RLVR / GRPO) post-training phase.

This is the missing 2025 paradigm (DeepSeek-R1, o1): large-scale RL against
*verifiable* rewards. Unlike SFT/DPO, the dominant cost is **generation** (G
rollouts per prompt, each a long chain-of-thought), not the gradient update, and
the dominant capability gain shows up on reasoning/agentic evals and arena ELO.

The phase consumes GPU compute for rollouts + updates, charges verifier CPU
(sandboxed code/math workers), advances the clock, and returns a
``reasoning_quality`` multiplier that the eval phase applies on top of the
pretraining-only scaling-law scores.

See docs/15-reasoning-rl-rlvr.md and platform/rl/ for the real loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .clock import Clock
from .cluster import Cluster, GPU_SPECS
from .economy import CostBook
from .events import EventBus
from .scaling import reasoning_rl_quality


@dataclass
class ReasoningRLSpec:
    enabled: bool = False
    prompts: int = 100_000           # verifiable prompts (math/code/STEM)
    group_size: int = 8              # G rollouts per prompt (GRPO group)
    steps: int = 1_000               # optimizer steps
    avg_response_tokens: int = 4_000  # long-CoT rollouts are token-heavy
    prompt_tokens: int = 512
    epochs: int = 1                  # passes over the prompt set
    mfu: float = 0.35                # RL rollouts run at lower MFU than pretrain
    verifier_cpu_seconds_per_rollout: float = 0.05   # sandboxed exec/math check
    cpu_dollar_per_hour: float = 1.60
    # label cost for the (smaller) reasoning-SFT cold-start set
    coldstart_examples: int = 5_000
    label_dollar_per_coldstart: float = 6.0


def simulate_reasoning_rl(
    spec: ReasoningRLSpec,
    n_params: float,
    pretrain_flops: float,
    base_capability: float,
    cluster: Cluster,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    seed: int = 0,
) -> dict:
    """Run the RLVR phase. Returns dict incl. ``reasoning_quality`` (>=1.0).

    ``base_capability`` is mean(MMLU, HumanEval, GSM8K) of the base model — RLVR
    reinforces correct rollouts, so a stronger base both produces more reward
    signal and gains more.
    """
    rng = random.Random(seed)
    if not spec.enabled:
        bus.emit("reasoning_rl.skipped")
        return {"reasoning_quality": 1.0, "rl_compute_flops": 0.0, "compute_dollars": 0.0}

    bus.emit("reasoning_rl.start", **spec.__dict__, n_params=n_params,
             base_capability=base_capability)

    # --- compute model ---
    # Each optimizer step processes a batch of (prompt, G rollouts). We size the
    # total generated+scored tokens across the whole run.
    rollouts = spec.prompts * spec.group_size * spec.epochs
    tokens_per_rollout = spec.prompt_tokens + spec.avg_response_tokens
    total_tokens = rollouts * tokens_per_rollout

    # Generation is ~forward-only (2 N D) but done for every rollout; the policy
    # update is the usual 6 N D over the kept tokens. Approximate the whole phase
    # as forward(all rollouts) + backward(update tokens).
    gen_flops = 2.0 * n_params * total_tokens
    update_tokens = spec.steps * spec.group_size * spec.avg_response_tokens
    update_flops = 6.0 * n_params * update_tokens
    rl_flops = gen_flops + update_flops

    achieved_tflops = cluster.peak_tflops * spec.mfu
    gpu_seconds = rl_flops / (achieved_tflops * 1e12)
    clock.advance(gpu_seconds)
    gpu_dollars = cluster.total_gpus * (gpu_seconds / 3600.0) * GPU_SPECS[cluster.gpu_type]["price"]
    cost.charge("reasoning_rl.compute", f"gpu_{cluster.gpu_type}", gpu_dollars)

    # --- verifier CPU (the sandbox fleet, often the real bottleneck) ---
    verifier_seconds = rollouts * spec.verifier_cpu_seconds_per_rollout
    # assume the verifier fleet runs alongside (overlapped), so it doesn't add
    # wall-clock, but it does add $.
    verifier_dollars = (verifier_seconds / 3600.0) * spec.cpu_dollar_per_hour
    cost.charge("reasoning_rl.verifier", "cpu_nodes", verifier_dollars)

    # --- cold-start reasoning-SFT labels ---
    label_dollars = spec.coldstart_examples * spec.label_dollar_per_coldstart
    cost.charge("reasoning_rl.labels", "human_labels", label_dollars)

    # --- capability lift ---
    quality = reasoning_rl_quality(base_capability, rollouts, spec.steps)
    quality *= 1 + rng.gauss(0, 0.005)

    bus.emit("reasoning_rl.done",
             rollouts=rollouts, total_tokens=total_tokens, rl_flops=rl_flops,
             gpu_hours=gpu_seconds / 3600.0, gpu_dollars=gpu_dollars,
             verifier_dollars=verifier_dollars, label_dollars=label_dollars,
             reasoning_quality=quality,
             rl_vs_pretrain_compute=rl_flops / max(pretrain_flops, 1.0))

    return {
        "reasoning_quality": quality,
        "rl_compute_flops": rl_flops,
        "compute_dollars": gpu_dollars + verifier_dollars + label_dollars,
        "gpu_hours": gpu_seconds / 3600.0,
    }
