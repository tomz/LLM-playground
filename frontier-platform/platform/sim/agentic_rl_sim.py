"""Simulate an agentic / tool-use RL post-training phase (gap #4).

Agentic RL differs from single-turn RLVR (reasoning_rl_sim.py) in cost shape:
each *episode* is a multi-turn trajectory (the agent calls tools, gets
observations, iterates), so generation cost scales with turns × tokens/turn, and
the reward is **terminal and sparse** (task completed or not). Tool execution
(code sandboxes, browsers, retrieval) is a real CPU/IO cost that often dominates
the GPU bill — just like the verifier fleet in RLVR.

The phase advances the clock, charges GPU (rollouts + updates), tool-exec CPU,
and trajectory-labeling, and returns an ``agentic_quality`` multiplier the eval
phase applies to long-horizon/agentic benchmarks (SWE-bench etc.).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .clock import Clock
from .cluster import Cluster, GPU_SPECS
from .economy import CostBook
from .events import EventBus


@dataclass
class AgenticRLSpec:
    enabled: bool = False
    tasks: int = 50_000            # long-horizon tool-use tasks
    group_size: int = 8           # trajectories per task (GRPO group)
    steps: int = 1_000            # optimizer steps
    avg_turns: int = 6            # agent<->env turns per episode
    tokens_per_turn: int = 800    # agent tokens emitted per turn
    obs_tokens_per_turn: int = 400  # tool-result tokens fed back (context cost)
    epochs: int = 1
    mfu: float = 0.30             # agentic rollouts run at low MFU (lots of stalls)
    tool_cpu_seconds_per_turn: float = 0.20   # sandbox/browser/retrieval per turn
    cpu_dollar_per_hour: float = 1.60
    coldstart_trajectories: int = 3_000        # SFT on expert agent traces
    label_dollar_per_trajectory: float = 12.0  # long traces are expensive to label


def agentic_rl_quality(base_capability: float, reasoning_quality: float,
                       episodes: float, steps: float) -> float:
    """Multiplier (>=1.0) on long-horizon/agentic evals.

    Agentic competence builds on *both* base capability and reasoning skill
    (a good agent must reason across turns), and needs enough episode experience
    and optimizer steps. Saturating, like the reasoning model.
    """
    if episodes <= 0 or steps <= 0:
        return 1.0
    exp_term = 1.0 - math.exp(-episodes / 8.0e5)
    step_term = 1.0 - math.exp(-steps / 3.0e3)
    # reasoning skill amplifies agentic gains (long-horizon needs CoT per turn)
    skill = base_capability * (0.5 + 0.5 * reasoning_quality)
    return 1.0 + 0.35 * skill * exp_term * step_term


def simulate_agentic_rl(
    spec: AgenticRLSpec,
    n_params: float,
    base_capability: float,
    reasoning_quality: float,
    cluster: Cluster,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    seed: int = 0,
) -> dict:
    """Run the agentic-RL phase. Returns dict incl. ``agentic_quality`` (>=1.0)."""
    rng = random.Random(seed)
    if not spec.enabled:
        bus.emit("agentic_rl.skipped")
        return {"agentic_quality": 1.0, "rl_compute_flops": 0.0, "compute_dollars": 0.0}

    bus.emit("agentic_rl.start", **spec.__dict__, n_params=n_params,
             base_capability=base_capability, reasoning_quality=reasoning_quality)

    episodes = spec.tasks * spec.group_size * spec.epochs
    turns = episodes * spec.avg_turns
    # Each turn the model generates tokens_per_turn AND re-attends the growing
    # context (prior turns' agent+obs tokens). Approximate generation as
    # forward over (emitted + observed) tokens per turn.
    gen_tokens = turns * (spec.tokens_per_turn + spec.obs_tokens_per_turn)
    gen_flops = 2.0 * n_params * gen_tokens
    update_tokens = spec.steps * spec.group_size * spec.avg_turns * spec.tokens_per_turn
    update_flops = 6.0 * n_params * update_tokens
    rl_flops = gen_flops + update_flops

    achieved_tflops = cluster.peak_tflops * spec.mfu
    gpu_seconds = rl_flops / (achieved_tflops * 1e12)
    clock.advance(gpu_seconds)
    gpu_dollars = cluster.total_gpus * (gpu_seconds / 3600.0) * GPU_SPECS[cluster.gpu_type]["price"]
    cost.charge("agentic_rl.compute", f"gpu_{cluster.gpu_type}", gpu_dollars)

    # Tool execution fleet (sandboxes/browsers) — charged on CPU, overlapped so
    # it adds $ but not wall-clock (like the RLVR verifier fleet).
    tool_seconds = turns * spec.tool_cpu_seconds_per_turn
    tool_dollars = (tool_seconds / 3600.0) * spec.cpu_dollar_per_hour
    cost.charge("agentic_rl.tools", "cpu_nodes", tool_dollars)

    label_dollars = spec.coldstart_trajectories * spec.label_dollar_per_trajectory
    cost.charge("agentic_rl.labels", "human_labels", label_dollars)

    quality = agentic_rl_quality(base_capability, reasoning_quality, episodes, spec.steps)
    quality *= 1 + rng.gauss(0, 0.005)

    bus.emit("agentic_rl.done",
             episodes=episodes, turns=turns, rl_flops=rl_flops,
             gpu_hours=gpu_seconds / 3600.0, gpu_dollars=gpu_dollars,
             tool_dollars=tool_dollars, label_dollars=label_dollars,
             agentic_quality=quality)

    return {
        "agentic_quality": quality,
        "rl_compute_flops": rl_flops,
        "compute_dollars": gpu_dollars + tool_dollars + label_dollars,
        "gpu_hours": gpu_seconds / 3600.0,
    }
