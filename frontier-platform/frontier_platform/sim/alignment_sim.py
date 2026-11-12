"""Simulate SFT, reward model training, DPO/PPO. Quality multipliers feed into eval."""
from __future__ import annotations
import math, random
from dataclasses import dataclass

from .clock import Clock
from .cluster import Cluster, GPU_SPECS
from .economy import CostBook
from .events import EventBus


@dataclass
class AlignmentSpec:
    sft_examples: int = 250_000
    sft_epochs: int = 3
    sft_seq_len: int = 4096
    pref_pairs: int = 200_000
    rlhf: str = "dpo"        # 'dpo' | 'ppo' | 'none'
    label_dollar_per_pair: float = 4.0
    label_dollar_per_sft: float = 0.50


def simulate_alignment(
    spec: AlignmentSpec,
    n_params: float,
    cluster: Cluster,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    bus.emit("align.start", **spec.__dict__, n_params=n_params)

    # 1) SFT compute
    sft_tokens = spec.sft_examples * spec.sft_seq_len * spec.sft_epochs
    sft_flops = 6.0 * n_params * sft_tokens
    sft_tflops = cluster.peak_tflops * 0.45
    sft_seconds = sft_flops / (sft_tflops * 1e12)
    clock.advance(sft_seconds)
    sft_dollars = cluster.total_gpus * (sft_seconds / 3600) * GPU_SPECS[cluster.gpu_type]["price"]
    cost.charge("sft.compute", f"gpu_{cluster.gpu_type}", sft_dollars)
    cost.charge("sft.labels", "human_labels", spec.sft_examples * spec.label_dollar_per_sft)

    bus.emit("align.sft.done", hours=sft_seconds / 3600, dollars=sft_dollars)

    rlhf_dollars = 0.0
    if spec.rlhf != "none":
        # 2) Preference labels
        cost.charge("rlhf.labels", "human_labels", spec.pref_pairs * spec.label_dollar_per_pair)
        # 3) RM train (cheap; ~10% of SFT for DPO this is skipped)
        if spec.rlhf == "ppo":
            rm_seconds = sft_seconds * 0.4
            ppo_seconds = sft_seconds * 4.0
            rl_seconds = rm_seconds + ppo_seconds
        else:  # dpo
            rl_seconds = sft_seconds * 1.5
        clock.advance(rl_seconds)
        rlhf_dollars = cluster.total_gpus * (rl_seconds / 3600) * GPU_SPECS[cluster.gpu_type]["price"]
        cost.charge("rlhf.compute", f"gpu_{cluster.gpu_type}", rlhf_dollars)
        bus.emit("align.rlhf.done", method=spec.rlhf, hours=rl_seconds / 3600, dollars=rlhf_dollars)

    # quality multipliers (sigmoid in label volume)
    sft_q = 0.85 + 0.13 * (1 - math.exp(-spec.sft_examples / 100_000))
    rlhf_q = 1.0 if spec.rlhf == "none" else (
        0.95 + 0.10 * (1 - math.exp(-spec.pref_pairs / 100_000))
    )
    # tiny stochastic wobble
    sft_q *= 1 + rng.gauss(0, 0.01)
    rlhf_q *= 1 + rng.gauss(0, 0.01)
    bus.emit("align.done", sft_quality=sft_q, rlhf_quality=rlhf_q)
    return {"sft_quality": sft_q, "rlhf_quality": rlhf_q,
            "compute_dollars": sft_dollars + rlhf_dollars}
