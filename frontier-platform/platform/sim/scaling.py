"""Chinchilla-style loss law and capability prediction.

L(N, D) = E + A / N**alpha + B / D**beta

Default constants from Hoffmann et al. 2022 (Chinchilla, "Training
Compute-Optimal Large Language Models", arXiv:2203.15556, Table 3):
    E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28

NOTE: subsequent reanalyses (Hoffmann errata; Besiroglu et al. 2024,
"Chinchilla scaling: a replication attempt", arXiv:2404.10102) report
slightly different α≈0.35, β≈0.35 with rescaled A/B. We keep the original
2022 fit here so historical comparisons line up; if you re-derive
constants, override this module's globals before calling.
"""
from __future__ import annotations
import math

E = 1.69
A = 406.4
B = 410.7
ALPHA = 0.34
BETA = 0.28


def chinchilla_loss(n_params: float, n_tokens: float) -> float:
    """Predicted train cross-entropy (nats) at compute (N, D)."""
    return E + A / max(n_params, 1.0) ** ALPHA + B / max(n_tokens, 1.0) ** BETA


def compute_flops(n_params: float, n_tokens: float) -> float:
    """Standard 6 N D FLOPs estimate."""
    return 6.0 * n_params * n_tokens


def step_loss_curve(n_params: float, total_tokens: float, step: int, total_steps: int,
                    noise: float = 0.02) -> float:
    """Loss at training step `step`, simulated by interpolating along the
    Chinchilla curve from D=tokens-per-step to D=total_tokens."""
    progress = (step + 1) / total_steps
    d_seen = max(1.0, total_tokens * progress)
    base = chinchilla_loss(n_params, d_seen)
    # mild log-decreasing noise to look realistic
    import random
    return base * (1.0 + random.gauss(0, noise) * (1 - progress * 0.8))


# --- Eval-score predictors (sigmoid-of-log-compute, calibrated to public scores) ---

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict_mmlu(n_params: float, n_tokens: float) -> float:
    """Returns MMLU 5-shot in [0.25, 0.90]. 0.25 = random chance."""
    flops = compute_flops(n_params, n_tokens)
    # rough fit to public reports: 1B/1T → ~0.32, 7B/2T → ~0.50,
    # 70B/2T → ~0.69, 400B/15T → ~0.85
    x = (math.log10(flops) - 22.5) / 1.4
    return 0.25 + 0.65 * _sigmoid(x)


def predict_humaneval(n_params: float, n_tokens: float, code_frac: float = 0.15) -> float:
    flops = compute_flops(n_params, n_tokens) * (0.5 + code_frac)
    x = (math.log10(flops) - 23.0) / 1.2
    return 0.05 + 0.85 * _sigmoid(x)


def predict_gsm8k(n_params: float, n_tokens: float, math_frac: float = 0.07) -> float:
    flops = compute_flops(n_params, n_tokens) * (0.6 + math_frac * 1.5)
    x = (math.log10(flops) - 23.4) / 1.1
    return 0.05 + 0.90 * _sigmoid(x)


def predict_arena_elo(mmlu: float, humaneval: float, gsm8k: float, sft_quality: float = 1.0,
                      rlhf_quality: float = 1.0, reasoning_quality: float = 1.0) -> float:
    """Arena ELO over a 1000-baseline. Weighted blend.

    ``reasoning_quality`` (>=1.0) is the multiplier produced by a reasoning-RL
    (RLVR/GRPO) phase; it lifts ELO the way o1/R1-style post-training does on
    top of a fixed base model.
    """
    cap = 0.5 * mmlu + 0.25 * humaneval + 0.25 * gsm8k
    return 1000.0 + 1400.0 * cap * sft_quality * rlhf_quality * reasoning_quality


# --- MoE active-parameter economics ---------------------------------------

def moe_active_params(total_params: float, n_experts: int, top_k: int,
                      shared_experts: int = 1) -> float:
    """Active (per-token) parameters for a fine-grained MoE.

    A sparse MoE only routes each token through ``top_k`` of ``n_experts`` plus
    any always-on ``shared_experts``. The non-expert weights (attention,
    embeddings, norms) are always active. We approximate the expert share of
    params as ~2/3 of the model (SwiGLU FFN dominates) and scale only that part.

    Returns the *active* parameter count that drives training/inference FLOPs.
    Dense models (n_experts<=1) return ``total_params`` unchanged.
    """
    if n_experts <= 1:
        return total_params
    expert_share = 2.0 / 3.0                      # FFN fraction of params
    dense_share = 1.0 - expert_share
    active_frac_of_experts = (top_k + shared_experts) / float(n_experts + shared_experts)
    return total_params * (dense_share + expert_share * active_frac_of_experts)


# --- Low-precision (FP8/NVFP4) training economics --------------------------

# Achieved throughput multipliers vs bf16 for the same hardware, from public
# reports (DeepSeek-V3 FP8 run; NVIDIA NVFP4). These are *effective* speedups
# after accounting for the high-precision accumulation/master-weight overhead.
PRECISION_SPEEDUP = {
    "bf16": 1.0,
    "fp8": 1.55,     # ~1.5-1.6x (DeepSeek-V3-class)
    "nvfp4": 2.2,    # ~2x+ on Blackwell (NVIDIA NVFP4), aggressive
}


def precision_speedup(precision: str) -> float:
    """Throughput multiplier vs bf16 for a training numeric format."""
    return PRECISION_SPEEDUP.get(precision, 1.0)


def reasoning_rl_quality(base_cap: float, rollouts: float, steps: float) -> float:
    """Multiplier (>=1.0) on arena ELO / reasoning evals from an RLVR phase.

    Models the empirical shape of DeepSeek-R1 / o1-style reasoning RL: the gain
    comes from *enough* verified-rollout experience and optimizer steps, gated by
    base capability (RLVR reinforces correct rollouts, so a stronger base both
    produces more reward signal and gains more). Crucially the gain is NOT tied
    to the pretraining FLOP budget — R1 got large reasoning gains from compute
    that was tiny next to pretraining. Saturates so you can't RL past the ceiling.
    """
    if rollouts <= 0 or steps <= 0:
        return 1.0
    # Saturating in experience (rollouts) and in update steps.
    exp_term = 1.0 - math.exp(-rollouts / 1.5e6)     # ~char. scale 1.5M rollouts
    step_term = 1.0 - math.exp(-steps / 3.0e3)        # ~char. scale 3k steps
    max_lift = 0.30
    lift = max_lift * base_cap * exp_term * step_term
    return 1.0 + lift
