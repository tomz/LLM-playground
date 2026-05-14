"""Chinchilla-style loss law and capability prediction.

L(N, D) = E + A / N**alpha + B / D**beta

Default constants from Hoffmann et al. 2022 (Chinchilla):
    E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28
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
                      rlhf_quality: float = 1.0) -> float:
    """Arena ELO over a 1000-baseline. Weighted blend."""
    cap = 0.5 * mmlu + 0.25 * humaneval + 0.25 * gsm8k
    return 1000.0 + 1400.0 * cap * sft_quality * rlhf_quality
