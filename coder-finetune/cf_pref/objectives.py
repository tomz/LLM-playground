"""Small reference losses for preference-optimization variants.

These are pure tensor helpers used by tests and by lightweight experiments.
The TRL trainers remain the production path when available; keeping the math
here makes SimPO/KTO behavior explicit without tying tests to a specific TRL
minor release.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def simpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    *,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Reference-free SimPO loss.

    SimPO uses average log probability as the implicit reward and applies a
    Bradley-Terry margin. Larger chosen-vs-rejected normalized margins lower
    the loss.
    """
    chosen_reward = chosen_logps / chosen_lengths.clamp_min(1)
    rejected_reward = rejected_logps / rejected_lengths.clamp_min(1)
    return -F.logsigmoid(beta * (chosen_reward - rejected_reward - gamma)).mean()


def kto_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    desirable: torch.Tensor,
    *,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> torch.Tensor:
    """KTO-style unary preference loss for desirable/undesirable examples.

    KTO consumes binary feedback, not chosen/rejected pairs. This helper pins
    the monotonic behavior: desirable completions are rewarded by increasing
    the policy-reference log-ratio, while undesirable completions are rewarded
    by decreasing it.
    """
    desirable = desirable.to(torch.bool)
    log_ratio = policy_logps - reference_logps
    desired_loss = -F.logsigmoid(beta * log_ratio)
    undesired_loss = -F.logsigmoid(-beta * log_ratio)
    weights = torch.where(
        desirable,
        torch.full_like(log_ratio, desirable_weight),
        torch.full_like(log_ratio, undesirable_weight),
    )
    losses = torch.where(desirable, desired_loss, undesired_loss) * weights
    return losses.mean()


__all__ = ["simpo_loss", "kto_loss"]
