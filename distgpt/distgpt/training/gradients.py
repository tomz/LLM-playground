"""Logical-model gradient reduction and clipping for replicated or TP parameters."""
from __future__ import annotations

import math

import torch
from torch.distributed.tensor import DTensor

from ..utils.metrics import compute_grad_norm


def finalize_gradients(model: torch.nn.Module) -> None:
    """Resolve Partial gradients (SP norms) before norm calculation/optimization.

    DDP already averages its replicated gradients. DTensor gradients must match
    the parameter placement: a replicated norm weight needs the SUM of its
    sequence slices, not independent per-rank optimizer moments.
    """
    for parameter in model.parameters():
        grad = parameter.grad
        if isinstance(grad, DTensor):
            if not isinstance(parameter, DTensor):
                raise TypeError("DTensor gradient on a non-DTensor parameter")
            parameter.grad = grad.redistribute(placements=parameter.placements)


def clip_grad_norm(model: torch.nn.Module, max_norm: float) -> float:
    """Return the logical pre-clip L2 norm and apply one common clipping factor.

    Norm reductions operate on DTensor scalars, not full gathered parameter
    tensors. Replicated parameters count once. ``max_norm=0`` disables clipping.
    Nonfinite norms are returned unmodified for a collective trainer decision.
    """
    if max_norm < 0 or not math.isfinite(max_norm):
        raise ValueError("grad_clip must be finite and nonnegative")
    finalize_gradients(model)
    norm = compute_grad_norm(model)
    if max_norm and math.isfinite(norm):
        coefficient = min(1.0, max_norm / (norm + 1e-6))
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(coefficient)
    return norm
