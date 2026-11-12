"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Ported from `nanogpt-edu/muon.py` into the frontier platform. Muon is the
optimizer behind the modded-nanogpt speedrun (~1.35x sample-efficiency over a
tuned AdamW). It orthogonalizes the SGD-momentum update for 2D weight matrices
via a 5-step Newton-Schulz iteration, amplifying the rare directions a near
low-rank momentum buffer drowns out.

Scope rules (important):
  * Muon is for 2D *hidden* weight matrices only (attn q/k/v/o, MLP/expert weights).
  * Embeddings, lm_head, MTP heads, all 1-D params (RMSNorm gains, biases),
    routing gates and the MLA latent projections that behave like IO layers
    should be optimized by AdamW instead.

Single-GPU path only (the official Muon adds a distributed all-gather variant);
when we have multi-GPU this is the natural place to add it.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration that approximately orthogonalizes G.

    Coefficients (3.4445, -4.7750, 2.0315) are Keller Jordan's tuned quintic.
    Runs in bf16 on GPU for speed; on CPU (no bf16 matmul kernel) we fall back to
    fp32 so the iteration still runs in tests.
    """
    assert G.ndim == 2, "Muon orthogonalization is defined for 2D matrices"
    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(work_dtype)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters. Route everything else to AdamW."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only accepts 2D params; got shape {tuple(p.shape)}. "
                        "Route 1-D params / embeddings / lm_head to AdamW."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = newton_schulz5(update, steps=ns_steps).to(p.dtype)
                # Scale by sqrt(max(rows, cols)) so update RMS is comparable
                # across differently-shaped matrices (Keller Jordan convention).
                scale = max(p.size(0), p.size(1)) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


# Substrings in a param's qualified name that mark it as an IO / non-hidden
# layer to keep on AdamW even though it may be 2D.
_IO_NAME_MARKERS = ("tok_emb", "lm_head", "mtp_heads", "gate", "routing_bias")


def split_muon_params(model):
    """Partition model params into (muon_2d, adamw_other).

    Muon takes 2D hidden weight matrices. Embeddings/lm_head/MTP heads are 2D too
    but excluded by name (input/output layers). MoE router gates are tiny and
    behave like IO, so they go to AdamW. Everything 1-D → AdamW.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_io = any(marker in name for marker in _IO_NAME_MARKERS)
        if p.ndim == 2 and not is_io:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
