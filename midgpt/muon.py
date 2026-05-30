"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

A compact, single-GPU adaptation of Keller Jordan's Muon — the optimizer behind
the modded-nanogpt speedrun (~1.35x sample-efficiency over a tuned AdamW on the
FineWeb GPT-2 task). See https://kellerjordan.github.io/posts/muon/.

Idea: take the ordinary SGD-momentum update for a 2D weight matrix, then replace
it with the nearest semi-orthogonal matrix via a 5-step Newton-Schulz iteration
(runs stably in bf16). Orthogonalizing the update amplifies the "rare
directions" that a near-low-rank momentum buffer otherwise drowns out.

Scope rules (important):
  * Muon is for 2D *hidden* weight matrices only (attn qkv/proj, MLP weights).
  * Embeddings (tok_emb), the learned position table (pos_emb), the lm_head, all
    1-D params (LayerNorm gains, biases) → optimize with AdamW instead.

This file deliberately keeps the single-GPU path only — the official Muon also
has a distributed all-gather variant for multi-GPU that we omit for clarity.
Ported from `nanogpt-edu/muon.py`; the only midgpt-specific change is that
`split_muon_params` also excludes the 2D learned `pos_emb` table (nanogpt-edu
uses RoPE and has no such matrix).
"""
from __future__ import annotations
import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration that approximately orthogonalizes G.

    Coefficients (3.4445, -4.7750, 2.0315) are Keller Jordan's tuned quintic;
    they make 5 steps suffice. Runs in bf16 for speed; numerically stable
    because we first normalize G to bring its singular values into [0, 1].
    """
    assert G.ndim == 2, "Muon orthogonalization is defined for 2D matrices"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
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
    """Muon for 2D parameters. Use AdamW for everything else.

    Args:
        params: iterable of 2D weight tensors only.
        lr: learning rate. Muon's effective step is scale-invariant in the
            update direction, so its LR usually differs from AdamW's; the
            modded-nanogpt records use a notably higher Muon LR.
        momentum: SGD momentum coefficient (Nesterov by default).
        nesterov: use Nesterov-style lookahead on the momentum buffer.
        ns_steps: number of Newton-Schulz iterations (5 is plenty).
    """

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
                # Scale by sqrt(max(rows, cols)) so the update RMS is roughly
                # comparable across differently-shaped matrices (Keller Jordan's
                # convention — keeps a single LR sane across the network).
                scale = max(p.size(0), p.size(1)) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


def split_muon_params(model):
    """Partition model params into (muon_2d, adamw_other).

    Muon takes 2D hidden weight matrices. The token embedding, the *learned
    position table*, and the lm_head are 2D too but are excluded by name —
    they're the input/output layers, which Muon's author recommends keeping on
    AdamW. Everything 1-D (norms, biases) → AdamW.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Input/output layers stay on AdamW (Muon is for hidden 2D weights only).
        # midgpt's `pos_emb` is a 2D learned table, so it must be named here too.
        is_io = ("tok_emb" in name) or ("pos_emb" in name) or ("lm_head" in name)
        if p.ndim == 2 and not is_io:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
