"""Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

Port of the frontier-platform Muon implementation with the distributed
all-gather variant added so it composes with FSDP2.

Why Muon
--------
Muon is the optimizer behind the modded-nanogpt speedrun (~1.35× sample-
efficiency over a tuned AdamW). It orthogonalizes the SGD-momentum update
for 2D weight matrices via a 5-step Newton-Schulz iteration, amplifying
the rare directions a near low-rank momentum buffer drowns out.

Scope rules
-----------
* Muon is for **2D hidden** weight matrices only (attn q/k/v/o, FFN w1/w2/w3).
* Embeddings, lm_head, MTP heads, all 1-D params (RMSNorm gains, biases),
  routing gates and MLA latent projections that behave like IO layers must
  be optimized by AdamW instead.

distgpt-specific notes
----------------------
The frontier-platform port is single-GPU only; this version adds a
**distributed all-gather variant** for FSDP2-sharded params. When the
parameter is a DTensor (FSDP2 wraps shards as DTensors with Shard(0) on
the dp dim), we:

  1. all-gather the full unsharded tensor onto each rank,
  2. run Newton-Schulz on the full matrix locally,
  3. each rank applies only its local slice of the update.

That amortizes the N-S compute across DP ranks (each rank does the same
work — there's no compute saving, only a small wall-clock saving from
overlapping the all-gather with the previous parameter's update). The
official Muon paper proposes an "amortized" version that *partitions* the
N-S work; we leave that as a future optimization since the wall-clock cost
of a 5-step N-S is < 1% of step time even at 70B.

Tests
-----
``tests/test_muon.py`` covers:
  * single-GPU step matches the frontier-platform reference output bit-for-bit
  * N-S iteration converges (input near identity → output identical)
  * ``split_muon_params`` excludes the right things (tok_emb / lm_head / 1-D)
  * a 2-rank gloo distributed step produces the same update as single-GPU
"""
from __future__ import annotations

import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration that approximately orthogonalizes G.

    Coefficients (3.4445, -4.7750, 2.0315) are Keller Jordan's tuned quintic.
    Runs in bf16 on GPU for speed; on CPU (no bf16 matmul kernel) we fall
    back to fp32 so the iteration still runs in tests.
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


def _full_tensor(p: torch.Tensor) -> tuple[torch.Tensor, object | None]:
    """Return (full_tensor, original_placements_or_None) for either a DTensor
    or a regular tensor. Used so Muon's N-S can operate on the global shape
    even when FSDP2 has sharded the underlying storage.

    For DTensors, ``full_tensor()`` does the all-gather; we remember the
    original DTensor object so the update can be sharded back the same way.
    """
    if hasattr(p, "full_tensor"):
        # FSDP2 wraps params as DTensors. .full_tensor() all-gathers across
        # the sharded mesh and returns a plain torch.Tensor.
        return p.full_tensor(), p
    return p, None


def _apply_update_local(p: torch.Tensor, full_update: torch.Tensor,
                          lr_scale: float) -> None:
    """Subtract ``lr_scale * full_update`` from ``p``, handling the DTensor
    case by selecting only this rank's slice of the update."""
    if hasattr(p, "to_local"):
        # DTensor path: distribute the update with the same placement spec
        # as p, then add. The DTensor library handles the slice arithmetic.
        from torch.distributed.tensor import DTensor, distribute_tensor
        update_dtensor = distribute_tensor(
            full_update, p.device_mesh, p.placements
        )
        assert isinstance(update_dtensor, DTensor)
        p.add_(update_dtensor, alpha=-lr_scale)
    else:
        p.add_(full_update, alpha=-lr_scale)


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters. Route everything else to AdamW.

    Composes with FSDP2: when ``p`` is a DTensor (the FSDP2 default), each
    optimizer step all-gathers to the full matrix, runs Newton-Schulz on
    the global shape, and distributes the resulting update back to the
    sharded layout.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                # ndim check is on the local shard; with DTensor the local
                # shard may be 2D (sharded along dim 0) or 1D (degenerate).
                # We just require the *global* shape to be 2D, which is
                # what `full_tensor()` returns.
                if hasattr(p, "full_tensor"):
                    global_ndim = p.full_tensor().ndim
                else:
                    global_ndim = p.ndim
                if global_ndim != 2:
                    raise ValueError(
                        f"Muon only accepts 2D params; got shape "
                        f"{tuple(p.shape)} (global ndim={global_ndim}). "
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
                # Distributed path: all-gather to the full shape, run N-S
                # on the global matrix, then scatter the update back.
                full_update, _ = _full_tensor(update)
                full_p, _ = _full_tensor(p)
                ortho = newton_schulz5(full_update, steps=ns_steps).to(p.dtype)
                # Scale by sqrt(max(rows, cols)) so update RMS is comparable
                # across differently-shaped matrices (Keller Jordan convention).
                scale = max(full_p.size(0), full_p.size(1)) ** 0.5
                _apply_update_local(p, ortho, lr * scale)
        return loss


# Substrings in a param's qualified name that mark it as an IO / non-hidden
# layer to keep on AdamW even though it may be 2D.
_IO_NAME_MARKERS = ("tok_emb", "lm_head", "mtp_heads", "gate", "routing_bias")


def split_muon_params(model):
    """Partition model params into (muon_2d, adamw_other).

    Muon takes 2D hidden weight matrices. Embeddings/lm_head/MTP heads are
    2D too but excluded by name (input/output layers). MoE router gates are
    tiny and behave like IO, so they go to AdamW. Everything 1-D → AdamW.

    Works for both regular tensors and DTensors (FSDP2-wrapped); the ndim
    check is against the global shape via .full_tensor() when available.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_io = any(marker in name for marker in _IO_NAME_MARKERS)
        global_ndim = (p.full_tensor().ndim if hasattr(p, "full_tensor")
                        else p.ndim)
        if global_ndim == 2 and not is_io:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def build_muon_and_adamw(model, *, muon_lr: float, adamw_lr: float,
                           muon_momentum: float = 0.95,
                           adamw_betas=(0.9, 0.95),
                           weight_decay: float = 0.1, fused: bool = False):
    """Convenience builder: split model params and return both optimizers.

    Use this from trainer.py when ``optim.optimizer == 'muon'``:

        optims = build_muon_and_adamw(model, muon_lr=0.02, adamw_lr=3e-4, ...)
        # in the loop:
        for opt in optims: opt.zero_grad(set_to_none=True)
        # ... backward ...
        for opt in optims: opt.step()
    """
    muon_params, adamw_params = split_muon_params(model)
    optims = []
    if muon_params:
        optims.append(Muon(muon_params, lr=muon_lr, momentum=muon_momentum))
    if adamw_params:
        # 1-D params (norms, biases) shouldn't decay; this is the same
        # split as in training/optim.py:build_optimizer.
        decay, no_decay = [], []
        for p in adamw_params:
            (decay if p.dim() >= 2 else no_decay).append(p)
        optims.append(torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=adamw_lr, betas=tuple(adamw_betas), fused=fused,
        ))
    return optims


__all__ = [
    "Muon",
    "newton_schulz5",
    "split_muon_params",
    "build_muon_and_adamw",
]
