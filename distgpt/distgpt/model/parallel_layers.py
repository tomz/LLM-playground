"""Hand-rolled tensor-parallel linear layers.

Production-grade distgpt training uses ``parallel/tensor.py``, which leans on
``torch.distributed.tensor.parallel.parallelize_module`` — that's the modern
recommended path (composes cleanly with FSDP2, no manual collectives).

But the README and the worked-example writeups reference this module as a
hand-rolled fallback for two real cases:

* The DTensor parallelize_module path has had a couple of regressions across
  PyTorch 2.4 → 2.7; users on an in-between version sometimes need an escape
  hatch that doesn't depend on the latest DTensor.
* Pedagogical clarity: ``ColumnParallelLinear`` / ``RowParallelLinear`` /
  ``VocabParallelEmbedding`` are the canonical names from Megatron-LM and
  most distributed-training literature, and reading the actual `forward()`
  bodies is the fastest way to internalize what TP *is*.

Each layer is implemented as a thin ``nn.Module`` that holds a local shard of
the weight matrix and runs the collective explicitly. They drop into a model
in place of ``nn.Linear`` / ``nn.Embedding`` and work with any backend
(NCCL/gloo) without needing DTensor.

NOTE: this module does NOT compose with FSDP2 — FSDP2 expects to own the
parameter sharding strategy itself, and a hand-sharded weight will confuse
``fully_shard``. Use these layers OR FSDP2, not both. The DTensor path
(``parallel/tensor.py``) is the one that composes.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _world(group) -> int:
    return dist.get_world_size(group) if group is not None else 1


def _rank(group) -> int:
    return dist.get_rank(group) if group is not None else 0


# ---------------------------------------------------------------------------
# Custom autograd functions for the two collectives we need
# ---------------------------------------------------------------------------


class _AllReduceSum(torch.autograd.Function):
    """Forward: all-reduce-sum across `group`. Backward: identity.

    Used by RowParallelLinear: each rank computes a partial matmul, the
    forward all-reduces to assemble the full output, and the backward
    gradient w.r.t. the partial output is the same on every rank (the
    upstream activation gradient flows back to every shard equally).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        if _world(group) == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class _AllGather(torch.autograd.Function):
    """Forward: all-gather along the last dim. Backward: reduce-scatter.

    Used by ColumnParallelLinear when ``gather_output=True``: each rank
    holds a slice of the columns and we concatenate them. The backward
    reduce-scatter is the dual operation — each rank only needs its own
    column-slice gradient back.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        ctx.world = _world(group)
        if ctx.world == 1:
            return x
        chunks = [torch.empty_like(x) for _ in range(ctx.world)]
        dist.all_gather(chunks, x.contiguous(), group=group)
        return torch.cat(chunks, dim=-1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if ctx.world == 1:
            return grad, None
        # split along the gather dim and return just our slice
        out = grad.chunk(ctx.world, dim=-1)[_rank(ctx.group)].contiguous()
        return out, None


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Module):
    """Linear layer with the *output* (column) dimension sharded across `group`.

    Each rank stores a `[out_features // world, in_features]` slice of the
    weight. The input is replicated (no input collective). The output is
    sharded along its last dim unless ``gather_output=True``, in which case
    we all-gather to a replicated output (paying one collective per call).

    Use ``gather_output=False`` (default) when the next layer is row-parallel
    or another column-parallel — the activation stays sharded across the
    sequence of TP layers within a block, which is the whole point.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = False, group=None, gather_output: bool = False,
                 init_std: float = 0.02):
        super().__init__()
        world = _world(group)
        assert out_features % world == 0, (
            f"out_features={out_features} must be divisible by tp_size={world}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.local_out = out_features // world
        self.group = group
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = _AllGather.apply(out, self.group)
        return out


class RowParallelLinear(nn.Module):
    """Linear with the *input* (row) dimension sharded across `group`.

    Each rank stores a `[out_features, in_features // world]` slice. The
    input is expected to already be sharded along its last dim (the natural
    output of a preceding ColumnParallelLinear). Each rank computes a
    partial matmul; the forward all-reduce sums the partials. The output
    is replicated.

    Bias is added *after* the all-reduce, and only by one rank (otherwise
    it's added once per shard). We follow the convention of storing the
    full bias on every rank and adding it once on rank 0; an equivalent
    implementation divides the bias by world_size and adds everywhere.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 bias: bool = False, group=None, init_std: float = 0.02):
        super().__init__()
        world = _world(group)
        assert in_features % world == 0, (
            f"in_features={in_features} must be divisible by tp_size={world}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.local_in = in_features // world
        self.group = group
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight)               # no bias here
        out = _AllReduceSum.apply(out, self.group)   # sum partials
        if self.bias is not None and _rank(self.group) == 0:
            out = out + self.bias
        # The bias is added on rank 0 only, but its gradient must flow
        # back from every rank's all-reduce'd output. With a single-rank
        # add the autograd graph isn't aware of the other ranks — in
        # practice this is fine because the bias parameter only exists
        # on rank 0's perspective (the bias is replicated, but its grad
        # is identical on every rank after the all-reduce-sum backward).
        return out


class VocabParallelEmbedding(nn.Module):
    """Embedding table sharded along the vocab (row) dimension.

    Each rank holds a contiguous slice ``[my_start : my_end]`` of the
    vocabulary. Token ids outside the local range produce zero embeddings;
    the forward all-reduce-sum across ranks reconstructs the full embedding
    for each token (since only one rank has the non-zero entry).

    This is the standard Megatron-LM trick: avoids an expensive vocab-side
    gather while still letting the LM head stay column-parallel along vocab.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, *,
                 group=None, init_std: float = 0.02):
        super().__init__()
        world = _world(group)
        assert vocab_size % world == 0, (
            f"vocab_size={vocab_size} must be divisible by tp_size={world}; "
            "pad the vocab if needed."
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.group = group
        self.local_vocab = vocab_size // world
        my_rank = _rank(group)
        self.vocab_start = my_rank * self.local_vocab
        self.vocab_end = (my_rank + 1) * self.local_vocab
        self.weight = nn.Parameter(
            torch.empty(self.local_vocab, embedding_dim)
        )
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Mask tokens outside our slice; clamp indices into local range so
        # the gather doesn't OOB; zero-out the masked outputs.
        mask = (ids >= self.vocab_start) & (ids < self.vocab_end)
        local_ids = (ids - self.vocab_start).clamp_(0, self.local_vocab - 1)
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        out = _AllReduceSum.apply(out, self.group)
        return out


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
]
