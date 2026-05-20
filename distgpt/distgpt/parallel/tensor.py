"""Tensor-parallel sharding via DTensor's `parallelize_module`.

For each transformer Block we shard q/k/v column-parallel and o_proj
row-parallel; SwiGLU's gate/up (w1, w3) are column-parallel and the down
projection (w2) row-parallel. This keeps the activations sharded along the
hidden dim within a block and gathers naturally on the residual.

The token embedding is row-parallel along the vocab dimension (each rank
holds a slice of the embedding table) and the lm_head is also column-parallel
along the vocab dim. Critically, we set `output_layouts=Replicate()` on the
lm_head so the loss sees the full logits tensor — otherwise cross-entropy
silently computes on a sharded tensor.
"""
from __future__ import annotations
import torch.nn as nn


def apply_tp(model: nn.Module, tp_mesh) -> nn.Module:
    if tp_mesh is None or tp_mesh.size() == 1:
        return model
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel,
    )
    # Vocab-parallel embedding: rowwise on the vocab (input) dim, output replicated.
    parallelize_module(
        model.tok_emb, tp_mesh,
        RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
    )
    # lm_head is column-parallel on the vocab (output) dim, but we *gather*
    # on the way out so cross_entropy sees full [B, T, V] logits. Without the
    # explicit `output_layouts=Replicate()` the loss is computed on a sharded
    # tensor and the per-rank values are wrong by a factor of tp_size.
    parallelize_module(
        model.lm_head, tp_mesh,
        ColwiseParallel(output_layouts=Replicate()),
    )
    for blk in model.layers:
        plan = {
            "attn.q_proj": ColwiseParallel(),
            "attn.k_proj": ColwiseParallel(),
            "attn.v_proj": ColwiseParallel(),
            "attn.o_proj": RowwiseParallel(),
            "ffn.w1":      ColwiseParallel(),
            "ffn.w3":      ColwiseParallel(),
            "ffn.w2":      RowwiseParallel(),
        }
        parallelize_module(blk, tp_mesh, plan)
    return model
