"""Tensor-parallel sharding via DTensor's `parallelize_module`.

We shard each Block's q/k/v/o_proj and the SwiGLU's w1/w2/w3 column- or
row-parallel; embedding and lm_head are sharded too.
"""
from __future__ import annotations
import torch.nn as nn


def apply_tp(model: nn.Module, tp_mesh) -> nn.Module:
    if tp_mesh is None or tp_mesh.size() == 1:
        return model
    from torch.distributed.tensor.parallel import (
        parallelize_module, ColwiseParallel, RowwiseParallel,
    )
    # Embedding rowwise (vocab dim sharded): inputs are token ids replicated.
    parallelize_module(model.tok_emb, tp_mesh, RowwiseParallel())
    parallelize_module(model.lm_head, tp_mesh, ColwiseParallel())
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
