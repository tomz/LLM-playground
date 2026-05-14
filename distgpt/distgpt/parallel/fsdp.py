"""FSDP2 wrapping: shard each transformer block independently."""
from __future__ import annotations
import torch
import torch.nn as nn


def apply_fsdp(model: nn.Module, dp_mesh, dtype: torch.dtype) -> nn.Module:
    """Apply FSDP2 (`fully_shard`) to each Block, then to the whole model."""
    if dp_mesh is None or dp_mesh.size() == 1:
        return model
    # FSDP2 API (PyTorch 2.4+)
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    # shard each transformer block first (better overlap), then the root
    for blk in model.layers:
        fully_shard(blk, mesh=dp_mesh, mp_policy=mp)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp)
    return model
