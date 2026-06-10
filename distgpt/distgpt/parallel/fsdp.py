"""FSDP2 wrapping: shard each transformer block independently."""
from __future__ import annotations
import torch
import torch.nn as nn


def apply_fsdp(
    model: nn.Module,
    dp_mesh,
    dtype: torch.dtype,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Apply FSDP2 (`fully_shard`) to each Block, then to the whole model.

    ``reshard_after_forward`` (default True) controls the classic FSDP
    memory/comm tradeoff:

    * **True** (memory-light): params are freed after the forward all-gather
      and re-gathered again for the backward pass. Lowest peak VRAM, but on a
      fabric without NVLink (e.g. two consumer GPUs over PCIe with
      ``NCCL_P2P_DISABLE=1``) the doubled all-gather traffic can make FSDP
      *slower* than a single GPU.
    * **False** (comm-light): params stay resident (unsharded) after the
      forward, so the backward needs no re-gather. Costs one full unsharded
      param copy per GPU — negligible for a sub-1B model (~0.8 GB in bf16) —
      and roughly halves the per-step collective volume. Optimizer state is
      *still* sharded either way, so the FSDP memory win is preserved.

    Set it False in configs when the model comfortably fits and the
    interconnect is the bottleneck (see ``configs/gpt_416m_fweb_2gpu.yaml``).
    """
    if dp_mesh is None or dp_mesh.size() == 1:
        return model
    # FSDP2 API (PyTorch 2.4+)
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    # shard each transformer block first (better overlap), then the root
    for blk in model.layers:
        fully_shard(blk, mesh=dp_mesh, mp_policy=mp,
                    reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp,
                reshard_after_forward=reshard_after_forward)
    return model
