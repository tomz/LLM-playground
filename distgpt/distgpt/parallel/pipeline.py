"""Pipeline parallelism: split layers across PP ranks, run 1F1B schedule.

This is a thin wrapper around `torch.distributed.pipelining`. We carve the model
into N stages by even layer split.
"""
from __future__ import annotations
import torch.nn as nn


def build_pipeline(model: nn.Module, pp_mesh, n_microbatches: int):
    """Return (stage, schedule). If pp size == 1, returns (model, None)."""
    if pp_mesh is None or pp_mesh.size() == 1:
        return model, None
    from torch.distributed.pipelining import ScheduleGPipe
    pp_size = pp_mesh.size()
    pp_rank = pp_mesh.get_local_rank()
    n_layers = len(model.layers)
    layers_per_stage = n_layers // pp_size
    # Trim to this rank's layers; keep emb on stage 0, head/norm on last stage.
    start = pp_rank * layers_per_stage
    end = (pp_rank + 1) * layers_per_stage if pp_rank < pp_size - 1 else n_layers
    model.layers = nn.ModuleList(model.layers[start:end])
    if pp_rank != 0:
        model.tok_emb = nn.Identity()  # type: ignore
    if pp_rank != pp_size - 1:
        model.lm_head = nn.Identity()  # type: ignore
        model.final_norm = nn.Identity()  # type: ignore
    # Build schedule (1F1B = ScheduleGPipe with interleaved=True for advanced; here GPipe).
    schedule = ScheduleGPipe(model, n_microbatches=n_microbatches)
    return model, schedule
