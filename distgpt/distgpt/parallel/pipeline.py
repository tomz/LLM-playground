"""Pipeline parallelism: split layers across PP ranks, run 1F1B schedule.

Thin wrapper around `torch.distributed.pipelining`. We carve the model into
`pp_size` even stages by layer count, then wrap each rank's trimmed module in
a `PipelineStage` so `Schedule1F1B` can drive it. Returns a pair
`(stage_module, schedule)`; the schedule expects to be called once per
**macro-batch**, with `n_microbatches` already configured.

The `PipelineStage` wrap is the part that was missing pre-fix — Schedule1F1B
reads `stage.num_stages`, which the raw GPT module obviously doesn't expose.
That bug went uncaught because nothing in CI actually invoked the PP path.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def build_pipeline(model: nn.Module, pp_mesh, n_microbatches: int,
                   example_input: torch.Tensor | None = None):
    """Return (stage, schedule). If pp size == 1, returns (model, None).

    ``example_input`` is optionally used by PipelineStage to trace the stage's
    output shape so downstream stages know the activation shape to recv. When
    omitted, PipelineStage auto-detects from the first forward; that path
    occasionally deadlocks on slow backends, so the caller may pre-provide
    a shape-correct dummy.
    """
    if pp_mesh is None or pp_mesh.size() == 1:
        return model, None
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B
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
    # Wrap in PipelineStage so Schedule1F1B has the (num_stages, stage_index)
    # metadata it needs. Without this wrap Schedule1F1B raises
    # `AttributeError: 'GPT' object has no attribute 'num_stages'`.
    device = next(model.parameters()).device
    stage = PipelineStage(
        submodule=model,
        stage_index=pp_rank,
        num_stages=pp_size,
        device=device,
        input_args=example_input,
        group=pp_mesh.get_group(),
    )
    # 1F1B (one forward / one backward) — better memory than GPipe at >1 stage.
    schedule = Schedule1F1B(stage, n_microbatches=n_microbatches)
    return model, schedule
