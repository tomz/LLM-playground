"""Tensor-parallel sharding via DTensor's `parallelize_module`, with optional
sequence parallelism (SP) on the norm + residual path.

What we always shard (TP)
-------------------------
For each transformer Block:
  * q_proj, k_proj, v_proj   → ColwiseParallel  (output sharded on head dim)
  * o_proj                   → RowwiseParallel  (input sharded, output reduced)
  * ffn.w1, ffn.w3 (SwiGLU)  → ColwiseParallel
  * ffn.w2                   → RowwiseParallel  (reduces back to full hidden)

At the model root:
  * tok_emb                  → vocab-parallel embedding (rowwise on vocab dim)
  * lm_head                  → vocab-parallel linear   (colwise on vocab dim,
                                                       gathered for CE loss)

Sequence parallelism (`sequence_parallel=True`)
-----------------------------------------------
When SP is enabled, the RMSNorm layers in each block are wrapped with
`SequenceParallel` so their inputs/outputs stay sharded along the sequence
dimension. This converts the redundant per-rank LayerNorm/RMSNorm work into
1/tp_size cost AND cuts activation memory at the norm boundary by tp_size.
SP is a pure memory + compute win on long context (T >> d/tp) and a small
LOSS on short context (the extra all-gather before the colwise matmul
swamps the savings). Rule of thumb: enable when seq_len >= 4096.

The activation flow inside a block becomes:

    Shard(seq) ── attn_norm (SP) ── Shard(seq)
                                  ↓ ColwiseParallel.input_layouts gathers
                                  ↓ q/k/v compute, attention, o_proj
                                  ↓ RowwiseParallel.output_layouts re-shards
    Shard(seq) ── residual + ──── Shard(seq)
                  ffn_norm (SP)
                                  ↓ gather → w1/w3 → w2 → re-shard
    Shard(seq) ── residual + ──── Shard(seq)

The model-root layouts adapt so the SP boundary lives at the embed→block
and final-norm→lm_head interfaces:

    tok_emb : output_layouts = Shard(seq)         (was Replicate)
    lm_head : input_layouts  = Shard(seq), output_layouts = Replicate

What we don't do
----------------
The final `model.final_norm` is left as a plain (replicated) norm — its
output feeds `lm_head` which expects full `Shard(seq)` input. The marginal
gain from SP-ing one more norm is tiny next to the cost of an extra gather
on the long axis. Same call we made in the writeup design notes.

Convenience marker
------------------
We tag the model with `_dgpt_sp_enabled: bool` so downstream code (logging,
FSDP policy) can detect SP. The name deliberately AVOIDS the `_dist*`
prefix — torch's distributed.checkpoint module monkey-patches
`nn.Module.__getattr__` so any attribute name starting with `_dist*`
silently returns False instead of raising AttributeError, which made an
earlier `_distgpt_sequence_parallel` attribute appear to always be False
even after assignment. See tests/test_pytorch_quirks.py.
"""
from __future__ import annotations
import torch.nn as nn


def apply_tp(model: nn.Module, tp_mesh,
              sequence_parallel: bool = False) -> nn.Module:
    """Apply tensor parallelism (with optional sequence parallelism).

    Args:
      model: distgpt GPT instance (not yet sharded)
      tp_mesh: DTensor sub-mesh for the TP dimension, or None for tp=1 (no-op)
      sequence_parallel: when True, wrap norm layers with `SequenceParallel`
        and switch embed/lm_head layouts to sequence-sharded.

    Returns:
      The same model object (parallelize_module mutates in place). The model
      is tagged with `_dgpt_sp_enabled` reflecting the effective SP setting
      (False at tp=1 even if `sequence_parallel=True` was passed, because
      the function short-circuits before any sharding).
    """
    if tp_mesh is None or tp_mesh.size() == 1:
        return model
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module,
    )

    # ---- Root: embedding + lm_head ----
    if sequence_parallel:
        # Embedding emits sequence-sharded activations directly into the
        # first SP-wrapped norm; lm_head consumes sequence-sharded input
        # from the final norm and gathers logits for CE loss.
        parallelize_module(
            model.tok_emb, tp_mesh,
            RowwiseParallel(input_layouts=Replicate(),
                            output_layouts=Shard(1)),
        )
        parallelize_module(
            model.lm_head, tp_mesh,
            ColwiseParallel(input_layouts=Shard(1),
                            output_layouts=Replicate()),
        )
    else:
        # Non-SP: original layouts. Output gathered so cross_entropy sees
        # full [B, T, V] (the silent-wrong-loss footgun from the docstring).
        parallelize_module(
            model.tok_emb, tp_mesh,
            RowwiseParallel(input_layouts=Replicate(),
                            output_layouts=Replicate()),
        )
        parallelize_module(
            model.lm_head, tp_mesh,
            ColwiseParallel(output_layouts=Replicate()),
        )

    # ---- Per-block: linears (always sharded) + norms (SP-wrapped if on) ----
    for blk in model.layers:
        plan: dict = {
            "attn.q_proj": ColwiseParallel(),
            "attn.k_proj": ColwiseParallel(),
            "attn.v_proj": ColwiseParallel(),
            "attn.o_proj": RowwiseParallel(
                # Re-shard the residual back to Shard(seq) so the *next*
                # block's SP-wrapped norm receives sequence-sharded input.
                output_layouts=Shard(1) if sequence_parallel else None,
            ),
            "ffn.w1": ColwiseParallel(),
            "ffn.w3": ColwiseParallel(),
            "ffn.w2": RowwiseParallel(
                output_layouts=Shard(1) if sequence_parallel else None,
            ),
        }
        if sequence_parallel:
            # The two block-internal norms operate on Shard(seq) inputs;
            # SP makes the norm itself a per-rank-local op over the 1/tp
            # slice of the sequence dimension. Each colwise linear that
            # follows does its own gather automatically.
            plan["attn_norm"] = SequenceParallel()
            plan["ffn_norm"] = SequenceParallel()
        parallelize_module(blk, tp_mesh, plan)

    # See module docstring for why this attr name avoids the `_dist*` prefix.
    model._dgpt_sp_enabled = bool(sequence_parallel)
    return model
