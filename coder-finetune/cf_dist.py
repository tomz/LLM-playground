"""Distributed-env helpers for the TRL/accelerate post-training entry points.

coder-finetune does **not** own its training loop — TRL's ``Trainer`` (built on
HuggingFace ``accelerate``) does. accelerate, not this code, calls
``init_process_group`` and pins devices when you launch under ``torchrun`` or
``accelerate launch``. So the right thing for *our* code to do is *read* the
distributed topology accelerate/torchrun has already published into the
environment — never re-derive it, and in particular never guess it from
``torch.cuda.device_count()``, which counts GPUs visible to *this process*, not
the size of the distributed group.

That distinction is the whole reason this module exists. ``device_count()`` and
the real world size diverge in both directions:

  * **single process, many visible GPUs** (``python train.py`` on a 2-GPU box):
    ``device_count() == 2`` but ``WORLD_SIZE`` is unset, so the job is really
    one process — world size 1.
  * **one process per GPU** (``torchrun --nproc_per_node 2`` with each rank
    pinned to one card): ``device_count() == 1`` per process but ``WORLD_SIZE
    == 2``.

accelerate uses ``WORLD_SIZE`` (its ``num_processes``); so must we.

Launch examples that publish these env vars::

    accelerate launch --multi_gpu --num_processes 2 train.py --config ...
    torchrun --standalone --nproc_per_node 2 -m cf_rl.grpo_train --config ...

The three numbers we read:

    WORLD_SIZE  — total processes in the job (== accelerate ``num_processes``);
                  1 for a plain ``python ...`` launch.
    RANK        — global index of this process in ``[0, WORLD_SIZE)``.
    LOCAL_RANK  — index of this process on its node; the GPU ordinal to pin.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistEnv:
    """A read-only snapshot of the distributed topology from the environment.

    Frozen because it is a *view* of what the launcher published — nothing in
    this codebase should mutate the process topology (accelerate owns that).
    """

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        """True only on global rank 0 — the gate for print/log de-duplication."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """True when the job spans more than one process (real DDP)."""
        return self.world_size > 1


def dist_env() -> DistEnv:
    """Read the distributed topology accelerate/torchrun published to the env.

    Pure read-only: we never call ``init_process_group`` — accelerate owns the
    process-group lifecycle. Every field falls back to the single-process
    default, so a plain ``python train.py`` run gets ``rank=0, local_rank=0,
    world_size=1`` and every downstream helper behaves exactly as it did before
    this module existed.
    """
    return DistEnv(
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=_env_int("WORLD_SIZE", 1),
    )


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, tolerating empty/garbage values as the default.

    Some launchers export an empty ``RANK=`` rather than unsetting it; treat
    anything non-integer as "not set" instead of crashing the entry point.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def rank0_print(*args, **kwargs) -> None:
    """``print()`` that fires only on global rank 0.

    Under DDP every status line would otherwise be emitted ``WORLD_SIZE`` times,
    interleaved across ranks. On a single-process run ``is_main`` is True, so
    this is byte-for-byte identical to ``print`` — the documented single-GPU
    behavior is unchanged.
    """
    if dist_env().is_main:
        print(*args, **kwargs)


def placement_device_map(env: DistEnv | None = None) -> dict[str, int] | None:
    """Per-rank ``device_map`` for quantized (QLoRA) loads under DDP.

    bitsandbytes places 4-bit weights on a concrete device *at load time* and
    accelerate cannot relocate already-quantized weights afterwards. So under
    DDP each rank must pin the quantized model to its own local GPU; otherwise
    every process either piles onto ``cuda:0`` (OOM) or trips transformers'
    "can't train a model loaded with ``device_map='auto'`` in distributed mode".

    Returns ``{"": local_rank}`` when the job is distributed, else ``None`` —
    on a single process we leave placement to ``Trainer``/accelerate, which is
    the current (working) single-GPU path. **Only the quantized path should use
    this**: plain LoRA / full FT under DDP must let ``accelerate.prepare()`` do
    the ``.to(device)`` move and must not be pre-empted with a ``device_map``.
    """
    env = env or dist_env()
    if env.is_distributed:
        return {"": env.local_rank}
    return None
