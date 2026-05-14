"""Resumable, deterministic, distributed streaming dataloader.

Contract:
  • Given (rank, world_size, global_step, seed) the next batch is fully determined.
  • State for resume = (epoch, shard_idx, intra_shard_offset, rng_state) per rank.
  • No host-side shuffle buffer larger than O(seq_len * batch).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class LoaderState:
    epoch: int
    shard_idx: int
    intra_shard_offset: int
    rng_state: bytes


class StreamingLoader:
    def __init__(self, mixture, seq_len: int, micro_batch: int, rank: int, world_size: int, seed: int):
        ...
    def __iter__(self):
        raise NotImplementedError
    def state_dict(self) -> LoaderState:
        raise NotImplementedError
    def load_state_dict(self, s: LoaderState) -> None:
        raise NotImplementedError
