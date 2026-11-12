"""Resumable, deterministic, streaming dataloader over uint32 .bin shards.

Reads shards picked by a `MixtureSampler` (not a glob); the per-rank state
(epoch, shard_idx, intra_shard_offset, rng_state) is JSON-serializable so
training can resume mid-epoch.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoaderState:
    epoch: int = 0
    shard_idx: int = 0
    intra_shard_offset: int = 0
    rng_state: list = field(default_factory=list)
    step: int = 0
    current_shard: str | None = None
    shard_pick_step: int = 0


class StreamingLoader:
    def __init__(
        self,
        mixture,
        seq_len: int,
        micro_batch: int,
        rank: int,
        world_size: int,
        seed: int,
    ):
        self.mixture = mixture
        self.seq_len = int(seq_len)
        self.micro_batch = int(micro_batch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.rng = random.Random(seed * 1000003 + rank)
        self.state = LoaderState()
        self._current_shard: np.ndarray | None = None
        self._current_path: str | None = None

    def _ensure_shard(self) -> np.ndarray:
        if self._current_shard is None:
            if self.state.current_shard is None:
                path = self.mixture.next_shard(self.rank, self.state.shard_pick_step)
                self.state.current_shard = path
                self.state.shard_pick_step += 1
            self._current_shard = np.memmap(self.state.current_shard, dtype=np.uint32, mode="r")
            self._current_path = self.state.current_shard
        return self._current_shard

    def _advance_shard(self) -> None:
        self.state.shard_idx += 1
        self.state.intra_shard_offset = 0
        self.state.current_shard = None
        self._current_shard = None

    def __iter__(self):
        bs, T = self.micro_batch, self.seq_len
        need = bs * (T + 1)
        while True:
            shard = self._ensure_shard()
            if self.state.intra_shard_offset + need > len(shard):
                self._advance_shard()
                continue
            start = self.state.intra_shard_offset
            chunk = np.asarray(shard[start : start + need], dtype=np.int64).reshape(bs, T + 1)
            self.state.intra_shard_offset += bs * T
            self.state.step += 1
            x = chunk[:, :-1].copy()
            y = chunk[:, 1:].copy()
            yield x, y

    def state_dict(self) -> dict:
        return {
            "epoch": self.state.epoch,
            "shard_idx": self.state.shard_idx,
            "intra_shard_offset": self.state.intra_shard_offset,
            "step": self.state.step,
            "current_shard": self.state.current_shard,
            "shard_pick_step": self.state.shard_pick_step,
            "rng_state": list(self.rng.getstate()[1]),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.state = LoaderState(
            epoch=sd["epoch"],
            shard_idx=sd["shard_idx"],
            intra_shard_offset=sd["intra_shard_offset"],
            step=sd.get("step", 0),
            current_shard=sd.get("current_shard"),
            shard_pick_step=sd.get("shard_pick_step", 0),
            rng_state=list(sd.get("rng_state", [])),
        )
        self._current_shard = None
