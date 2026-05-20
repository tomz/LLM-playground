"""Resumable streaming token loader over uint16 .bin shards.

Each rank gets a strided subset of shards (`files[rank::world_size]`). State
(`epoch`, `shard_idx`, `cursor`) is serializable so training can resume
mid-epoch with bit-exact data ordering per-rank.

Token packing
-------------
Sequences are packed contiguously: each `(x, y)` pair consumes `bs * (T+1)`
tokens (T input tokens plus 1 next-token target alignment), and we then
advance the cursor by `bs * (T+1)`. This makes consecutive batches strictly
non-overlapping, which is what most pretrain pipelines want.
"""
from __future__ import annotations
import glob, os
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class LoaderState:
    epoch: int = 0
    shard_idx: int = 0
    cursor: int = 0


class StreamingLoader:
    def __init__(
        self,
        shard_dir: str,
        seq_len: int,
        micro_batch: int,
        rank: int,
        world_size: int,
        seed: int,
        device: str,
    ):
        self.files = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
        if not self.files:
            raise FileNotFoundError(f"no .bin shards in {shard_dir}")
        self.seq_len = seq_len
        self.micro_batch = micro_batch
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.seed = seed
        self.state = LoaderState()
        # rank gets a strided subset of shards
        self.my_files = self.files[rank::world_size] or self.files

    def _load_shard(self, idx: int) -> np.ndarray:
        return np.memmap(self.my_files[idx % len(self.my_files)], dtype=np.uint16, mode="r")

    def next_batch(self):
        bs, T = self.micro_batch, self.seq_len
        chunk_tokens = bs * (T + 1)
        shard = self._load_shard(self.state.shard_idx)
        # If not enough room left, advance to next shard.
        if self.state.cursor + chunk_tokens > len(shard):
            self.state.shard_idx += 1
            self.state.cursor = 0
            if self.state.shard_idx % len(self.my_files) == 0:
                self.state.epoch += 1
            shard = self._load_shard(self.state.shard_idx)
        chunk = shard[self.state.cursor : self.state.cursor + chunk_tokens].astype(np.int64)
        # Advance by the full chunk so consecutive batches don't overlap. The
        # +1 stride for next-token targeting is handled within the chunk by the
        # bs*(T+1) reshape below.
        self.state.cursor += chunk_tokens
        chunk = chunk.reshape(bs, T + 1)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def state_dict(self) -> dict:
        s = self.state
        return dict(epoch=s.epoch, shard_idx=s.shard_idx, cursor=s.cursor,
                    rank=self.rank, world_size=self.world_size)

    def load_state_dict(self, sd: dict) -> None:
        self.state = LoaderState(
            epoch=sd["epoch"], shard_idx=sd["shard_idx"], cursor=sd["cursor"],
        )
        # Cross-check that we are restoring into the same topology.
        if "world_size" in sd and sd["world_size"] != self.world_size:
            raise ValueError(
                f"loader resume across topology change unsupported "
                f"(saved world_size={sd['world_size']}, current={self.world_size})"
            )
