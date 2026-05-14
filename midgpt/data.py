"""Memory-mapped uint16 token shards. DDP-aware deterministic sampling."""
from __future__ import annotations
import os, glob
import numpy as np
import torch


class ShardDataset:
    """Concatenates all *.bin shards in a directory into one logical token stream."""
    def __init__(self, shard_dir: str, block_size: int, device: str, rank: int = 0, world_size: int = 1):
        files = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
        if not files:
            raise FileNotFoundError(f"no shards in {shard_dir}; run prepare.py first")
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.sizes = [len(s) for s in self.shards]
        self.total = sum(self.sizes)
        self.block_size = block_size
        self.device = device
        self.rank = rank
        self.world_size = world_size

    def _locate(self, idx: int) -> tuple[int, int]:
        for s, n in enumerate(self.sizes):
            if idx < n - self.block_size - 1:
                return s, idx
            idx -= n
        # fallback to first shard
        return 0, 0

    def get_batch(self, batch_size: int, generator: torch.Generator):
        # rank-disjoint sampling: each rank pulls from a different RNG stream
        ix = torch.randint(0, self.total - self.block_size - 1, (batch_size,), generator=generator).tolist()
        xs, ys = [], []
        for i in ix:
            s, off = self._locate(i)
            buf = self.shards[s][off : off + self.block_size + 1].astype(np.int64)
            xs.append(torch.from_numpy(buf[:-1]))
            ys.append(torch.from_numpy(buf[1:]))
        x = torch.stack(xs); y = torch.stack(ys)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
