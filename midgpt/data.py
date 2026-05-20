"""Memory-mapped uint16 token shards. DDP-aware deterministic sampling.

Train and val live in separate subdirectories so they cannot leak into each
other:
    data/<dataset>/train/*.bin
    data/<dataset>/val/*.bin
For backwards compatibility we fall back to a flat directory of *.bin files
and pattern-match on `train_*.bin` / `val_*.bin` filenames.
"""
from __future__ import annotations
import os, glob
import numpy as np
import torch


class ShardDataset:
    """Concatenates all *.bin shards in a directory into one logical token stream.

    `split` selects either the `train/` or `val/` subdir (preferred); if those
    don't exist it falls back to the flat-directory layout written by older
    versions of `prepare.py` (`{split}_*.bin` in the same directory).
    """
    def __init__(self, shard_root: str, block_size: int, device: str,
                 rank: int = 0, world_size: int = 1, split: str = "train"):
        sub = os.path.join(shard_root, split)
        if os.path.isdir(sub):
            files = sorted(glob.glob(os.path.join(sub, "*.bin")))
        else:
            files = sorted(glob.glob(os.path.join(shard_root, f"{split}_*.bin")))
            if not files:
                # last-ditch: any .bin in the dir
                files = sorted(glob.glob(os.path.join(shard_root, "*.bin")))
        if not files:
            raise FileNotFoundError(
                f"no shards in {shard_root!r} for split={split!r}; run prepare.py first"
            )
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        sizes = np.asarray([len(s) for s in self.shards], dtype=np.int64)
        self.cumsum = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.cumsum[-1])
        self.block_size = block_size
        self.device = device
        self.rank = rank
        self.world_size = world_size

    def _locate(self, idx: int) -> tuple[int, int]:
        # O(log num_shards) instead of O(num_shards).
        s = int(np.searchsorted(self.cumsum, idx, side="right") - 1)
        s = max(0, min(s, len(self.shards) - 1))
        off = idx - int(self.cumsum[s])
        if off + self.block_size + 1 > len(self.shards[s]):
            # Fall back to the start of this shard if we landed too close to end.
            off = 0
        return s, off

    def get_batch(self, batch_size: int, generator: torch.Generator):
        ix = torch.randint(0, max(1, self.total - self.block_size - 1),
                           (batch_size,), generator=generator).tolist()
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
