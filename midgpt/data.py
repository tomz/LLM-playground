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
                 split: str = "train"):
        """Build a token-stream view over ``shard_root/<split>/*.bin``.

        Sampling is *unsharded* — every caller draws from the same global
        index space. Per-rank disjointness in DDP comes from each rank
        seeding its own ``torch.Generator`` differently (see ``train.py``);
        nothing in this class is rank-aware. The old ``rank=`` / ``world_size=``
        kwargs were dead and were removed in Tier 6.1.
        """
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

    def _locate(self, idx: int) -> tuple[int, int] | None:
        """Map a global token index to (shard, offset). Returns None if the
        landing spot doesn't have room for a full block — the caller should
        re-sample. Returning None (instead of the old ``off = 0`` clamp) avoids
        biasing the sample distribution toward the first ``block_size+1`` tokens
        of every shard, which over a long run measurably skews the trained
        token mix toward whatever happens to live at each shard's start.
        """
        # O(log num_shards) instead of O(num_shards).
        s = int(np.searchsorted(self.cumsum, idx, side="right") - 1)
        s = max(0, min(s, len(self.shards) - 1))
        off = idx - int(self.cumsum[s])
        if off + self.block_size + 1 > len(self.shards[s]):
            return None
        return s, off

    def get_batch(self, batch_size: int, generator: torch.Generator):
        hi = max(1, self.total - self.block_size - 1)
        xs, ys = [], []
        # Re-sample any index that lands in the last `block_size+1` tokens of
        # some shard (where _locate would otherwise return None). At the typical
        # shard sizes (>=1M tokens) this rejects <0.1% of samples per shard end.
        while len(xs) < batch_size:
            need = batch_size - len(xs)
            ix = torch.randint(0, hi, (need,), generator=generator).tolist()
            for i in ix:
                loc = self._locate(i)
                if loc is None:
                    continue
                s, off = loc
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
