"""Memory-mapped uint16 token shards. Random window sampling."""
import os, pickle
import numpy as np
import torch


class ShardDataset:
    """Concatenates a single `{split}.bin` shard into one logical token stream.

    Sampling is intentionally tiny: a uniform random offset, then a contiguous
    block. Callers should pass an explicit ``generator`` so the training and
    eval RNGs stay independent of ``torch.default_generator`` (which is also
    consumed by model init / dropout); without that, val-batch draws drift
    between resumes.
    """
    def __init__(self, data_dir: str, split: str, block_size: int, device: str):
        path = os.path.join(data_dir, f"{split}.bin")
        # mmap so we don't load the whole shard into RAM
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        # randint's upper bound is *exclusive*: drawing from [0, len-block-1)
        # guarantees idx+block_size+1 ≤ len(data), so the y-shift is always
        # in-bounds. No need for a re-sample loop (midgpt's _locate fix doesn't
        # apply here — that bug came from a multi-shard cumsum edge).
        ix = torch.randint(
            len(self.data) - self.block_size - 1, (batch_size,), generator=generator
        )
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)
