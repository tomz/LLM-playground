"""Memory-mapped uint16 token shards. Random window sampling."""
import os, pickle
import numpy as np
import torch


class ShardDataset:
    def __init__(self, data_dir: str, split: str, block_size: int, device: str):
        path = os.path.join(data_dir, f"{split}.bin")
        # mmap so we don't load the whole shard into RAM
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
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
