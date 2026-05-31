"""Sharded data loader: locate() bias, batch shapes, x/y alignment."""
from __future__ import annotations
import sys, pathlib
import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data import ShardDataset


def _make_shards(tmp_path: pathlib.Path, sizes: list[int], split: str = "train") -> pathlib.Path:
    """Write each shard with a recognizable pattern so we can assert positions
    survive end-to-end. Shard *s* token at offset *o* is encoded as
    ``(s * 1_000_003 + o) % 65521`` (a 16-bit prime modulus that fits in uint16).
    """
    root = tmp_path / "ds" / split
    root.mkdir(parents=True)
    for s, n in enumerate(sizes):
        arr = ((s * 1_000_003 + np.arange(n, dtype=np.int64)) % 65521).astype(np.uint16)
        arr.tofile(str(root / f"shard_{s:06d}.bin"))
    return tmp_path / "ds"


def test_locate_never_clamps_to_zero(tmp_path: pathlib.Path):
    """Regression: the old _locate() fell back to ``off=0`` whenever the random
    landing point couldn't fit a full block, biasing the sample distribution
    toward shard-start tokens. The new contract returns None so the caller
    re-samples."""
    root = _make_shards(tmp_path, sizes=[200, 200])
    ds = ShardDataset(str(root), block_size=64, device="cpu", split="train")
    # Last 64 indices of each shard should map to None (no room for block+1).
    # Need off+block_size+1 <= len(shard); shard 0 has len 200, block_size=64,
    # so max valid off is 200-64-1 = 135 → off in [136, 199] return None.
    for i in range(136, 200):
        assert ds._locate(i) is None, i
    # And the last valid index (135) *does* fit.
    s, off = ds._locate(135)
    assert s == 0 and off == 135
    # Crossing the boundary into shard 1: index 200 lands at shard 1 offset 0.
    s, off = ds._locate(200)
    assert s == 1 and off == 0


def test_get_batch_xy_alignment(tmp_path: pathlib.Path):
    """y must be x shifted left by one — that's the next-token-prediction
    invariant the whole training loop depends on."""
    root = _make_shards(tmp_path, sizes=[1024, 1024])
    ds = ShardDataset(str(root), block_size=16, device="cpu", split="train")
    gen = torch.Generator(); gen.manual_seed(0)
    x, y = ds.get_batch(batch_size=8, generator=gen)
    assert x.shape == (8, 16)
    assert y.shape == (8, 16)
    # y[i, t] == x[i, t+1] for all interior positions.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_sampling_distribution_unbiased(tmp_path: pathlib.Path):
    """Draw many batches; the empirical first-position distribution should be
    close to uniform over each shard's valid window. The OLD ``off=0`` fallback
    would give a huge spike at the shard-start tokens (~14% of samples instead
    of ~0.16% per token); we assert no single token gets >5× the mean count."""
    # Two shards of 500 tokens, block_size=64 → 437 valid offsets per shard,
    # 874 valid positions total. Draw enough samples that the binomial std is
    # small compared to 5x mean.
    root = _make_shards(tmp_path, sizes=[500, 500])
    ds = ShardDataset(str(root), block_size=64, device="cpu", split="train")
    gen = torch.Generator(); gen.manual_seed(0)
    n_samples = 8_000
    starts: list[int] = []
    while len(starts) < n_samples:
        x, _ = ds.get_batch(batch_size=64, generator=gen)
        # Decode shard+offset from the first token of each row (we wrote a
        # deterministic pattern so we can do this).
        for first in x[:, 0].tolist():
            starts.append(int(first))
    counts = np.bincount(np.asarray(starts), minlength=65521)
    nz = counts[counts > 0]
    mean = nz.mean()
    # Each of the 874 valid positions has its own unique "first token" value.
    # With ~9 samples/position on average, a 5× cap rules out the old bias.
    assert nz.max() < 5 * mean, (nz.max(), mean)


def test_split_subdir_layout(tmp_path: pathlib.Path):
    """Train and val live in separate subdirectories — verify the loader
    picks the right one and isn't fooled by stray files in the parent dir."""
    root = _make_shards(tmp_path, sizes=[1024], split="train")
    # Also write a val shard.
    _make_shards(tmp_path, sizes=[512], split="val")
    train_ds = ShardDataset(str(root), block_size=16, device="cpu", split="train")
    val_ds = ShardDataset(str(root), block_size=16, device="cpu", split="val")
    assert train_ds.total == 1024
    assert val_ds.total == 512


def test_missing_split_raises(tmp_path: pathlib.Path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        ShardDataset(str(root), block_size=16, device="cpu", split="train")
