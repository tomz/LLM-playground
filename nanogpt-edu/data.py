"""Memory-mapped uint16 shards, measured byte counts, and split-manifest checks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import torch

MANIFEST_SCHEMA = "nanogpt.document-shards/v1"
SPLIT_ROLES = {"train": "training", "val": "search_validation", "holdout": "final_holdout"}


def sha256_file(path: str | Path) -> str:
    """Hash artifacts with bounded memory, including large training shards."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_dir: str | Path, *, splits: tuple[str, ...], meta: dict) -> dict | None:
    """Validate identities and selected artifacts without opening other splits.

    Search callers request train/val only. The holdout's role/counts are public
    metadata, but its tokens and document ledger are never opened by search.
    Legacy directories without a manifest remain usable, unless metadata says
    a manifest was prepared (a missing manifest then signals incomplete data).
    """
    root = Path(data_dir)
    path = root / "manifest.json"
    if not path.exists():
        if "manifest_sha256" in meta or "data_manifest_schema" in meta:
            raise ValueError("prepared data manifest is missing")
        return None
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported data manifest schema")
    if meta.get("manifest_sha256") not in (None, hashlib.sha256(raw).hexdigest()):
        raise ValueError("data manifest identity differs from meta.pkl")
    if manifest.get("dtype") != "<u2":
        raise ValueError("manifest must describe little-endian uint16 tokens")
    tokenizer = manifest["tokenizer"]
    if tokenizer["n_vocab"] != meta["vocab_size"] or tokenizer["name"] != meta.get("tokenizer"):
        raise ValueError("tokenizer identity differs from meta.pkl")
    if set(manifest["splits"]) != set(SPLIT_ROLES):
        raise ValueError("manifest must declare train, val and holdout")
    for split, role in SPLIT_ROLES.items():
        if manifest["splits"][split]["role"] != role:
            raise ValueError(f"incorrect evaluation role for {split}")
    for split in splits:
        info = manifest["splits"][split]
        count, nbytes = info["tokens"], info["utf8_bytes"]
        if type(count) is not int or count <= 0 or type(nbytes) is not int or nbytes <= 0:
            raise ValueError(f"{split} is empty or has invalid byte/token counts; collect more documents")
        ratio = info["bytes_per_token"]
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio) or not math.isclose(ratio, nbytes / count, rel_tol=1e-12)):
            raise ValueError(f"invalid measured bytes_per_token for {split}")
        if meta.get("bytes_per_token", {}).get(split) != ratio:
            raise ValueError(f"{split} normalization differs from meta.pkl")
        for key, filename in (("artifact", f"{split}.bin"), ("documents", f"{split}.documents.jsonl")):
            artifact = info[key]
            if artifact["path"] != filename:
                raise ValueError(f"unexpected {split} artifact path")
            target = root / filename
            if target.stat().st_size != artifact["size_bytes"] or sha256_file(target) != artifact["sha256"]:
                raise ValueError(f"{filename} artifact identity mismatch")
        if info["artifact"]["size_bytes"] != 2 * count:
            raise ValueError(f"{split} token count differs from artifact size")
    return manifest


def character_bytes_per_token(data_dir: str | Path, split: str, meta: dict) -> float:
    """Measure UTF-8 bytes from legacy character IDs, weighted by occurrences.

    An unweighted average over the vocabulary is not a corpus byte ratio. This
    scan uses a fixed-size lookup table and bounded token chunks, not text
    reconstruction or a full-shard allocation.
    """
    chars = meta["itos"]
    items = chars.items() if isinstance(chars, dict) else enumerate(chars)
    lengths = np.zeros(65536, dtype=np.int64)
    for token, char in items:
        if type(token) is not int or not 0 <= token < 65536 or not isinstance(char, str) or len(char) != 1:
            raise ValueError("invalid legacy character vocabulary")
        lengths[token] = len(char.encode("utf-8"))
    path = Path(data_dir) / f"{split}.bin"
    if path.stat().st_size == 0 or path.stat().st_size % 2:
        raise ValueError(f"{split} must contain nonempty uint16 tokens")
    count = nbytes = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 65536), b""):
            ids = np.frombuffer(chunk, dtype="<u2")
            sizes = lengths[ids]
            if np.any(sizes == 0):
                raise ValueError(f"{split} contains IDs absent from the character vocabulary")
            count += len(ids)
            nbytes += int(sizes.sum())
    return nbytes / count


class ShardDataset:
    """Sample random windows from one shard using an explicit independent RNG."""

    def __init__(self, data_dir: str, split: str, block_size: int, device: str):
        path = os.path.join(data_dir, f"{split}.bin")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        size = os.path.getsize(path)
        if size % 2 or size // 2 < block_size + 1:
            raise ValueError(f"{split} needs at least block_size + 1 uint16 tokens")
        self.data = np.memmap(path, dtype="<u2", mode="r")
        self.block_size = block_size
        self.device = device

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None):
        # Valid starts include len(data)-block_size-1; randint's high is exclusive.
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def load_meta(data_dir: str) -> dict:
    """Load trusted local tokenizer metadata; pickle is not an untrusted format."""
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)
