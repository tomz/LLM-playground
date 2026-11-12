"""Deterministic uint16 token shards with immutable manifests and DP ownership.

``rank`` and ``world_size`` are DATA-parallel coordinates, never global ranks.
TP peers use the same coordinates and therefore the same examples. Each DP
replica owns files[rank::world_size]; insufficient shards are an error, not a
fallback to duplicated training data. A batch consumes B*(T+1) tokens; short
shard tails are skipped. Full restart requires identical data and batch layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..utils.state import sha256_file

MANIFEST_SCHEMA = "distgpt.tokens/v1"
LOADER_SCHEMA = "distgpt.loader/v2"


def read_manifest(shard_dir: str | None, manifest_path: str | None = None) -> dict:
    """Snapshot a directory or verify every member of an explicit JSON manifest.

    Manifest paths may be absolute or relative to the manifest file. Verification
    is content-based, not just filenames/mtime. No dataset download is performed.
    """
    expected = None
    if manifest_path:
        source = Path(manifest_path)
        expected = json.loads(source.read_text())
        if expected.get("schema") != MANIFEST_SCHEMA or expected.get("dtype") != "uint16":
            raise ValueError("unsupported token manifest schema/dtype")
        entries = expected["shards"]
        paths = [(source.parent / item["path"]).resolve() for item in entries]
    else:
        if not shard_dir:
            raise ValueError("a shard directory or manifest is required")
        paths = sorted(p.resolve() for p in Path(shard_dir).glob("*.bin"))
    if not paths:
        raise FileNotFoundError(f"no .bin shards in {manifest_path or shard_dir}")
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate paths in token manifest")
    shards = []
    for i, path in enumerate(paths):
        size = path.stat().st_size
        if not path.is_file() or size == 0 or size % 2:
            raise ValueError(f"invalid uint16 shard size: {path}")
        digest = sha256_file(path)
        if expected is not None and (size != entries[i]["size_bytes"] or digest != entries[i]["sha256"]):
            raise ValueError(f"token manifest hash/size mismatch: {path}")
        shards.append({"path": str(path), "size_bytes": size, "sha256": digest})
    return {"schema": MANIFEST_SCHEMA, "dtype": "uint16", "shards": shards}


def assert_disjoint(train: "StreamingLoader", validation: "StreamingLoader") -> None:
    """Reject shared files AND byte-identical copies; not semantic deduplication."""
    training = train.manifest["shards"]
    held_out = validation.manifest["shards"]
    if ({s["path"] for s in training} & {s["path"] for s in held_out}
            or {s["sha256"] for s in training} & {s["sha256"] for s in held_out}):
        raise ValueError("validation overlaps training shards (same path or identical content)")


@dataclass
class LoaderState:
    epoch: int = 0
    shard_idx: int = 0
    cursor: int = 0


class StreamingLoader:
    """Resumable, immutable-corpus loader for one DP coordinate."""

    def __init__(self, shard_dir: str | None, seq_len: int, micro_batch: int,
                 rank: int, world_size: int, seed: int, device: str,
                 manifest_path: str | None = None):
        for name, value in (("seq_len", seq_len), ("micro_batch", micro_batch), ("world_size", world_size)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("invalid DP loader rank")
        self.manifest = read_manifest(shard_dir, manifest_path)
        self.files = [s["path"] for s in self.manifest["shards"]]
        if len(self.files) < world_size:
            raise ValueError(f"need at least {world_size} shards for DP={world_size}; got {len(self.files)} (no replicated fallback)")
        self.seq_len, self.micro_batch = seq_len, micro_batch
        self.rank, self.world_size = rank, world_size
        self.device, self.seed = str(device), seed
        self.state = LoaderState()
        self.my_files = self.files[rank::world_size]
        self._stats = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in self.files}
        self.manifest_sha256 = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()
        if not any(self._stats[p][0] // 2 >= self.batch_tokens for p in self.my_files):
            raise ValueError(f"DP rank {rank} has no shard large enough for a batch of {self.batch_tokens} tokens")

    @property
    def batch_tokens(self) -> int:
        return self.micro_batch * (self.seq_len + 1)

    def _load_shard(self, idx: int) -> np.ndarray:
        path = self.my_files[idx % len(self.my_files)]
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != self._stats[path]:
            raise ValueError(f"immutable token shard changed during run: {path}")
        return np.memmap(path, dtype=np.uint16, mode="r")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(len(self.my_files) + 1):
            shard = self._load_shard(self.state.shard_idx)
            if self.state.cursor + self.batch_tokens <= len(shard):
                break
            self.state.shard_idx += 1
            self.state.cursor = 0
            self.state.epoch = self.state.shard_idx // len(self.my_files)
        else:
            raise ValueError("no complete token batch remains in this DP shard set")
        chunk = shard[self.state.cursor:self.state.cursor + self.batch_tokens].astype(np.int64)
        self.state.cursor += self.batch_tokens
        chunk = chunk.reshape(self.micro_batch, self.seq_len + 1)
        x, y = torch.from_numpy(chunk[:, :-1]), torch.from_numpy(chunk[:, 1:])
        if self.device.startswith("cuda"):
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)

    def state_dict(self) -> dict:
        return {"schema": LOADER_SCHEMA, "epoch": self.state.epoch,
                "shard_idx": self.state.shard_idx, "cursor": self.state.cursor,
                "rank": self.rank, "world_size": self.world_size, "seed": self.seed,
                "seq_len": self.seq_len, "micro_batch": self.micro_batch,
                "manifest_sha256": self.manifest_sha256, "manifest": self.manifest}

    def validate_state_dict(self, state: dict) -> None:
        """Check every invariant BEFORE mutating the live cursor."""
        current = self.state_dict()
        for key in ("schema", "rank", "world_size", "seq_len", "micro_batch", "seed", "manifest_sha256", "manifest"):
            if state.get(key) != current[key]:
                raise ValueError(f"loader resume {key} mismatch; data/topology/layout changes require a warm start")
        for key in ("epoch", "shard_idx", "cursor"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f"invalid loader {key}")
        if state["epoch"] != state["shard_idx"] // len(self.my_files):
            raise ValueError("inconsistent loader epoch and shard index")
        path = self.my_files[state["shard_idx"] % len(self.my_files)]
        if state["cursor"] > self._stats[path][0] // 2 or state["cursor"] % self.batch_tokens:
            raise ValueError("invalid or unaligned loader cursor")

    def load_state_dict(self, state: dict) -> None:
        self.validate_state_dict(state)
        self.state = LoaderState(state["epoch"], state["shard_idx"], state["cursor"])
