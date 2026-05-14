"""Async, sharded, reshardable checkpoints."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CheckpointMeta:
    run_id: str
    step: int
    model_config_sha: str
    parallel_layout: dict   # {tp: 8, pp: 16, dp: 32}
    shard_uris: list[str]
    sha256: dict[str, str]


class CheckpointManager:
    def __init__(self, root_uri: str, run_id: str, keep_last: int = 10, milestone_every: int = 10_000):
        ...
    def save_async(self, engine, dataloader, step: int) -> None:
        """Snapshot to pinned host memory, return; background thread uploads."""
        raise NotImplementedError
    def load(self, step: int | str, target_layout: dict) -> CheckpointMeta:
        """Load + reshard to target_layout if needed."""
        raise NotImplementedError
    def latest(self) -> int | None:
        raise NotImplementedError
