"""Domain mixing weights & sampler."""
from __future__ import annotations
import glob
import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DomainSpec:
    name: str
    shard_glob: str
    weight: float
    epochs_cap: float = 4.0


@dataclass
class _DomainState:
    shards: list[str] = field(default_factory=list)
    visits: int = 0  # number of times we've handed out a shard from this domain


class MixtureSampler:
    """Multinomial sampler over domains; per-domain shuffled shard iterator.

    Deterministic given (global_seed, rank, step).
    """
    def __init__(self, specs: list[DomainSpec], global_seed: int):
        self.specs = list(specs)
        self.seed = int(global_seed)
        total = sum(s.weight for s in self.specs)
        if total <= 0:
            raise ValueError("sum of domain weights must be > 0")
        self.weights = np.array([s.weight / total for s in self.specs], dtype=np.float64)
        self._states: list[_DomainState] = []
        for s in self.specs:
            shards = sorted(glob.glob(s.shard_glob))
            self._states.append(_DomainState(shards=shards))
        self._exhausted: set[int] = set()

    def _rng(self, rank: int, step: int) -> np.random.Generator:
        key = hashlib.blake2b(struct.pack("<qqq", self.seed, rank, step), digest_size=16).digest()
        seed = int.from_bytes(key[:8], "little")
        return np.random.default_rng(seed)

    def next_shard(self, rank: int, step: int) -> str:
        if len(self._exhausted) >= len(self.specs):
            raise StopIteration("all domains hit their epochs_cap")
        rng = self._rng(rank, step)
        weights = self.weights.copy()
        for i in self._exhausted:
            weights[i] = 0.0
        s = weights.sum()
        if s <= 0:
            raise StopIteration("all domains hit their epochs_cap")
        weights /= s
        idx = int(rng.choice(len(self.specs), p=weights))
        spec = self.specs[idx]
        state = self._states[idx]
        if not state.shards:
            # No files yet — mark exhausted and recurse.
            self._exhausted.add(idx)
            return self.next_shard(rank, step)
        n = len(state.shards)
        # Per-domain shuffled order, seeded by (seed, domain).
        order_rng = np.random.default_rng(
            int.from_bytes(
                hashlib.blake2b(struct.pack("<qq", self.seed, idx), digest_size=8).digest(),
                "little",
            )
        )
        order = order_rng.permutation(n)
        within = (step // max(1, 1)) % n  # plain step (world_size handled by caller seeding rank-distinct)
        shard = state.shards[int(order[within])]
        state.visits += 1
        if spec.epochs_cap is not None and state.visits >= spec.epochs_cap * n:
            self._exhausted.add(idx)
        return shard
