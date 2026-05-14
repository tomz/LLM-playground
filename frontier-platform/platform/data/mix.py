"""Domain mixing weights & sampler."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DomainSpec:
    name: str
    shard_glob: str
    weight: float        # sampling probability (will be re-normalized)
    epochs_cap: float = 4.0   # max times we are allowed to revisit this domain


class MixtureSampler:
    """Multinomial sampler over domains; per-domain shuffled shard iterator.

    Honors `epochs_cap` to avoid memorization on small high-quality domains.
    Deterministic given (global_seed, rank, step).
    """
    def __init__(self, specs: list[DomainSpec], global_seed: int):
        self.specs = specs
        self.seed = global_seed

    def next_shard(self, rank: int, step: int) -> str:
        raise NotImplementedError
