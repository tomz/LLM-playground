"""Request router across model tiers (nano/mid/pro/max) with SLO-aware queueing."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Tier:
    name: str
    engine_endpoint: str
    target_ttft_ms: int
    target_itl_ms: int
    cost_per_mtok: float


class Router:
    def __init__(self, tiers: list[Tier]):
        self.tiers = {t.name: t for t in tiers}
    def select(self, hint: str | None, prompt_len: int) -> Tier:
        """Pick a tier given an explicit hint or a heuristic on prompt size."""
        raise NotImplementedError
