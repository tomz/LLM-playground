"""Request router across model tiers."""
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
        """Heuristic: explicit `hint` wins if it names a known tier; otherwise
        pick the cheapest tier whose `target_ttft_ms` budget can absorb the
        prompt (`prompt_len * 0.05`ms per token, a stand-in for prefill time).
        Ties broken by lowest cost."""
        if hint and hint in self.tiers:
            return self.tiers[hint]
        budget_ms = prompt_len * 0.05
        candidates = [t for t in self.tiers.values() if t.target_ttft_ms >= budget_ms]
        if not candidates:
            # Nothing fits; fall back to the fastest tier (lowest ttft) — best effort.
            return min(self.tiers.values(), key=lambda t: t.target_ttft_ms)
        return min(candidates, key=lambda t: (t.cost_per_mtok, t.target_ttft_ms))
