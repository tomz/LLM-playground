"""Simulated tokenizer training: ~12h on 96-core box for 100GB."""
from __future__ import annotations
from .clock import Clock
from .economy import CostBook
from .events import EventBus


def simulate_tokenizer_training(
    sample_gb: float,
    vocab_size: int,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    cores: int = 96,
    dollar_per_core_h: float = 0.04,
) -> dict:
    bus.emit("tokenizer.start", sample_gb=sample_gb, vocab=vocab_size)
    # rough: 100GB on 96 cores = 12h. Scale linearly with bytes & inverse with cores.
    hours = (sample_gb / 100.0) * (96 / max(cores, 1)) * 12.0
    clock.advance(hours * 3600.0)
    cost.charge("tokenizer", "cpu_nodes", cores * hours * dollar_per_core_h)
    bus.emit("tokenizer.done", hours=hours, dollars=cores * hours * dollar_per_core_h)
    return {"vocab_size": vocab_size, "hours": hours}
