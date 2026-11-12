"""Simulated data pipeline. Produces deterministic statistics for downstream stages."""
from __future__ import annotations
from dataclasses import dataclass
from .clock import Clock
from .economy import CostBook
from .events import EventBus


@dataclass
class DataMixSpec:
    name: str
    raw_tb: float          # raw bytes scraped
    yield_pct: float       # fraction surviving filter+dedup
    tokens_per_byte: float # post-tokenize density
    weight: float


DEFAULT_MIX = [
    DataMixSpec("web_en",        50_000.0, 0.18, 0.25, 0.40),
    DataMixSpec("web_multi",     20_000.0, 0.16, 0.27, 0.10),
    DataMixSpec("code",           3_500.0, 0.55, 0.30, 0.15),
    DataMixSpec("books",            900.0, 0.85, 0.22, 0.10),
    DataMixSpec("papers",           600.0, 0.80, 0.23, 0.08),
    DataMixSpec("math",             120.0, 0.70, 0.30, 0.07),
    DataMixSpec("stackexchange",    150.0, 0.75, 0.27, 0.05),
    DataMixSpec("synthetic",        300.0, 0.95, 0.30, 0.05),
]


def simulate_data_pipeline(
    target_tokens: float,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    mix: list[DataMixSpec] = DEFAULT_MIX,
    nodes: int = 200,
    bytes_per_node_per_s: float = 50e6,    # 50 MB/s/node post-extraction
    cpu_node_dollar_per_h: float = 1.6,
) -> dict:
    """Compute how long & how much it costs to ingest+tokenize target_tokens."""
    bus.emit("data.start", target_tokens=target_tokens, nodes=nodes)
    # how many tokens does the *raw mix* yield?
    yield_tokens_per_raw_byte = sum(m.weight * m.yield_pct * m.tokens_per_byte for m in mix) / sum(m.weight for m in mix)
    raw_bytes_needed = target_tokens / yield_tokens_per_raw_byte
    # CPU throughput is the bottleneck.
    seconds_needed = raw_bytes_needed / (bytes_per_node_per_s * nodes)
    hours = seconds_needed / 3600.0
    cost.charge("data", "cpu_nodes", nodes * hours * cpu_node_dollar_per_h)
    # storage egress estimate (~$0.01/GB out, paid once)
    cost.charge("data", "storage_egress", (raw_bytes_needed / 1e9) * 0.01)
    clock.advance(seconds_needed)
    bus.emit("data.done", raw_tb=raw_bytes_needed / 1e12,
             yield_tokens=target_tokens, hours=hours,
             dollars=nodes * hours * cpu_node_dollar_per_h)
    return {
        "raw_tb": raw_bytes_needed / 1e12,
        "target_tokens": target_tokens,
        "yield_pct": yield_tokens_per_raw_byte,
        "hours": hours,
        "dollars": nodes * hours * cpu_node_dollar_per_h,
    }
