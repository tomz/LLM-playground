"""Simulated serving cluster: throughput, latency, $/Mtok."""
from __future__ import annotations
from dataclasses import dataclass
from .cluster import GPU_SPECS
from .events import EventBus


@dataclass
class ServingTier:
    name: str
    n_params: float
    quant: str            # 'bf16' | 'fp8' | 'int4'
    gpu: str = "H100"
    gpus_per_replica: int = 1
    target_throughput_tok_s: float = 2000.0   # decode tok/s per replica
    ttft_ms: int = 250


def simulate_serving(
    tiers: list[ServingTier],
    qps_per_tier: dict[str, float],
    bus: EventBus,
    avg_prompt_tokens: int = 600,
    avg_completion_tokens: int = 400,
    seconds: float = 86400.0,
) -> dict:
    out = {}
    for t in tiers:
        qps = qps_per_tier.get(t.name, 0.0)
        tok_per_request = avg_prompt_tokens + avg_completion_tokens
        tok_per_s_total = qps * tok_per_request
        replicas = max(1, int(tok_per_s_total / t.target_throughput_tok_s + 0.999))
        gpus = replicas * t.gpus_per_replica
        gpu_h = gpus * (seconds / 3600)
        gpu_dollars = gpu_h * GPU_SPECS[t.gpu]["price"]
        total_tokens = qps * tok_per_request * seconds
        cost_per_mtok = (gpu_dollars / max(1.0, total_tokens / 1e6))
        out[t.name] = {
            "qps": qps, "replicas": replicas, "gpus": gpus,
            "daily_dollars": gpu_dollars,
            "cost_per_mtok": cost_per_mtok,
            "ttft_ms": t.ttft_ms,
        }
        bus.emit("serve.tier", tier=t.name, **out[t.name])
    return out
