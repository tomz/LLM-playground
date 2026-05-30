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
    # Attention KV-cache format. "mla" (Multi-head Latent Attention) compresses
    # the cache ~K-fold vs "gqa", letting a replica hold a larger batch / longer
    # context in the same HBM → higher effective decode throughput. The frontier
    # answer for cheap long-context serving (docs/14 §3, docs/03).
    attn_kind: str = "gqa"
    kv_compression: float = 1.0   # effective throughput multiplier from KV savings


def _kv_throughput_mult(t: "ServingTier") -> float:
    """Throughput uplift from KV-cache compression (MLA). Decode is often
    KV-bandwidth/-capacity bound, so smaller KV ≈ proportionally larger batch.
    Capped so it stays a modeling approximation, not a fantasy."""
    if t.attn_kind == "mla":
        # default to a conservative 3x if not explicitly set
        return min(6.0, max(1.0, t.kv_compression if t.kv_compression > 1.0 else 3.0))
    return 1.0


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
        kv_mult = _kv_throughput_mult(t)
        eff_throughput = t.target_throughput_tok_s * kv_mult
        replicas = max(1, int(tok_per_s_total / eff_throughput + 0.999))
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
            "attn_kind": t.attn_kind,
            "kv_throughput_mult": kv_mult,
        }
        bus.emit("serve.tier", tier=t.name, **out[t.name])
    return out
