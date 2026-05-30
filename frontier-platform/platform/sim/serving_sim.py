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
    # Speculative decoding. A draft (small model, or — for free — the model's own
    # MTP heads as in our nanogpt-edu tools/bench_mtp_spec.py, EAGLE/Medusa
    # family) proposes `spec_draft_len` tokens that one verify pass checks in
    # parallel. Decode is latency-bound (one trunk pass per accepted token), so
    # accepting on average L>1 tokens per verify pass lifts throughput ~L-fold,
    # net of the draft's own cost. Output is unchanged (exact verification), so
    # this is a pure $/Mtok win. "none" disables.
    spec_decode: str = "none"     # 'none' | 'mtp' | 'eagle' | 'draft_model'
    spec_draft_len: int = 0       # K draft tokens proposed per step
    spec_accept_rate: float = 0.0  # per-draft-token acceptance prob α in [0, 1)
    # Fraction of the verify-step compute spent producing the draft. Self-draft
    # heads (MTP/EAGLE) are nearly free (~0.1); a separate draft model costs more.
    spec_draft_overhead: float = 0.1


def _kv_throughput_mult(t: "ServingTier") -> float:
    """Throughput uplift from KV-cache compression (MLA). Decode is often
    KV-bandwidth/-capacity bound, so smaller KV ≈ proportionally larger batch.
    Capped so it stays a modeling approximation, not a fantasy."""
    if t.attn_kind == "mla":
        # default to a conservative 3x if not explicitly set
        return min(6.0, max(1.0, t.kv_compression if t.kv_compression > 1.0 else 3.0))
    return 1.0


def _spec_decode_mult(t: "ServingTier") -> float:
    """Throughput uplift from speculative decoding.

    With K draft tokens accepted independently with probability α, the expected
    number of accepted draft tokens before the first rejection is the truncated
    geometric sum Σ_{i=1..K} α^i; plus the always-correct verified token gives a
    mean of `1 + Σ α^i` tokens emitted per verify pass (this matches the
    nanogpt-edu MTP benchmark: K=2, near-perfect α → ~3.0 tokens/round, ~1.5×).

    The verify pass costs a bit more than a plain decode step because it also
    runs the draft (`spec_draft_overhead`), so the net speedup divides the
    accepted length by that overhead factor. Capped to stay conservative.
    """
    if t.spec_decode in ("none", "") or t.spec_draft_len <= 0:
        return 1.0
    alpha = min(max(t.spec_accept_rate, 0.0), 0.999)
    accepted = 1.0
    term = 1.0
    for _ in range(t.spec_draft_len):
        term *= alpha
        accepted += term
    net = accepted / (1.0 + max(0.0, t.spec_draft_overhead))
    return min(4.0, max(1.0, net))


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
        spec_mult = _spec_decode_mult(t)
        eff_throughput = t.target_throughput_tok_s * kv_mult * spec_mult
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
            "spec_decode": t.spec_decode,
            "spec_throughput_mult": spec_mult,
        }
        bus.emit("serve.tier", tier=t.name, **out[t.name])
    return out
