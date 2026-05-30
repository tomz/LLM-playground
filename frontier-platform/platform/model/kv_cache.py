"""Incremental KV cache for autoregressive decoding (GQA + MLA).

Prefill computes attention over the whole prompt and stores per-layer keys/values
(GQA) or the compressed latent + decoupled RoPE key (MLA). Each decode step then
appends one position and attends over the cache — O(T) per token instead of
O(T^2) re-encode. This is what the serving engine uses; the same `Transformer`
runs both the training path (no cache) and the decode path (with cache).

MLA's whole point is that the cache stores only the low-rank latent `c_kv`
(dim `mla_kv_latent_dim`) plus a shared RoPE key (`mla_rope_dim`) — far smaller
than GQA's per-head K and V (see `ModelConfig.kv_bytes_per_token`). The K/V are
re-expanded from the latent on the fly each step.
"""
from __future__ import annotations

import torch


class LayerCache:
    """Per-layer ring of cached tensors. Stores whatever the attention module
    hands it (GQA: K,V; MLA: c_kv, k_rope) as a growing dict of tensors along the
    time axis (dim=-2 for [B,H,T,D]; dim=1 for [B,T,*])."""

    def __init__(self) -> None:
        self.data: dict[str, torch.Tensor] = {}
        self.seq_len = 0

    def append(self, key: str, value: torch.Tensor, time_dim: int) -> torch.Tensor:
        prev = self.data.get(key)
        cat = value if prev is None else torch.cat([prev, value], dim=time_dim)
        self.data[key] = cat
        return cat


class KVCache:
    """Holds one :class:`LayerCache` per transformer layer plus the running
    absolute position (number of tokens already in the cache)."""

    def __init__(self, n_layers: int) -> None:
        self.layers = [LayerCache() for _ in range(n_layers)]
        self.pos = 0  # absolute position of the next token to be written

    def advance(self, n: int) -> None:
        self.pos += n

    def reset(self) -> None:
        for lc in self.layers:
            lc.data.clear()
            lc.seq_len = 0
        self.pos = 0
