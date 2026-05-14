"""Reference decoder-only transformer skeleton (PyTorch-flavored pseudocode).

Production would back this with Megatron-Core or Transformer Engine modules.
We deliberately keep one file readable end-to-end.
"""
from __future__ import annotations
from .config import ModelConfig


class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-5): ...
    def __call__(self, x): raise NotImplementedError


class RoPE:
    def __init__(self, head_dim: int, base: float, max_seq: int): ...
    def apply(self, q, k, positions): raise NotImplementedError


class GQAttention:
    """Grouped-Query Attention. Backed by FlashAttention-2/3 in production."""
    def __init__(self, cfg: ModelConfig): ...
    def __call__(self, x, kv_cache=None): raise NotImplementedError


class SwiGLU:
    def __init__(self, d_model: int, d_ffn: int): ...
    def __call__(self, x): raise NotImplementedError


class MoEFFN:
    """Optional sparse FFN. Top-k routing with load-balance + z-loss."""
    def __init__(self, cfg: ModelConfig): ...
    def __call__(self, x): raise NotImplementedError


class Block:
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = MoEFFN(cfg) if cfg.moe_num_experts else SwiGLU(cfg.d_model, cfg.d_ffn)

    def __call__(self, x, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        # embedding, blocks x n_layer, final norm, lm_head

    def forward(self, tokens, positions=None):
        """Returns logits [B, T, V]."""
        raise NotImplementedError

    def init_weights(self, scheme: str = "muP"):
        """Width-scaled init so that hyperparams transfer across model sizes."""
        raise NotImplementedError
