from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 24
    n_head: int = 16
    n_kv_head: int = 8
    d_model: int = 2048
    d_ffn: int = 5632
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = True
    # --- stability knobs (default-off; adopted from nanogpt-edu / modded-nanogpt) ---
    qk_norm: bool = False        # per-head RMSNorm on Q and K before RoPE. Bounds
                                 # attention-logit scale at large d_model/long
                                 # context, letting you push LR higher without
                                 # loss spikes. Per-head + local, so it stays
                                 # tensor-parallel-friendly (no cross-shard reduce).
    zero_init_proj: bool = False  # zero-init the residual-write matrices (attn
                                 # o_proj + ffn down-proj) so each block starts as
                                 # identity → stable high-LR warmup at scale.

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    def param_count(self) -> int:
        emb = self.vocab_size * self.d_model
        attn = self.n_layer * (
            self.d_model * (self.n_head + 2 * self.n_kv_head) * self.head_dim
            + self.d_model * self.d_model
        )
        ffn = self.n_layer * 3 * self.d_model * self.d_ffn
        out = 0 if self.tie_embeddings else emb
        return emb + attn + ffn + out
