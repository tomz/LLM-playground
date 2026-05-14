from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 100_352
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8           # GQA
    d_model: int = 4096
    d_ffn: int = 11008           # ~ 8/3 * d_model, rounded
    max_seq_len: int = 8192
    rope_base: float = 500_000.0
    rms_eps: float = 1e-5
    tie_embeddings: bool = False
    # MoE (optional)
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    def param_count(self) -> int:
        """Approximate dense parameter count."""
        emb = self.vocab_size * self.d_model
        attn = self.n_layer * (
            self.d_model * (self.n_head + 2 * self.n_kv_head) * self.head_dim  # qkv
            + self.d_model * self.d_model                                       # o_proj
        )
        ffn = self.n_layer * 3 * self.d_model * self.d_ffn  # SwiGLU has 3 matrices
        out = 0 if self.tie_embeddings else emb
        return emb + attn + ffn + out
