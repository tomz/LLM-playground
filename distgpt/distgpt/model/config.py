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
