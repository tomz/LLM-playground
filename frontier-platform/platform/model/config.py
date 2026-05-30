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
    # MoE (sparse Mixture-of-Experts). Fine-grained + shared-expert routing with
    # aux-loss-free (bias-based) load balancing is the 2025 frontier default
    # (DeepSeek-V3). Set moe_num_experts>1 to enable.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25
    # Fine-grained experts: each routed expert's FFN hidden dim. None => d_ffn
    # (Mixtral-style coarse experts). Set smaller (e.g. d_ffn // 4) for
    # fine-grained DeepSeek-style routing with more, narrower experts.
    moe_expert_d_ffn: int | None = None
    # Always-on shared expert(s): every token also passes through these (captures
    # common knowledge so routed experts can specialize). DeepSeek-V3 uses 1.
    moe_shared_experts: int = 0
    # Load balancing: "aux_free" (bias-based, DeepSeek-V3) or "aux_loss"
    # (Switch-style load-balance loss, the 2023 default).
    moe_balance: str = "aux_free"
    # Step size for the aux-free routing-bias update (per forward, training only).
    moe_bias_update_speed: float = 1e-3
    # Activation checkpointing: "none" or "selective" (wraps each Block in
    # torch.utils.checkpoint to trade compute for memory).
    activation_ckpt: str = "none"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    @property
    def expert_d_ffn(self) -> int:
        """Hidden dim of a single routed/shared expert FFN."""
        return self.moe_expert_d_ffn if self.moe_expert_d_ffn else self.d_ffn

    def _attn_emb_params(self) -> tuple[int, int, int]:
        emb = self.vocab_size * self.d_model
        attn = self.n_layer * (
            self.d_model * (self.n_head + 2 * self.n_kv_head) * self.head_dim  # qkv
            + self.d_model * self.d_model                                       # o_proj
        )
        out = 0 if self.tie_embeddings else emb
        return emb, attn, out

    def param_count(self) -> int:
        """Approximate *total* parameter count (dense or MoE).

        For MoE this counts every expert (routed + shared); for the per-token
        cost see :meth:`active_param_count`.
        """
        emb, attn, out = self._attn_emb_params()
        if self.moe_num_experts and self.moe_num_experts > 1:
            per_expert = 3 * self.d_model * self.expert_d_ffn  # SwiGLU = 3 mats
            n_ffn_experts = self.moe_num_experts + self.moe_shared_experts
            ffn = self.n_layer * (
                n_ffn_experts * per_expert
                + self.d_model * self.moe_num_experts          # router gate
            )
        else:
            ffn = self.n_layer * 3 * self.d_model * self.d_ffn
        return emb + attn + ffn + out

    def active_param_count(self) -> int:
        """Per-token *active* parameter count (== param_count for dense models).

        Only ``top_k`` routed experts plus all shared experts fire per token, so
        a sparse MoE trains/serves at this much smaller cost while carrying the
        full :meth:`param_count` of knowledge.
        """
        emb, attn, out = self._attn_emb_params()
        if self.moe_num_experts and self.moe_num_experts > 1:
            per_expert = 3 * self.d_model * self.expert_d_ffn
            n_active_experts = self.moe_top_k + self.moe_shared_experts
            ffn = self.n_layer * (
                n_active_experts * per_expert
                + self.d_model * self.moe_num_experts          # router always runs
            )
        else:
            ffn = self.n_layer * 3 * self.d_model * self.d_ffn
        return emb + attn + ffn + out
