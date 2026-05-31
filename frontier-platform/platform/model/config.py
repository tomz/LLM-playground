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
    # QK-norm: RMSNorm applied to per-head queries and keys before attention.
    # A cheap stability trick (used in many 2024-2025 frontier models) that tames
    # attention-logit blowups during large-scale training. Present in nanogpt-edu.
    qk_norm: bool = False
    # Multi-Token Prediction (DeepSeek-V3): extra auxiliary heads predict tokens
    # n+2, n+3, ... from the final hidden state, densifying the gradient for
    # better sample efficiency. Train-only (discarded at inference, so zero infer
    # cost) and doubles as a speculative-decoding draft signal. 0 = off.
    mtp_tokens: int = 0
    mtp_weight: float = 0.3
    # Attention variant: "gqa" (Grouped-Query, Llama-style) or "mla" (Multi-head
    # Latent Attention, DeepSeek-V2/V3). MLA compresses the KV cache 5-10x via a
    # low-rank latent projection at near-equal quality — the frontier answer for
    # long-context serving cost.
    attn_kind: str = "gqa"
    # MLA: dimension of the shared KV latent (the compressed cache). Typical
    # frontier value ~512 (vs n_kv_head*head_dim for GQA). Only used when
    # attn_kind == "mla".
    mla_kv_latent_dim: int = 512
    # MLA: per-head dimension carrying decoupled RoPE position info (DeepSeek-V3
    # uses 64). The rest of the head dim is the "nope" (content) part.
    mla_rope_head_dim: int = 0   # 0 => head_dim // 2
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
    # MoE dispatch backend: "batched" sorts tokens by expert id and runs each
    # expert on a contiguous slice with one GEMM (the right shape for later
    # expert-parallel all-to-all); "loop" is the original per-expert Python
    # for-loop, kept for parity tests and ablations. Default "batched".
    moe_dispatch: str = "batched"
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

    @property
    def mla_rope_dim(self) -> int:
        """Per-head RoPE (positional) dimension for MLA."""
        return self.mla_rope_head_dim if self.mla_rope_head_dim else self.head_dim // 2

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """KV-cache size per token per layer, in bytes.

        GQA caches K and V for ``n_kv_head`` heads. MLA caches only the shared
        low-rank latent (``mla_kv_latent_dim``) plus the decoupled RoPE key
        (``mla_rope_dim``) — the 5-10x compression that makes long context
        affordable. Multiply by ``n_layer`` and context length for the total.
        """
        if self.attn_kind == "mla":
            per_tok = self.mla_kv_latent_dim + self.mla_rope_dim
        else:
            per_tok = 2 * self.n_kv_head * self.head_dim   # K and V
        return per_tok * dtype_bytes

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
