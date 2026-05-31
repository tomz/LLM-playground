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
    fused_ce: bool = False       # Liger fused linear-cross-entropy: fuse the
                                 # lm_head matmul + CE into a single Triton
                                 # kernel so the [B*T, vocab] logits tensor is
                                 # never materialized. Exact (not approximate).
                                 # Measured in midgpt: ~20% peak-VRAM saving on
                                 # 350M GPT-2 (10.1 vs 12.8 GiB) but throughput
                                 # depends on HW — on Blackwell (RTX 5060 Ti)
                                 # it ran ~26% SLOWER than dense matmul + CE.
                                 # Treat as a VRAM-headroom lever (fit bigger
                                 # batch/vocab/model), not a speedup.
                                 # Requires `pip install liger-kernel` + a
                                 # Triton-compatible GPU.

    # --- Sparse MoE FFN (DeepSeek-V3 style, default-off) ---
    # When ``moe_num_experts > 0`` the per-block SwiGLU is replaced by a
    # top-k-routed MoE block with optional always-on shared experts and
    # aux-loss-free (bias-based) load balancing. Set 0 to keep the dense path.
    # See ``distgpt/model/transformer.py:MoEFFN`` for the runtime details and
    # ``tests/test_moe.py`` for the contract.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    # Fine-grained experts: each routed expert's FFN hidden dim. 0 => d_ffn
    # (Mixtral-style coarse experts). Set smaller (e.g. d_ffn // 4) for the
    # fine-grained DeepSeek-style routing with more, narrower experts.
    moe_expert_d_ffn: int = 0
    # Always-on shared expert(s): every token also passes through these (so
    # routed experts can specialize on tail knowledge). DeepSeek-V3 uses 1.
    moe_shared_experts: int = 0
    # Load balancing: "aux_free" (bias-based nudge, DeepSeek-V3 default) or
    # "aux_loss" (Switch-style load-balance loss added to the main objective).
    moe_balance: str = "aux_free"
    # Step size for the aux-free routing-bias update (per training forward).
    moe_bias_update_speed: float = 1e-3
    # Coefficient on the routed-expert aux loss term added to the main loss.
    # For "aux_free" this is just the router z-loss; for "aux_loss" it is the
    # z-loss + Switch-style f·P load-balance term. Multiplied at GPT.forward.
    moe_aux_loss_weight: float = 1.0
    # MoE dispatch backend: "batched" (sort-by-expert + one GEMM per expert,
    # the shape an expert-parallel all-to-all would dispatch over) or "loop"
    # (per-expert Python loop, kept for parity tests + tiny ablations).
    moe_dispatch: str = "batched"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head

    @property
    def expert_d_ffn(self) -> int:
        """Hidden dim of a single routed/shared MoE expert FFN."""
        return self.moe_expert_d_ffn if self.moe_expert_d_ffn else self.d_ffn

    @property
    def moe_enabled(self) -> bool:
        """True iff the FFN should be sparse-MoE rather than dense SwiGLU.

        Single source of truth so trainer / parallel / export checks stay
        consistent with ``MoEFFN``'s construction in ``Block.__init__``.
        ``moe_num_experts == 1`` is treated as "off" because a 1-expert MoE
        is just a dense FFN with extra routing overhead.
        """
        return bool(self.moe_num_experts and self.moe_num_experts > 1)

    def _attn_emb_params(self) -> tuple[int, int, int]:
        emb = self.vocab_size * self.d_model
        attn = self.n_layer * (
            self.d_model * (self.n_head + 2 * self.n_kv_head) * self.head_dim
            + self.d_model * self.d_model
        )
        out = 0 if self.tie_embeddings else emb
        return emb, attn, out

    def param_count(self) -> int:
        """Approximate *total* parameter count (dense or MoE).

        For MoE this counts every expert (routed + shared); for the per-token
        active-FLOPs cost see :meth:`active_param_count`.
        """
        emb, attn, out = self._attn_emb_params()
        if self.moe_enabled:
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
        """Per-token *active* parameter count (= param_count for dense models).

        Only ``moe_top_k`` routed experts + every shared expert fire per token,
        so a sparse MoE trains/serves at this smaller per-token cost while
        carrying the full :meth:`param_count` of knowledge.
        """
        emb, attn, out = self._attn_emb_params()
        if self.moe_enabled:
            per_expert = 3 * self.d_model * self.expert_d_ffn
            n_active_experts = self.moe_top_k + self.moe_shared_experts
            ffn = self.n_layer * (
                n_active_experts * per_expert
                + self.d_model * self.moe_num_experts          # router always runs
            )
        else:
            ffn = self.n_layer * 3 * self.d_model * self.d_ffn
        return emb + attn + ffn + out
