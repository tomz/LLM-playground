"""Tensor-parallel-friendly transformer.

The linear layers here are plain `nn.Linear`; tensor-parallel sharding is
applied later via `torch.distributed.tensor.parallelize_module` so this same
file works in single-GPU, FSDP, and TP+FSDP regimes.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


def build_rope(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv_freq)
    return f.cos().to(dtype), f.sin().to(dtype)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * D, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * D, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * D, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * D, cfg.d_model, bias=False)
        # QK-Norm: per-head RMSNorm over head_dim, applied after projection but
        # before RoPE. Local per-head op → safe under tensor parallelism.
        self.q_norm = RMSNorm(D, cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(D, cfg.rms_eps) if cfg.qk_norm else None

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        D = self.cfg.head_dim
        # Under tensor parallelism the q/k/v projections are column-sharded,
        # so each rank produces a slice of the heads. Reshape by the *local*
        # head count (output_features // head_dim), not the global one;
        # otherwise the view() call shape-asserts on TP > 1. The same logic
        # naturally yields the global head count when TP == 1.
        q_proj = self.q_proj(x)
        k_proj = self.k_proj(x)
        v_proj = self.v_proj(x)
        H = q_proj.shape[-1] // D
        Hk_local = k_proj.shape[-1] // D
        q = q_proj.view(B, T, H, D).transpose(1, 2)
        k = k_proj.view(B, T, Hk_local, D).transpose(1, 2)
        v = v_proj.view(B, T, Hk_local, D).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if Hk_local != H:
            rep = H // Hk_local
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(y)


class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3) — KV-cache compression.

    The KV cache is the dominant cost of long-context serving. MLA compresses
    it by projecting the input to a small shared **latent** ``c_kv`` (of dim
    ``mla_kv_latent_dim``, the only thing the cache would store), then
    up-projecting to per-head K/V on the fly. Position information is carried
    by a small **decoupled-RoPE** key (dim ``mla_rope_dim``) computed once
    per token and shared across heads. Result: ~5-10× KV-cache compression at
    near-equal quality, the frontier answer for long-context inference cost.

    Per-head layout: ``head_dim = nope_dim (content, from the latent) +
    rope_dim (position, decoupled)``. Queries get their own latent
    down/up projection so the query-side latent is independent of the KV one
    (frontier-platform's convention; we expose ``mla_q_latent_dim`` for
    asymmetric setups but default to the same dim as KV for simplicity).

    distgpt's training path computes on positions ``[0, T)`` every step (no
    kv-cache decode loop — that's a serving concern), so the MLA forward is
    a strict in-place replacement for ``GQAttention.forward(x, cos, sin)``
    even though the cos/sin passed in are sized for ``head_dim`` (we re-derive
    cos/sin at ``mla_rope_dim`` from those positions; see ``_rope_for``).
    Ported from ``frontier-platform/platform/model/transformer.py:MLAttention``;
    we drop the kv_cache / start_pos plumbing because training doesn't need it.

    Naming convention used here is followed by the Muon split rules in
    ``training/muon.py``: the ``_down`` projections (``q_down``, ``kv_down``,
    ``k_rope``) are IO-shaped (project from ``d_model`` to a tiny rank), so
    they are excluded from Muon and run through AdamW. The ``_up``
    projections and ``o_proj`` are full-width 2D hidden weights → Muon.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.n_head, cfg.head_dim
        self.n_head = H
        self.head_dim = D
        self.rope_dim = cfg.mla_rope_dim
        self.nope_dim = D - self.rope_dim
        assert self.nope_dim > 0, "mla_rope_head_dim must be < head_dim"
        self.kv_latent = cfg.mla_kv_latent_dim
        self.q_latent = cfg.mla_q_lat

        # Query: down-project to a small latent, then up to per-head (nope + rope).
        self.q_down = nn.Linear(cfg.d_model, self.q_latent, bias=False)
        self.q_up = nn.Linear(self.q_latent, H * D, bias=False)
        # KV: down-project to the cached latent (the compressed quantity) ...
        self.kv_down = nn.Linear(cfg.d_model, self.kv_latent, bias=False)
        # ... then up-project to per-head K(nope) and V (full head_dim).
        self.k_up = nn.Linear(self.kv_latent, H * self.nope_dim, bias=False)
        self.v_up = nn.Linear(self.kv_latent, H * D, bias=False)
        # Decoupled-RoPE key: one shared key per token (broadcast across heads).
        self.k_rope = nn.Linear(cfg.d_model, self.rope_dim, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.d_model, bias=False)
        # QK-Norm: per-head RMSNorm before attention (operates on full head_dim
        # including the rope slice — matches frontier-platform's GQAttention
        # behaviour). Default-off; on with cfg.qk_norm=True.
        self.q_norm = RMSNorm(D, cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(D, cfg.rms_eps) if cfg.qk_norm else None
        # Lazy cos/sin cache sized for the *rope* dim (smaller than head_dim).
        # Built on first forward so we get the right device/dtype.
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _rope_for_rope_dim(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[-2]
        if (self._rope_cache is None
                or self._rope_cache[0].device != x.device
                or self._rope_cache[0].dtype != x.dtype
                or self._rope_cache[0].shape[0] < T):
            self._rope_cache = build_rope(
                max(T, self.cfg.max_seq_len), self.rope_dim,
                self.cfg.rope_base, x.device, x.dtype,
            )
        return self._rope_cache

    def forward(self, x, cos, sin):
        # ``cos, sin`` are sized for the global head_dim; MLA's decoupled RoPE
        # operates on ``rope_dim`` only, so we ignore the passed-in tables and
        # build our own (or hit the cache).
        B, T, _ = x.shape
        H, D = self.n_head, self.head_dim
        # Queries: down → up → split into nope/rope halves per head.
        q = self.q_up(self.q_down(x)).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        q_nope, q_rope = q[..., : self.nope_dim], q[..., self.nope_dim :]
        # KV latent (the cached quantity in real serving).
        c_kv = self.kv_down(x)                                           # [B,T,kv_latent]
        # Decoupled-RoPE key, shared across heads.
        k_rope = self.k_rope(x).view(B, T, 1, self.rope_dim).transpose(1, 2)  # [B,1,T,r]

        # Re-expand per-head K(nope) and V from the latent (would be done from
        # the cache during decode; here we do it from the just-computed latent).
        k_nope = self.k_up(c_kv).view(B, T, H, self.nope_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T, H, D).transpose(1, 2)

        # RoPE on the rope slice of Q + on the (head-broadcast) rope key.
        cos_r, sin_r = self._rope_for_rope_dim(x)
        q_rope = apply_rope(q_rope, cos_r, sin_r)
        k_rope = apply_rope(k_rope, cos_r, sin_r)
        # Broadcast the single rope key to every head so we can concat with k_nope.
        k_rope = k_rope.expand(B, H, T, self.rope_dim).contiguous()

        # Reassemble per-head Q and K (content + position).
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)
        if self.q_norm is not None:
            # Normalise after RoPE-reassembly so the rope-side rotation is
            # preserved into attention (matches the frontier-platform call site,
            # which also runs the per-head norm after the q_rope/k_rope split
            # has been concatenated back).
            q_full = self.q_norm(q_full)
            k_full = self.k_norm(k_full)
        y = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Sparse MoE FFN — top-k routing over fine-grained experts + optional
    shared experts, with aux-loss-free (bias-based) or aux-loss balancing.

    This is the 2025 frontier default (DeepSeek-V3): many narrow routed
    experts, one or more always-on shared experts that capture common
    knowledge, and **aux-loss-free** load balancing — a per-expert routing
    bias is nudged up/down each training step to equalize load instead of
    adding a quality-degrading auxiliary loss to the main objective.

    Stores ``self.last_aux_loss`` (the router z-loss, plus a Switch-style
    f·P load-balance loss only when ``moe_balance == "aux_loss"``) for
    :class:`GPT` to add to the main loss; and ``self.last_expert_counts``
    for monitoring. The routing bias lives in ``self.routing_bias`` and
    is updated in-place (no gradient) on each training forward when
    balancing is aux-free.

    Ported from ``frontier-platform/platform/model/transformer.py``. Two
    dispatch backends are kept here:

      * ``"batched"`` — sort assignments by expert id, one GEMM per expert
        slab, index-add back. This is the shape an expert-parallel
        all-to-all would dispatch over and is the production default.
      * ``"loop"`` — the original per-expert Python loop. Kept for parity
        tests and tiny ablations; correctness is easy to read here.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        z_loss_coeff: float = 1e-3,
        lb_loss_coeff: float = 1e-2,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_experts = int(cfg.moe_num_experts)
        self.top_k = int(cfg.moe_top_k)
        self.n_shared = int(cfg.moe_shared_experts)
        self.balance = cfg.moe_balance
        self.bias_update_speed = float(cfg.moe_bias_update_speed)
        self.dispatch_mode = cfg.moe_dispatch
        if self.dispatch_mode not in ("batched", "loop"):
            raise ValueError(
                f"moe_dispatch must be 'batched' or 'loop', got {self.dispatch_mode!r}"
            )
        if self.balance not in ("aux_free", "aux_loss"):
            raise ValueError(
                f"moe_balance must be 'aux_free' or 'aux_loss', got {self.balance!r}"
            )
        assert self.n_experts > 0 and 1 <= self.top_k <= self.n_experts
        d_exp = cfg.expert_d_ffn
        self.gate = nn.Linear(cfg.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, d_exp) for _ in range(self.n_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.d_model, d_exp) for _ in range(self.n_shared)]
        )
        self.z_loss_coeff = z_loss_coeff
        self.lb_loss_coeff = lb_loss_coeff
        # Aux-loss-free routing bias: added to gate logits only for top-k
        # *selection* (not for the combine weights). Stored as a buffer so it
        # checkpoints with the model state but doesn't receive a gradient. The
        # ``_IO_NAME_MARKERS`` table in training/muon.py already excludes any
        # param whose name contains "routing_bias", so this is safe under the
        # Muon split even though it's a parameter-shaped object.
        self.register_buffer("routing_bias", torch.zeros(self.n_experts))
        # Stashed for GPT.forward to consume. Re-set every forward; the
        # initial 0.0 lets pre-forward inspection not crash.
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_expert_counts: torch.Tensor = torch.zeros(
            self.n_experts, dtype=torch.long
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        N = flat.shape[0]
        logits = self.gate(flat)                                  # [N, E]
        # z-loss: penalize large logsumexp (keeps router logits well-scaled).
        # This is cheap and stabilises the router regardless of balance mode.
        lse = torch.logsumexp(logits, dim=-1)
        z_loss = self.z_loss_coeff * (lse.pow(2).mean())
        probs = logits.softmax(dim=-1)                            # [N, E]

        # --- Selection: aux-free adds a per-expert bias to *choose* experts;
        # combine weights still come from the unbiased softmax probs so the
        # bias never leaks gradient into the routed compute. ---
        if self.balance == "aux_free":
            sel_score = probs + self.routing_bias.to(probs.dtype)
        else:
            sel_score = probs
        _, top_i = sel_score.topk(self.top_k, dim=-1)             # [N, k]
        top_w = probs.gather(1, top_i)                            # unbiased weights
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        # Per-expert assignment counts (each token contributes top_k slots).
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=probs.device, dtype=torch.long)
            counts.scatter_add_(
                0, top_i.reshape(-1),
                torch.ones(top_i.numel(), device=probs.device, dtype=torch.long),
            )
            self.last_expert_counts = counts

        if self.balance == "aux_loss":
            # Switch-style load-balance: f_e * P_e, encourages uniform routing.
            one_hot = torch.zeros_like(probs).scatter_(1, top_i, 1.0 / self.top_k)
            f = one_hot.mean(dim=0)        # fraction of tokens routed to e
            P = probs.mean(dim=0)          # mean gating probability per e
            lb_loss = self.lb_loss_coeff * self.n_experts * (f * P).sum()
            self.last_aux_loss = z_loss + lb_loss
        else:
            # Aux-loss-free: no balance term in the loss. Instead nudge the
            # routing bias toward under-loaded experts (DeepSeek-V3 §2.1.2).
            # Training only — eval forward keeps the bias frozen.
            self.last_aux_loss = z_loss
            if self.training and self.bias_update_speed > 0:
                with torch.no_grad():
                    load = counts.float() / max(1, N * self.top_k)  # frac of slots
                    target = 1.0 / self.n_experts
                    # under-loaded (load<target) -> raise bias; over-loaded -> lower
                    self.routing_bias += self.bias_update_speed * torch.sign(target - load)

        out = torch.zeros_like(flat)
        if self.dispatch_mode == "batched":
            self._dispatch_batched(flat, top_i, top_w, out)
        else:
            self._dispatch_loop(flat, top_i, top_w, out)

        # Shared expert(s): always on for every token (combine weight 1.0).
        for sh in self.shared:
            out = out + sh(flat)
        return out.view(B, T, D)

    # ------------------------------------------------------------------
    # Dispatch backends
    # ------------------------------------------------------------------

    def _dispatch_loop(
        self,
        flat: torch.Tensor,        # [N, D]
        top_i: torch.Tensor,       # [N, k] expert indices
        top_w: torch.Tensor,       # [N, k] combine weights (sum 1 per token)
        out: torch.Tensor,         # [N, D] accumulator (mutated in-place)
    ) -> None:
        """Reference dispatch: one Python-level pass per expert.

        Kept for parity testing and small-experiment ablations. Production
        paths use :meth:`_dispatch_batched`."""
        for e in range(self.n_experts):
            mask = (top_i == e)                                   # [N, k]
            if not mask.any():
                continue
            w_e = (top_w * mask).sum(dim=-1)                      # [N]
            sel = w_e > 0
            if not sel.any():
                continue
            y_e = self.experts[e](flat[sel])
            out[sel] = out[sel] + w_e[sel, None] * y_e

    def _dispatch_batched(
        self,
        flat: torch.Tensor,        # [N, D]
        top_i: torch.Tensor,       # [N, k]
        top_w: torch.Tensor,       # [N, k]
        out: torch.Tensor,         # [N, D]
    ) -> None:
        """Sort-by-expert dispatch (the shape EP all-to-all uses).

        Each (token, slot) becomes one assignment. We sort assignments by
        expert id so each expert sees a *contiguous* slab of tokens, run one
        GEMM per expert on its slab, then scatter the weighted outputs back
        to the source-token rows via ``index_add_``.
        """
        N, k = top_i.shape
        device = flat.device
        # Flatten the (N, k) routing table to (N*k,) "slots".
        slot_expert = top_i.reshape(-1)                            # [N*k]
        slot_weight = top_w.reshape(-1)                            # [N*k]
        slot_src = torch.arange(N, device=device).repeat_interleave(k)  # [N*k]

        # Drop zero-weight slots (rare unless top_w was zeroed).
        keep = slot_weight > 0
        if not bool(keep.any()):
            return
        slot_expert = slot_expert[keep]
        slot_weight = slot_weight[keep]
        slot_src = slot_src[keep]

        # Per-expert assignment counts and slab boundaries.
        counts = torch.bincount(slot_expert, minlength=self.n_experts)  # [E]
        # ``stable=True`` keeps tokens in original order within each expert's
        # slab so the index_add scatter is deterministic.
        order = torch.argsort(slot_expert, stable=True)
        sorted_src = slot_src[order]                                # [M]
        sorted_w = slot_weight[order]                               # [M]
        offsets = torch.zeros(self.n_experts + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(counts, dim=0)

        # Gather the routed inputs once into the sorted layout, then run one
        # GEMM per expert on its contiguous slab.
        gathered = flat.index_select(0, sorted_src)                # [M, D]
        for e in range(self.n_experts):
            lo, hi = int(offsets[e].item()), int(offsets[e + 1].item())
            if hi == lo:
                continue
            y = self.experts[e](gathered[lo:hi])
            y = y * sorted_w[lo:hi, None]                          # combine weight
            out.index_add_(0, sorted_src[lo:hi], y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        # MLA replaces GQA when attn_kind="mla". The forward signature is
        # identical (x, cos, sin) → Tensor, so activation-checkpointing and
        # the downstream residual + norm code don't care which attention ran.
        self.attn = MLAttention(cfg) if cfg.mla_enabled else GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        # Sparse-MoE FFN when configured, else dense SwiGLU. The signature of
        # ``ffn(x) -> Tensor`` is unchanged either way; MoE stashes its aux
        # loss on ``self.ffn.last_aux_loss`` for ``GPT.forward`` to pick up.
        self.ffn = MoEFFN(cfg) if cfg.moe_enabled else SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, activation_ckpt: str = "none"):
        super().__init__()
        self.cfg = cfg
        self.activation_ckpt = activation_ckpt
        # Liger fused linear-cross-entropy: fuses the lm_head matmul + CE into
        # one Triton kernel so the [B*T, vocab] logits tensor is never
        # materialized — the single largest activation in the forward pass.
        # Exact (matches dense CE to ~1.8e-3, grads to ~8e-3 rel in bf16).
        # Lazily resolved so liger-kernel stays an OPTIONAL install; raises
        # only when fused_ce=True is actually requested.
        self._liger_ce = None
        if cfg.fused_ce:
            try:
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            except ImportError as e:
                raise ImportError(
                    "model.fused_ce=True requires `pip install liger-kernel` "
                    "and a Triton-compatible GPU. Set fused_ce=False for the "
                    "dense lm_head + cross_entropy path."
                ) from e
            self._liger_ce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # Multi-Token Prediction auxiliary heads (DeepSeek-V3 §2.2). Head j
        # predicts the token at offset (j+2) from the same final hidden state
        # — the main lm_head already covers +1. Train-only: ``forward`` adds
        # the averaged-per-head MTP CE × ``mtp_weight`` to the main loss when
        # ``self.training``; inference path doesn't fire these heads (and
        # ``export_to_hf`` strips them silently). 0 disables.
        self.mtp_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            for _ in range(int(cfg.mtp_tokens))
        ])
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.apply(self._init)
        # Zero-init residual-write projections (attn o_proj + ffn down-proj) so
        # each block starts as the identity map — stabilises high-LR warmup at
        # scale. Done after apply(_init) so it isn't clobbered. Default-off.
        # For MoE blocks, we zero every routed+shared expert's down-proj so the
        # FFN's contribution to the residual is zero regardless of which experts
        # the router picks at step 0.
        if cfg.zero_init_proj:
            for blk in self.layers:
                nn.init.zeros_(blk.attn.o_proj.weight)
                if isinstance(blk.ffn, SwiGLU):
                    nn.init.zeros_(blk.ffn.w2.weight)
                elif isinstance(blk.ffn, MoEFFN):
                    for e in blk.ffn.experts:
                        nn.init.zeros_(e.w2.weight)
                    for sh in blk.ffn.shared:
                        nn.init.zeros_(sh.w2.weight)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_for(self, x):
        T = x.shape[1]
        if self._rope is None or self._rope[0].device != x.device or self._rope[0].shape[0] < T:
            self._rope = build_rope(self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_base, x.device, x.dtype)
        return self._rope

    def forward(self, idx, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        cos, sin = self._rope_for(x)
        for i, blk in enumerate(self.layers):
            do_ckpt = (
                self.activation_ckpt == "full"
                or (self.activation_ckpt == "selective" and i % 2 == 0)
            )
            if do_ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.final_norm(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        if self._liger_ce is not None:
            # Fused linear-CE path: hand the kernel hidden states + lm_head
            # weight, get back loss directly. The [B*T, vocab] logits tensor
            # is never built, which is where the VRAM saving comes from.
            # Returns (None, loss) — callers that need logits must set
            # model.fused_ce=False.
            loss = self._liger_ce(
                self.lm_head.weight,
                x.reshape(-1, x.size(-1)),
                targets.reshape(-1),
            )
            logits_out = None
        else:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
            logits_out = logits
        # MoE aux loss: each MoEFFN block stashes ``last_aux_loss`` on its
        # ``.ffn`` module during the forward we just ran. Sum those, weight by
        # ``moe_aux_loss_weight``, add to the main loss. The check is cheap
        # (an isinstance per layer) and free when MoE is off (no blocks have
        # MoEFFN ffns). Aux-free balance still contributes the router z-loss
        # term so the router stays well-scaled even without an LB term.
        if self.cfg.moe_enabled:
            aux = x.new_zeros(())
            n = 0
            for blk in self.layers:
                if isinstance(blk.ffn, MoEFFN):
                    # MoE may have skipped activation-checkpoint replay if
                    # last_aux_loss was set on the no-grad recomputation; the
                    # value is still scalar and finite. We just .to() it to
                    # the loss device/dtype to be safe across mixed-precision.
                    aux = aux + blk.ffn.last_aux_loss.to(aux)
                    n += 1
            if n > 0:
                loss = loss + self.cfg.moe_aux_loss_weight * aux
        # Multi-Token Prediction auxiliary heads (DeepSeek-V3). Train-only:
        # head j predicts the token at offset (j+2), i.e. targets shifted left
        # by (j+1). Sequence positions in [T-(j+1), T) can't compute their
        # +(j+1) loss because the target slides off the end, so we just slice
        # those positions away (no -100 ignore-index mask needed — we never
        # build the dropped predictions in the first place). Averaged over
        # heads, scaled by ``mtp_weight``. Gated on ``self.training`` so eval
        # reports pure next-token CE (and so the heads contribute zero cost at
        # inference). Skipped when targets is None (caller wants generation
        # logits only) and when the sequence is too short for any head.
        if self.mtp_heads and self.training:
            T = targets.size(1)
            aux = x.new_zeros(())
            used = 0
            for j, head in enumerate(self.mtp_heads):
                shift = j + 1
                if T - shift <= 0:
                    # T too short for this offset; skip this head this step.
                    continue
                pred = head(x[:, : T - shift, :])
                tgt = targets[:, shift:]
                aux = aux + F.cross_entropy(
                    pred.float().reshape(-1, pred.size(-1)),
                    tgt.reshape(-1),
                )
                used += 1
            if used:
                loss = loss + self.cfg.mtp_weight * (aux / used)
        return logits_out, loss
