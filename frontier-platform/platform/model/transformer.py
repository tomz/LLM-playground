"""Decoder-only transformer (real implementation, ported from distgpt).

Pascal-friendly: uses `F.scaled_dot_product_attention` (SDPA math backend
auto-selects on sm_60). No FlashAttention, no bf16 — fp16 or fp32 only.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.weight


def _build_rope(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv_freq)
    return f.cos().to(dtype), f.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _rope_at(rope: "RoPE", x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a single [B,H,T,Dr] tensor at the given absolute positions."""
    max_pos = int(positions.max().item()) + 1
    cos, sin = rope._ensure(max_pos, x.device, x.dtype)
    cos = cos[positions]
    sin = sin[positions]
    return _apply_rope(x, cos, sin)


class RoPE(nn.Module):
    """Rotary Position Embedding. Stateless except for a cached (cos, sin) buffer."""

    def __init__(self, head_dim: int, base: float = 10_000.0, max_seq: int = 8192):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq = max_seq
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _ensure(self, seq_len: int, device, dtype):
        if (
            self._cache is None
            or self._cache[0].device != device
            or self._cache[0].dtype != dtype
            or self._cache[0].shape[0] < seq_len
        ):
            self._cache = _build_rope(max(seq_len, self.max_seq), self.head_dim, self.base, device, dtype)
        return self._cache

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor | None = None):
        T = q.shape[-2]
        if positions is not None:
            need = int(positions.max().item()) + 1
            cos, sin = self._ensure(need, q.device, q.dtype)
            cos = cos[positions]
            sin = sin[positions]
        else:
            cos, sin = self._ensure(T, q.device, q.dtype)
            cos = cos[:T]
            sin = sin[:T]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


class GQAttention(nn.Module):
    """Grouped-Query Attention via SDPA. Supports an optional incremental
    ``kv_cache``: when provided, new K/V are appended and the query attends over
    the full cached history at absolute ``start_pos`` (used by the serving
    engine's decode path); without it, runs full causal self-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * D, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * D, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * D, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * D, cfg.d_model, bias=False)
        self.rope = RoPE(D, base=cfg.rope_base, max_seq=cfg.max_seq_len)
        self.qk_norm = bool(getattr(cfg, "qk_norm", False))
        if self.qk_norm:
            self.q_norm = RMSNorm(D, cfg.rms_eps)
            self.k_norm = RMSNorm(D, cfg.rms_eps)

    def forward(self, x: torch.Tensor, kv_cache=None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hk, D = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hk, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # RoPE at absolute positions [start_pos, start_pos+T) so cached and new
        # tokens share a consistent rotation.
        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        q, k = self.rope.apply(q, k, positions=positions)
        if kv_cache is not None:
            # Append new K,V to the cache and attend over the full history.
            k = kv_cache.append("k", k, time_dim=2)
            v = kv_cache.append("v", v, time_dim=2)
        if Hk != H:
            rep = H // Hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if kv_cache is not None and T == 1:
            # Single-token decode: query attends to all cached keys (no causal mask
            # needed — everything in the cache is in the past).
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(y)


class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3).

    The KV cache is the dominant cost of long-context serving. MLA compresses it
    by projecting the input to a small shared **latent** ``c_kv`` (dim
    ``mla_kv_latent_dim``) that is the *only* thing cached, then up-projecting to
    per-head K/V on the fly. Position information is carried by a small
    **decoupled RoPE** key (dim ``mla_rope_dim``) computed once and shared across
    heads. This gives 5-10x KV-cache compression at near-equal quality.

    Layout per head: head_dim = nope_dim (content, from the latent) + rope_dim
    (position, decoupled). Queries get their own latent down/up projection.

    Supports incremental decode: only the compressed latent ``c_kv`` and the
    single shared decoupled-RoPE key are cached (that is the whole point of MLA),
    and per-head K/V are re-expanded from the latent each step. The serving
    engine also consumes ``kv_bytes_per_token`` for the cache-size accounting.
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
        self.q_latent = cfg.mla_kv_latent_dim

        # Query: down-project to a latent, then up to per-head (nope + rope).
        self.q_down = nn.Linear(cfg.d_model, self.q_latent, bias=False)
        self.q_up = nn.Linear(self.q_latent, H * D, bias=False)
        # KV: down-project to the cached latent (this is what we'd cache) ...
        self.kv_down = nn.Linear(cfg.d_model, self.kv_latent, bias=False)
        # ... then up-project to per-head K(nope) and V.
        self.k_up = nn.Linear(self.kv_latent, H * self.nope_dim, bias=False)
        self.v_up = nn.Linear(self.kv_latent, H * D, bias=False)
        # Decoupled RoPE key: a single shared key carrying position (broadcast to heads).
        self.k_rope = nn.Linear(cfg.d_model, self.rope_dim, bias=False)
        self.o_proj = nn.Linear(H * D, cfg.d_model, bias=False)
        self.rope = RoPE(self.rope_dim, base=cfg.rope_base, max_seq=cfg.max_seq_len)

    def forward(self, x: torch.Tensor, kv_cache=None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_head, self.head_dim
        # Queries
        q = self.q_up(self.q_down(x)).view(B, T, H, D).transpose(1, 2)   # [B,H,T,D]
        q_nope, q_rope = q[..., : self.nope_dim], q[..., self.nope_dim :]
        # KV latent (the cached quantity) — for the *new* tokens
        c_kv_new = self.kv_down(x)                                        # [B,T,kv_latent]
        # Decoupled RoPE key for the new tokens, shared across heads
        k_rope_new = self.k_rope(x).view(B, T, 1, self.rope_dim).transpose(1, 2)  # [B,1,T,r]

        positions = torch.arange(start_pos, start_pos + T, device=x.device)
        # RoPE: queries rotate at their absolute positions; the new keys too.
        q_rope = _rope_at(self.rope, q_rope, positions)
        k_rope_new = _rope_at(self.rope, k_rope_new, positions)

        if kv_cache is not None:
            c_kv = kv_cache.append("c_kv", c_kv_new, time_dim=1)          # [B,T_tot,latent]
            k_rope = kv_cache.append("k_rope", k_rope_new, time_dim=2)    # [B,1,T_tot,r]
        else:
            c_kv, k_rope = c_kv_new, k_rope_new

        T_kv = c_kv.shape[1]
        # Re-expand per-head K(nope) and V from the (cached) latent.
        k_nope = self.k_up(c_kv).view(B, T_kv, H, self.nope_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(B, T_kv, H, D).transpose(1, 2)
        k_rope = k_rope.expand(B, H, T_kv, self.rope_dim).contiguous()
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)
        causal = not (kv_cache is not None and T == 1)
        y = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=causal)
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Sparse MoE FFN: top-k routing over fine-grained experts + optional shared
    expert(s), with either aux-loss-free (bias-based) or aux-loss balancing.

    This is the 2025 frontier default (DeepSeek-V3): many narrow routed experts,
    one or more always-on shared experts that capture common knowledge, and
    **aux-loss-free** load balancing — a per-expert bias is nudged up/down to
    equalize load instead of adding a quality-degrading auxiliary loss to the
    main objective.

    Stores ``self.last_aux_loss`` (the router z-loss, plus a load-balance loss
    only when ``moe_balance == 'aux_loss'``) for the trainer to add to the main
    loss, and ``self.last_expert_counts`` for monitoring. The routing bias lives
    in ``self.routing_bias`` and is updated in-place (no gradient) each training
    forward when balancing is aux-free.
    """

    def __init__(self, cfg: ModelConfig, z_loss_coeff: float = 1e-3, lb_loss_coeff: float = 1e-2):
        super().__init__()
        self.cfg = cfg
        self.n_experts = int(cfg.moe_num_experts)
        self.top_k = int(cfg.moe_top_k)
        self.n_shared = int(cfg.moe_shared_experts)
        self.balance = cfg.moe_balance
        self.bias_update_speed = float(cfg.moe_bias_update_speed)
        self.dispatch_mode = getattr(cfg, "moe_dispatch", "batched")
        if self.dispatch_mode not in ("batched", "loop"):
            raise ValueError(
                f"moe_dispatch must be 'batched' or 'loop', got {self.dispatch_mode!r}"
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
        # *selection* (not for the combine weights), nudged to equalize load.
        self.register_buffer("routing_bias", torch.zeros(self.n_experts))
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_expert_counts: torch.Tensor = torch.zeros(self.n_experts, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        N = flat.shape[0]
        logits = self.gate(flat)                                  # [N, E]
        # z-loss: penalize large logsumexp (keeps router logits well-scaled)
        lse = torch.logsumexp(logits, dim=-1)
        z_loss = self.z_loss_coeff * (lse.pow(2).mean())
        probs = logits.softmax(dim=-1)                            # [N, E]

        # --- selection: aux-free adds a per-expert bias to *choose* experts,
        # but the combine weights come from the unbiased softmax probs. ---
        if self.balance == "aux_free":
            sel_score = probs + self.routing_bias.to(probs.dtype)
        else:
            sel_score = probs
        _, top_i = sel_score.topk(self.top_k, dim=-1)             # [N, k]
        top_w = probs.gather(1, top_i)                            # unbiased weights
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-9)

        # Per-expert token counts (each token counted top_k times across slots).
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=probs.device, dtype=torch.long)
            counts.scatter_add_(0, top_i.reshape(-1),
                                torch.ones(top_i.numel(), device=probs.device, dtype=torch.long))
            self.last_expert_counts = counts

        if self.balance == "aux_loss":
            # Switch-style load-balance: f_e * P_e, encourages uniform routing.
            one_hot = torch.zeros_like(probs).scatter_(1, top_i, 1.0 / self.top_k)
            f = one_hot.mean(dim=0)        # fraction of tokens routed to e
            P = probs.mean(dim=0)          # mean gating probability per e
            lb_loss = self.lb_loss_coeff * self.n_experts * (f * P).sum()
            self.last_aux_loss = z_loss + lb_loss
        else:
            # Aux-loss-free: no balance term in the loss. Instead nudge the bias
            # toward under-loaded experts (DeepSeek-V3 §2.1.2). Training only.
            self.last_aux_loss = z_loss
            if self.training and self.bias_update_speed > 0:
                with torch.no_grad():
                    load = counts.float() / max(1, N * self.top_k)  # frac of slots
                    target = 1.0 / self.n_experts
                    # under-loaded (load<target) -> raise bias; over-loaded -> lower
                    self.routing_bias += self.bias_update_speed * torch.sign(target - load)

        out = torch.zeros_like(flat)
        # Routed experts: dispatch tokens to experts. Both backends are
        # mathematically equivalent up to floating-point accumulation order.
        if self.dispatch_mode == "batched":
            self._dispatch_batched(flat, top_i, top_w, out)
        else:
            self._dispatch_loop(flat, top_i, top_w, out)

        # Shared expert(s): always on for every token (weight 1.0 each).
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

        Kept for parity testing and small-experiment ablations. Correctness is
        easy to read here; production paths use :meth:`_dispatch_batched`."""
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

        Each (token, slot) pair becomes one assignment. We sort assignments by
        expert id so each expert sees a *contiguous* slab of tokens, run one
        GEMM per expert on its slab, then scatter the weighted outputs back to
        the source-token rows via ``index_add_``. This removes the per-expert
        Python overhead and is the structure a real expert-parallel
        implementation would dispatch over (sort then all-to-all the slabs).

        Complexity: ``O(N*k * D + sum_e (n_e * D * d_ffn))`` GEMM FLOPs (same as
        the loop) but with one launch per expert instead of one launch per
        ``mask.any() + boolean-select + matmul + scatter``.
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

        # Count assignments per expert (for the contiguous slab layout).
        counts = torch.bincount(slot_expert, minlength=self.n_experts)  # [E]
        # Sort slots by expert id. ``stable=True`` keeps tokens in original
        # order within each expert's slab, which makes the scatter deterministic.
        order = torch.argsort(slot_expert, stable=True)
        sorted_src = slot_src[order]                                # [M]
        sorted_w = slot_weight[order]                               # [M]

        # Per-expert slab boundaries.
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
            # Scale by combine weight and scatter-add back to the source rows.
            y = y * sorted_w[lo:hi, None]
            out.index_add_(0, sorted_src[lo:hi], y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = MLAttention(cfg) if cfg.attn_kind == "mla" else GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = MoEFFN(cfg) if cfg.moe_num_experts else SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(self, x: torch.Tensor, kv_cache=None, start_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, layer_idx=i) for i in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # Multi-Token Prediction auxiliary heads (train-only). Head j predicts the
        # token (j+2) ahead from the same final hidden state. Discarded at
        # inference; see config.mtp_tokens.
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
             for _ in range(int(getattr(cfg, "mtp_tokens", 0)))]
        )
        self.init_weights("default")

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None, positions=None):
        x = self.tok_emb(tokens)
        use_ckpt = getattr(self.cfg, "activation_ckpt", "none") == "selective" and self.training
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
            for blk in self.layers:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.layers:
                x = blk(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            # Add MoE aux losses if any
            for blk in self.layers:
                if isinstance(blk.ffn, MoEFFN):
                    loss = loss + blk.ffn.last_aux_loss
            # Multi-Token Prediction auxiliary loss (train-only). Head j predicts
            # the token (j+2) ahead: targets shifted left by (j+1). Averaged over
            # heads, scaled by mtp_weight. Gated on self.training so eval reports
            # pure next-token CE.
            if self.mtp_heads and self.training:
                T = targets.size(1)
                aux = logits.new_zeros(())
                used = 0
                for j, head in enumerate(self.mtp_heads):
                    shift = j + 1
                    if T - shift <= 0:
                        continue
                    pred = head(x[:, : T - shift, :])
                    tgt = targets[:, shift:]
                    aux = aux + F.cross_entropy(
                        pred.float().reshape(-1, pred.size(-1)), tgt.reshape(-1)
                    )
                    used += 1
                if used:
                    loss = loss + self.cfg.mtp_weight * (aux / used)
        return logits, loss

    @torch.no_grad()
    def forward_with_cache(self, tokens: torch.Tensor, kv_cache):
        """Incremental decode forward. ``tokens`` is ``[B, T]`` (T = prompt len on
        prefill, 1 per decode step). Uses ``kv_cache`` (a
        :class:`platform.model.kv_cache.KVCache`) to attend over history at
        absolute positions starting from ``kv_cache.pos``. Returns logits ``[B,T,V]``.

        Only the main head runs (MTP heads are train-only), so inference cost is
        identical to a non-MTP model.
        """
        start_pos = kv_cache.pos
        x = self.tok_emb(tokens)
        for blk, lc in zip(self.layers, kv_cache.layers):
            x = blk(x, kv_cache=lc, start_pos=start_pos)
        kv_cache.advance(tokens.shape[1])
        x = self.final_norm(x)
        return self.lm_head(x)

    def init_weights(self, scheme: str = "muP") -> None:
        """muP-flavored: residual output projections get smaller std."""
        n_layer = self.cfg.n_layer
        residual_std = 0.02 / math.sqrt(2 * max(1, n_layer)) if scheme == "muP" else 0.02
        base_std = 0.02

        def _init(m: nn.Module, std: float):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=base_std)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=base_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Overwrite the output projections with the residual-scaled std.
        for blk in self.layers:
            _init(blk.attn.o_proj, residual_std)
            if isinstance(blk.ffn, SwiGLU):
                _init(blk.ffn.w2, residual_std)
            elif isinstance(blk.ffn, MoEFFN):
                for e in blk.ffn.experts:
                    _init(e.w2, residual_std)
                for sh in blk.ffn.shared:
                    _init(sh.w2, residual_std)

        # Optional zero-init of the attention output (MAI-Thinking-1 §1): each
        # block starts as the identity x + 0·attn(x), so early-training attention
        # noise can't perturb MoE router assignments before routing settles. This
        # is the residual analogue of setting a post-attention norm gain to zero;
        # with pre-norm + no post-attn norm here, zeroing o_proj is exact.
        if getattr(self.cfg, "zero_init_attn_output", False):
            for blk in self.layers:
                nn.init.zeros_(blk.attn.o_proj.weight)
                if blk.attn.o_proj.bias is not None:
                    nn.init.zeros_(blk.attn.o_proj.bias)
