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
        cos, sin = self._ensure(T, q.device, q.dtype)
        if positions is not None:
            cos = cos[positions]
            sin = sin[positions]
        else:
            cos = cos[:T]
            sin = sin[:T]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


class GQAttention(nn.Module):
    """Grouped-Query Attention via SDPA. KV cache argument is accepted but ignored
    here (used by the serving engine via a separate code path)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * D, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * D, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * D, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * D, cfg.d_model, bias=False)
        self.rope = RoPE(D, base=cfg.rope_base, max_seq=cfg.max_seq_len)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hk, D = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hk, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)
        q, k = self.rope.apply(q, k)
        if Hk != H:
            rep = H // Hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        # Routed experts: per-expert mask + matmul. Fine for small/medium N.
        for e in range(self.n_experts):
            mask = (top_i == e)                                   # [N, k]
            if not mask.any():
                continue
            w_e = (top_w * mask).sum(dim=-1)                      # [N]
            sel = w_e > 0
            if not sel.any():
                continue
            y_e = self.experts[e](flat[sel])
            out[sel] += w_e[sel, None] * y_e

        # Shared expert(s): always on for every token (weight 1.0 each).
        for sh in self.shared:
            out += sh(flat)
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = MoEFFN(cfg) if cfg.moe_num_experts else SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), kv_cache=kv_cache)
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
        return logits, loss

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
