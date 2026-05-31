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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn)

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
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.apply(self._init)
        # Zero-init residual-write projections (attn o_proj + ffn down-proj) so
        # each block starts as the identity map — stabilises high-LR warmup at
        # scale. Done after apply(_init) so it isn't clobbered. Default-off.
        if cfg.zero_init_proj:
            for blk in self.layers:
                nn.init.zeros_(blk.attn.o_proj.weight)
                nn.init.zeros_(blk.ffn.w2.weight)

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
            return None, loss
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
