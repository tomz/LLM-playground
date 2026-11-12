"""Educational short-convolution/local/global mix, not an LFM3 reproduction.

Reuses the local GPT trunk; only explicitly selected sequence mixers change.
Local attention uses a dense mask, not a memory-efficient sliding-window kernel.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT, GPTConfig, GQAttention, apply_rope


def positive_int(name: str, value: int) -> None:
    """Validate dimensions and budgets before allocating tensors."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(cfg: GPTConfig) -> None:
    """Check the dimensional assumptions needed by both ablation arms."""
    for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_kv_head", "d_model", "d_ffn"):
        positive_int(name, getattr(cfg, name))
    if cfg.d_model % cfg.n_head or (cfg.d_model // cfg.n_head) % 2:
        raise ValueError("d_model / n_head must be an even integer for RoPE")
    if cfg.n_head % cfg.n_kv_head:
        raise ValueError("n_head must be divisible by n_kv_head")
    if cfg.attention_backend != "sdpa":
        raise ValueError("hybrid ablation requires the sdpa backend")


class CausalShortConv(nn.Module):
    """Gated depthwise convolution with left-only padding and pointwise output."""

    def __init__(self, d_model: int, kernel_size: int = 3) -> None:
        super().__init__()
        positive_int("d_model", d_model)
        positive_int("kernel_size", kernel_size)
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        # Never use symmetric padding: that would expose future tokens.
        value = F.pad(value.transpose(1, 2), (self.kernel_size - 1, 0))
        value = self.conv(value).transpose(1, 2)
        return self.o_proj(F.silu(value) * torch.sigmoid(gate))


class LocalAttention(GQAttention):
    """Causal attention over the current token and at most window-1 predecessors."""

    def __init__(self, cfg: GPTConfig, window: int) -> None:
        validate_config(cfg)
        positive_int("window", window)
        super().__init__(cfg)
        self.window = window

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        heads, kv_heads, dim = self.cfg.n_head, self.cfg.n_kv_head, self.cfg.head_dim
        q = self.q_proj(x).view(batch, length, heads, dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, kv_heads, dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, kv_heads, dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if heads != kv_heads:
            k = k.repeat_interleave(heads // kv_heads, dim=1)
            v = v.repeat_interleave(heads // kv_heads, dim=1)
        positions = torch.arange(length, device=x.device)
        distance = positions[:, None] - positions[None, :]
        allowed = (distance >= 0) & (distance < self.window)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch, length, heads * dim))


class HybridGPT(GPT):
    """GPT-compatible model with an explicit per-layer mixer recipe.

    Shared components start identically to a same-seed GPT. Replaced mixers are
    freshly initialized; this is token/width/depth matched, not parameter matched.
    No modification to GPT or its default configuration is required.
    """

    def __init__(
        self, cfg: GPTConfig, mixers: tuple[str, ...] = ("conv", "local", "global"),
        *, window: int = 8, kernel_size: int = 3,
    ) -> None:
        validate_config(cfg)
        positive_int("window", window)
        positive_int("kernel_size", kernel_size)
        if len(mixers) != cfg.n_layer or any(kind not in {"conv", "local", "global"} for kind in mixers):
            raise ValueError("mixers must name one conv/local/global mixer per layer")
        super().__init__(cfg)
        self.mixers = tuple(mixers)
        for block, kind in zip(self.blocks, mixers):
            if kind == "global":
                continue
            mixer = CausalShortConv(cfg.d_model, kernel_size) if kind == "conv" else LocalAttention(cfg, window)
            mixer.apply(self._init)
            if isinstance(mixer, CausalShortConv):
                nn.init.normal_(mixer.conv.weight, std=0.02)
            if cfg.zero_init_proj:
                nn.init.zeros_(mixer.o_proj.weight)
            block.attn = mixer
