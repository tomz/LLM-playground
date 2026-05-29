"""Vision encoder + projector adapter (see docs/16-multimodality.md, gap #2).

Late-fusion / LLaVA-style multimodality: a (here, randomly-initialized) ViT-like
patch encoder turns an image into a grid of patch embeddings, and a projector
maps those into the language model's hidden space so they can be prepended to the
token-embedding stream as "image tokens".

This is the *MM-1* (image understanding) entry point from the design doc. It is
toy-functional: it runs end-to-end on CPU with the tiny test Transformer, but the
encoder weights are not pretrained (a real build loads SigLIP/ViT). Audio/video
(MM-3/MM-4) are out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelConfig  # noqa: F401  (re-exported for callers)
from .transformer import RMSNorm, Transformer


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 14          # 224/14 = 16 patches per side -> 256 tokens
    in_channels: int = 3
    width: int = 384              # encoder hidden width (SigLIP-S-ish)
    depth: int = 4               # encoder transformer blocks (toy: shallow)
    n_head: int = 6
    projector: str = "mlp"        # 'linear' | 'mlp'
    pool: str = "none"            # 'none' (all patch tokens) | 'mean'

    @property
    def n_patches_per_side(self) -> int:
        assert self.image_size % self.patch_size == 0
        return self.image_size // self.patch_size

    @property
    def n_tokens(self) -> int:
        if self.pool == "mean":
            return 1
        return self.n_patches_per_side ** 2


class _EncoderBlock(nn.Module):
    """Pre-norm bidirectional self-attention + MLP (ViT block)."""

    def __init__(self, width: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.proj = nn.Linear(width, width, bias=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, N, 3, self.n_head, D // self.n_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, N, d]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # non-causal
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.proj(y)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """Patchify -> +pos -> transformer -> patch embeddings [B, n_tokens, width]."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(cfg.in_channels, cfg.width,
                               kernel_size=cfg.patch_size, stride=cfg.patch_size)
        n = cfg.n_patches_per_side ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, cfg.width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_EncoderBlock(cfg.width, cfg.n_head) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, C, H, W]
        x = self.patch(images)                       # [B, width, gh, gw]
        B, W, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)             # [B, gh*gw, width]
        x = x + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.cfg.pool == "mean":
            x = x.mean(dim=1, keepdim=True)
        return x                                     # [B, n_tokens, width]


class Projector(nn.Module):
    """Map vision width -> LM d_model so patches become image tokens."""

    def __init__(self, in_dim: int, out_dim: int, kind: str = "mlp"):
        super().__init__()
        if kind == "linear":
            self.net: nn.Module = nn.Linear(in_dim, out_dim, bias=False)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
            )
        else:
            raise ValueError(f"unknown projector kind: {kind}")
        self.norm = RMSNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class VisionLanguageModel(nn.Module):
    """LLaVA-style VLM: image tokens are prepended to the text-token embeddings.

    Forward signature mirrors the text path but injects projected image tokens at
    the front of the sequence. Loss is computed on text targets only (image token
    positions are excluded), matching standard VLM SFT.
    """

    def __init__(self, lm: Transformer, vcfg: VisionConfig | None = None):
        super().__init__()
        self.lm = lm
        self.vcfg = vcfg or VisionConfig()
        self.encoder = VisionEncoder(self.vcfg)
        self.projector = Projector(self.vcfg.width, lm.cfg.d_model, self.vcfg.projector)

    @property
    def n_image_tokens(self) -> int:
        return self.vcfg.n_tokens

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """images [B,C,H,W] -> image token embeddings [B, n_tokens, d_model]."""
        feats = self.encoder(images)
        return self.projector(feats)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ):
        """tokens [B,T] text ids; images [B,C,H,W] optional.

        Returns (logits_text, loss) where logits_text covers only the text
        positions (image tokens are dropped from the output), so it lines up
        with `targets` of shape [B, T].
        """
        import torch.nn.functional as F

        text_emb = self.lm.tok_emb(tokens)            # [B, T, D]
        if images is not None:
            img_emb = self.encode_image(images)       # [B, n_img, D]
            x = torch.cat([img_emb, text_emb], dim=1)  # [B, n_img+T, D]
            n_img = img_emb.shape[1]
        else:
            x = text_emb
            n_img = 0

        for blk in self.lm.layers:
            x = blk(x)
        x = self.lm.final_norm(x)
        logits = self.lm.lm_head(x)                    # [B, n_img+T, V]
        logits_text = logits[:, n_img:, :]             # drop image positions

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits_text.float().reshape(-1, logits_text.size(-1)),
                targets.reshape(-1),
            )
        return logits_text, loss
