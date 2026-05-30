"""Vision encoder + projector adapter (see docs/16-multimodality.md, gap #2).

Late-fusion / LLaVA-style multimodality: a ViT-like patch encoder turns an image
into a grid of patch embeddings, and a projector maps those into the language
model's hidden space so they can be prepended to the token-embedding stream as
"image tokens".

This is the *MM-1* (image understanding) entry point from the design doc. The
architecture (patch embed + bidirectional transformer + projector + text-only
loss) is production-correct. The encoder weights are the one external asset: a
real build loads a pretrained **SigLIP/ViT** tower via
:meth:`VisionEncoder.from_pretrained` (HuggingFace ``transformers``); when that
dependency or the weights are unavailable it falls back to in-house ViT blocks
with random init so everything still runs end-to-end on CPU. The forward
interface (``images -> [B, n_tokens, width]``) is identical either way, so the
pretrained tower is a drop-in swap. Audio/video (MM-3/MM-4) are out of scope.
"""
from __future__ import annotations

import warnings
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
    depth: int = 4                # encoder transformer blocks (in-house fallback)
    n_head: int = 6
    projector: str = "mlp"        # 'linear' | 'mlp'
    pool: str = "none"            # 'none' (all patch tokens) | 'mean'
    pretrained: str | None = None  # HF id, e.g. "google/siglip-base-patch16-224"
    freeze_encoder: bool = False   # freeze the vision tower (common for VLM SFT)

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
    """Patchify -> +pos -> transformer -> patch embeddings [B, n_tokens, width].

    By default this is the dependency-free in-house ViT (random init). Pass a
    ``backbone`` (or use :meth:`from_pretrained`) to delegate to a real pretrained
    tower (SigLIP/ViT). A backbone must expose ``forward(images) -> [B, N, width]``
    patch embeddings; this wrapper handles pooling so the output contract
    (``[B, n_tokens, width]``) is identical to the in-house path.
    """

    def __init__(self, cfg: VisionConfig, backbone: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        if backbone is None:
            self.patch = nn.Conv2d(cfg.in_channels, cfg.width,
                                   kernel_size=cfg.patch_size, stride=cfg.patch_size)
            n = cfg.n_patches_per_side ** 2
            self.pos = nn.Parameter(torch.zeros(1, n, cfg.width))
            nn.init.normal_(self.pos, std=0.02)
            self.blocks = nn.ModuleList([_EncoderBlock(cfg.width, cfg.n_head) for _ in range(cfg.depth)])
            self.norm = nn.LayerNorm(cfg.width)
        if cfg.freeze_encoder:
            for p in self.parameters():
                p.requires_grad_(False)

    @property
    def is_pretrained(self) -> bool:
        return self.backbone is not None

    @classmethod
    def from_pretrained(cls, cfg: VisionConfig) -> "VisionEncoder":
        """Build an encoder backed by a pretrained SigLIP/ViT tower when possible.

        Loads ``cfg.pretrained`` (an HF model id) via ``transformers``. If
        ``transformers`` is not installed or the weights cannot be fetched (e.g.
        offline), emits a warning and falls back to the in-house random-init ViT
        so callers always get a working encoder with the same interface.
        """
        if not cfg.pretrained:
            return cls(cfg)
        try:
            backbone = _HFVisionBackbone(cfg)
        except Exception as e:  # ImportError, network/offline, unknown id, ...
            warnings.warn(
                f"could not load pretrained vision tower {cfg.pretrained!r} "
                f"({type(e).__name__}: {e}); falling back to random-init in-house ViT",
                stacklevel=2,
            )
            return cls(cfg)
        return cls(cfg, backbone=backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, C, H, W]
        if self.backbone is not None:
            x = self.backbone(images)                # [B, N, width]
        else:
            x = self.patch(images)                   # [B, width, gh, gw]
            B, W, gh, gw = x.shape
            x = x.flatten(2).transpose(1, 2)         # [B, gh*gw, width]
            x = x + self.pos[:, : x.shape[1]]
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
        if self.cfg.pool == "mean":
            x = x.mean(dim=1, keepdim=True)
        return x                                     # [B, n_tokens, width]


class _HFVisionBackbone(nn.Module):
    """Adapter around a HuggingFace vision tower (SigLIP / CLIP / ViT).

    Returns last-hidden-state patch embeddings ``[B, N, hidden]``. A projection
    is added when the tower's hidden size differs from ``cfg.width`` so the rest
    of the VLM is agnostic to which backbone is used.
    """

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        from transformers import AutoModel  # raises ImportError if absent

        self.model = AutoModel.from_pretrained(cfg.pretrained)
        # Vision towers nest the actual encoder under .vision_model for CLIP/SigLIP.
        self.vision = getattr(self.model, "vision_model", self.model)
        hidden = int(getattr(self.model.config, "hidden_size", None)
                     or getattr(getattr(self.model.config, "vision_config", object()),
                                "hidden_size", cfg.width))
        self.out_proj = nn.Identity() if hidden == cfg.width else nn.Linear(hidden, cfg.width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.vision(pixel_values=images)
        feats = out.last_hidden_state           # [B, N(+1 cls), hidden]
        return self.out_proj(feats)


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
        # Use a pretrained SigLIP/ViT tower when configured (graceful fallback to
        # the in-house ViT otherwise); plain constructor keeps random init.
        if self.vcfg.pretrained:
            self.encoder = VisionEncoder.from_pretrained(self.vcfg)
        else:
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
