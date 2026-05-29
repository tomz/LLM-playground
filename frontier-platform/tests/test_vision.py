"""Multimodal (vision-language) adapter tests — see platform/model/vision.py."""
from __future__ import annotations

import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.model.vision import (
    Projector,
    VisionConfig,
    VisionEncoder,
    VisionLanguageModel,
)


def _tiny_lm() -> Transformer:
    cfg = ModelConfig(
        vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=512,
    )
    torch.manual_seed(0)
    return Transformer(cfg)


def _tiny_vcfg() -> VisionConfig:
    # 32x32 image, 8px patches -> 4x4 = 16 patch tokens
    return VisionConfig(image_size=32, patch_size=8, width=48, depth=2, n_head=4)


def test_vision_config_token_count():
    v = _tiny_vcfg()
    assert v.n_patches_per_side == 4
    assert v.n_tokens == 16
    vmean = VisionConfig(image_size=32, patch_size=8, pool="mean")
    assert vmean.n_tokens == 1


def test_vision_encoder_output_shape():
    v = _tiny_vcfg()
    enc = VisionEncoder(v)
    imgs = torch.randn(2, 3, 32, 32)
    out = enc(imgs)
    assert out.shape == (2, 16, 48)


def test_projector_maps_to_d_model():
    proj = Projector(in_dim=48, out_dim=64, kind="mlp")
    x = torch.randn(2, 16, 48)
    y = proj(x)
    assert y.shape == (2, 16, 64)
    lin = Projector(in_dim=48, out_dim=64, kind="linear")
    assert lin(x).shape == (2, 16, 64)


def test_vlm_forward_text_only_matches_lm_shape():
    lm = _tiny_lm()
    vlm = VisionLanguageModel(lm, _tiny_vcfg())
    tokens = torch.randint(0, 512, (2, 10))
    logits, loss = vlm(tokens)
    assert logits.shape == (2, 10, 512)
    assert loss is None


def test_vlm_forward_with_image_prepends_tokens_but_logits_align_to_text():
    lm = _tiny_lm()
    vlm = VisionLanguageModel(lm, _tiny_vcfg())
    tokens = torch.randint(0, 512, (2, 10))
    imgs = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 512, (2, 10))
    logits, loss = vlm(tokens, images=imgs, targets=targets)
    # logits cover only the 10 text positions (image tokens dropped)
    assert logits.shape == (2, 10, 512)
    assert loss is not None and torch.isfinite(loss)
    assert vlm.n_image_tokens == 16


def test_vlm_image_tokens_change_text_predictions():
    lm = _tiny_lm()
    vlm = VisionLanguageModel(lm, _tiny_vcfg())
    vlm.eval()
    tokens = torch.randint(0, 512, (1, 8))
    imgs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        text_only, _ = vlm(tokens)
        with_img, _ = vlm(tokens, images=imgs)
    # Conditioning on an image should shift the text-position logits.
    assert not torch.allclose(text_only, with_img, atol=1e-4)
