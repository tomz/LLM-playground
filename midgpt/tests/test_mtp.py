"""Multi-Token Prediction auxiliary heads (DeepSeek-V3 §2.2).

Train-only: head ``j`` predicts the token at offset (j+2) from the same
final hidden state. Default-off (``cfg.mtp_tokens=0``); pinned tests cover
both modes and the export/inference no-op invariant.
"""
from __future__ import annotations
import sys, pathlib
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig


def _tiny(mtp_tokens: int = 0, **overrides) -> GPT:
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2,
               d_model=32, d_ffn=64, tie_embeddings=True,
               mtp_tokens=mtp_tokens)
    cfg.update(overrides)
    return GPT(GPTConfig(**cfg))


def test_mtp_default_off_no_heads():
    """Zero is the default; no heads, no parameter cost, no behavior change."""
    m = _tiny(mtp_tokens=0)
    assert len(m.mtp_heads) == 0
    # And the same model can be built without specifying mtp_tokens at all.
    m2 = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2,
                       d_model=32, d_ffn=64))
    assert len(m2.mtp_heads) == 0


def test_mtp_builds_heads_when_enabled():
    """With ``mtp_tokens=N`` the model holds N Linear heads shaped like lm_head."""
    m = _tiny(mtp_tokens=3)
    assert len(m.mtp_heads) == 3
    for head in m.mtp_heads:
        assert head.weight.shape == (64, 32)


def test_mtp_loss_increases_under_training():
    """When training, the per-step loss with MTP must be STRICTLY larger than
    the base loss (we're adding a positive auxiliary). Pinned because the most
    likely silent regression is to forget to actually add the aux to ``loss``."""
    torch.manual_seed(0)
    base = _tiny(mtp_tokens=0).train()
    with_mtp = _tiny(mtp_tokens=2).train()
    # Make the two models start from identical shared params (everything but
    # the mtp heads, which only exist in `with_mtp`).
    with_mtp.tok_emb.load_state_dict(base.tok_emb.state_dict())
    with_mtp.pos_emb.load_state_dict(base.pos_emb.state_dict())
    with_mtp.lm_head.weight = with_mtp.tok_emb.weight  # restore tie
    with_mtp.blocks.load_state_dict(base.blocks.state_dict())
    with_mtp.ln_f.load_state_dict(base.ln_f.state_dict())

    x = torch.randint(0, 64, (2, 16))
    _, l_base = base(x, x)
    _, l_mtp = with_mtp(x, x)
    assert l_mtp.item() > l_base.item(), (l_base.item(), l_mtp.item())


def test_mtp_zero_aux_at_eval():
    """Critical invariant: in eval mode the model must report PURE next-token
    CE, identical to the no-MTP path. Catches silent regressions where MTP
    leaks into eval and inflates val ppl."""
    torch.manual_seed(0)
    base = _tiny(mtp_tokens=0).eval()
    with_mtp = _tiny(mtp_tokens=3).eval()
    with_mtp.tok_emb.load_state_dict(base.tok_emb.state_dict())
    with_mtp.pos_emb.load_state_dict(base.pos_emb.state_dict())
    with_mtp.lm_head.weight = with_mtp.tok_emb.weight
    with_mtp.blocks.load_state_dict(base.blocks.state_dict())
    with_mtp.ln_f.load_state_dict(base.ln_f.state_dict())

    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        _, l_base = base(x, x)
        _, l_mtp = with_mtp(x, x)
    assert torch.allclose(l_base, l_mtp), (l_base.item(), l_mtp.item())


def test_mtp_short_sequence_skips_heads():
    """If T <= mtp_tokens then some/all heads have no supervision (the
    target slides off the end). They must skip cleanly instead of crashing
    or contributing a garbage loss."""
    torch.manual_seed(0)
    m = _tiny(mtp_tokens=5, block_size=4).train()
    x = torch.randint(0, 64, (2, 4))
    _, loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)


def test_mtp_inference_returns_main_logits_only():
    """No targets → no loss → MTP path is fully skipped. The returned logits
    must come from the main lm_head only (shape-checked here, content-checked
    by the eval-equality test above)."""
    m = _tiny(mtp_tokens=2).eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits, loss = m(x, return_full_logits=True)
    assert loss is None
    assert logits.shape == (1, 8, 64)


def test_mtp_weight_scales_the_aux():
    """Larger ``mtp_weight`` ⇒ larger total loss with same data. Pinned to
    catch a silent constant-zero or constant-1 regression."""
    torch.manual_seed(0)
    light = _tiny(mtp_tokens=2, mtp_weight=0.0).train()
    heavy = _tiny(mtp_tokens=2, mtp_weight=10.0).train()
    # Identical params except for the mtp_weight scalar.
    heavy.tok_emb.load_state_dict(light.tok_emb.state_dict())
    heavy.pos_emb.load_state_dict(light.pos_emb.state_dict())
    heavy.lm_head.weight = heavy.tok_emb.weight
    heavy.blocks.load_state_dict(light.blocks.state_dict())
    heavy.ln_f.load_state_dict(light.ln_f.state_dict())
    for a, b in zip(light.mtp_heads, heavy.mtp_heads):
        b.load_state_dict(a.state_dict())

    x = torch.randint(0, 64, (2, 16))
    _, l_light = light(x, x)
    _, l_heavy = heavy(x, x)
    # mtp_weight=0 ⇒ base loss; mtp_weight=10 ⇒ much larger.
    assert l_heavy.item() > l_light.item()


def test_mtp_export_to_hf_strips_heads(tmp_path):
    """The HF GPT-2 layout has no MTP heads — export must drop them silently
    (they're train-only, not part of inference) and produce a working
    GPT2LMHeadModel. Catches the silent-regression class where MTP keys leak
    into ``_rename_state_dict`` and raise KeyError on the missing mapping."""
    from export_hf import export_to_hf

    m = _tiny(mtp_tokens=2).eval()
    out = export_to_hf(m, m.cfg, tmp_path / "hf", tokenizer_name=None)
    # The export must succeed and the safetensors must contain NO mtp_heads keys.
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file
    sd = load_file(str(out / "model.safetensors"))
    assert not any(k.startswith("mtp_heads") for k in sd)
    assert "transformer.wte.weight" in sd


def test_mtp_with_llamafied_composes():
    """MTP must work with the Llama-style flags (RoPE/RMSNorm/SwiGLU). All
    four orthogonal switches compose without surprise."""
    m = _tiny(mtp_tokens=2, pos_kind="rope", norm_kind="rmsnorm",
              mlp_kind="swiglu").train()
    x = torch.randint(0, 64, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)
