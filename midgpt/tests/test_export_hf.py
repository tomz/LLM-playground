"""HF export round-trip: midgpt → GPT2LMHeadModel → midgpt produces identical logits.

The bug class we're pinning: silently dropping or transposing weights when
mapping midgpt's flat ``nn.Linear`` layout to HF's ``Conv1D`` (which stores the
weight transposed). The numerical-equivalence assertion is the only way to
catch a one-axis transpose: shapes are symmetric so a swap would still load
cleanly and only show up as garbled outputs.
"""
from __future__ import annotations
import sys, pathlib
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig
from export_hf import export_to_hf, load_hf_state_into_midgpt


# Stock GPT-2 has biases on (almost) everything; midgpt defaults to bias=False
# but the export must support both — we test the cross product.
@pytest.fixture(params=[False, True], ids=["nobias", "bias"])
def has_bias(request) -> bool:
    return request.param


@pytest.fixture(params=[True, False], ids=["tied", "untied"])
def tied(request) -> bool:
    return request.param


def _tiny(bias: bool, tied: bool) -> tuple[GPT, GPTConfig]:
    cfg = GPTConfig(
        vocab_size=128, block_size=16, n_layer=2, n_head=2,
        d_model=32, d_ffn=64, dropout=0.0, bias=bias, tie_embeddings=tied,
    )
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    return m, cfg


def test_export_writes_expected_files(tmp_path, has_bias, tied):
    m, cfg = _tiny(has_bias, tied)
    out = export_to_hf(m, cfg, tmp_path / "hf", tokenizer_name=None)
    assert (out / "config.json").exists()
    assert (out / "generation_config.json").exists()
    assert (out / "model.safetensors").exists() or (out / "pytorch_model.bin").exists()


def test_export_then_hf_load_matches_midgpt_logits(tmp_path, has_bias, tied):
    """Export midgpt → load via transformers.GPT2LMHeadModel → forward must
    produce logits within float-noise of the source midgpt model.

    Catches: weight transposition errors, swapped QKV slices, bias mismatch,
    accidental layer-norm bias replacement, mistied lm_head.
    """
    transformers = pytest.importorskip("transformers")
    GPT2LMHeadModel = transformers.GPT2LMHeadModel

    m, cfg = _tiny(has_bias, tied)
    out = export_to_hf(m, cfg, tmp_path / "hf", tokenizer_name=None)
    hf = GPT2LMHeadModel.from_pretrained(str(out)).eval()

    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        mid_logits, _ = m(idx, return_full_logits=True)
        hf_logits = hf(idx).logits
    delta = (mid_logits - hf_logits).abs().max().item()
    # fp32 should agree to ~1e-4; the QKV split + Conv1D transpose order means
    # there's a tiny float accumulation difference. 1e-3 is comfortable headroom.
    assert delta < 1e-3, f"midgpt vs HF logit max-abs delta {delta:.3e} too large"


def test_export_then_reimport_roundtrip(tmp_path, has_bias, tied):
    """midgpt → HF dir → midgpt: the reloaded model must produce *identical*
    logits to the original (no Conv1D middle-man this time, just the rename
    table inverse). A round-trip is the strictest test of the rename table
    correctness."""
    src, cfg = _tiny(has_bias, tied)
    out = export_to_hf(src, cfg, tmp_path / "hf", tokenizer_name=None)
    dst = GPT(cfg).eval()
    load_hf_state_into_midgpt(dst, out)
    idx = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        a, _ = src(idx, return_full_logits=True)
        b, _ = dst(idx, return_full_logits=True)
    # Pure rename + double-transpose round trip — should be bit-identical.
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_export_refuses_qk_norm(tmp_path):
    """qk_norm has no GPT2LMHeadModel equivalent; export must raise loudly
    rather than silently strip the norms (which would change the forward
    pass)."""
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2,
                    d_model=32, d_ffn=64, qk_norm=True)
    m = GPT(cfg).eval()
    with pytest.raises(ValueError, match="qk_norm"):
        export_to_hf(m, cfg, tmp_path / "hf", tokenizer_name=None)


def test_tied_export_omits_lm_head_weight(tmp_path):
    """Tied embeddings must NOT write a separate lm_head.weight to disk —
    HF's tie_word_embeddings=True will recreate the tie at load. Writing both
    copies tricks safetensors into deduplicating shared storage and masks
    the tie in a subtle way."""
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    m, cfg = _tiny(bias=False, tied=True)
    out = export_to_hf(m, cfg, tmp_path / "hf", tokenizer_name=None)
    sd = load_file(str(out / "model.safetensors"))
    assert "lm_head.weight" not in sd, "tied export wrote lm_head.weight"
    assert "transformer.wte.weight" in sd


def test_untied_export_includes_lm_head_weight(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    m, cfg = _tiny(bias=False, tied=False)
    out = export_to_hf(m, cfg, tmp_path / "hf", tokenizer_name=None)
    sd = load_file(str(out / "model.safetensors"))
    assert "lm_head.weight" in sd, "untied export missing lm_head.weight"
