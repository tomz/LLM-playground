"""HuggingFace-export tests.

The export module's job is two things:
  1. Build a `config.json` that LlamaForCausalLM accepts and that reflects
     the distgpt ModelConfig faithfully.
  2. Rename the state_dict keys distgpt→HF so HF's loader doesn't drop
     weights silently.

Tests focus on the rename table — that's where regressions hide (one
typo turns a 1B-param model into a partially-loaded 700M one with the
extras silently dropped). The actual HF-load round-trip is covered when
`transformers` is installed; gated otherwise.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT  # noqa: E402
from distgpt.eval.export_hf import (  # noqa: E402
    _build_hf_config_dict, _rename_state_dict,
    export_to_hf, load_hf_state_into_distgpt,
)


def _tiny_cfg(**overrides):
    base = dict(vocab_size=64, n_layer=2, n_head=4, n_kv_head=2,
                d_model=32, d_ffn=64, max_seq_len=16,
                rope_base=10000.0, tie_embeddings=True)
    base.update(overrides)
    return ModelConfig(**base)


# ---------- config.json ----------


def test_hf_config_reflects_distgpt_settings():
    cfg = _tiny_cfg(n_layer=4, d_model=128, n_head=8, n_kv_head=4)
    out = _build_hf_config_dict(cfg)
    assert out["architectures"] == ["LlamaForCausalLM"]
    assert out["model_type"] == "llama"
    assert out["num_hidden_layers"] == 4
    assert out["hidden_size"] == 128
    assert out["num_attention_heads"] == 8
    assert out["num_key_value_heads"] == 4
    assert out["intermediate_size"] == cfg.d_ffn
    assert out["rope_theta"] == cfg.rope_base
    assert out["tie_word_embeddings"] is True


def test_hf_config_records_untied_embedding():
    cfg = _tiny_cfg(tie_embeddings=False)
    out = _build_hf_config_dict(cfg)
    assert out["tie_word_embeddings"] is False


# ---------- rename_state_dict ----------


def test_rename_state_dict_covers_every_distgpt_key():
    """The renamed dict must contain every weight the model holds — a
    missing entry would silently drop weights on HF load. (We test the
    raw rename here; the safetensors-tied-embedding de-dup happens later
    in `export_to_hf` and is covered separately.)"""
    cfg = _tiny_cfg(n_layer=3)
    m = GPT(cfg)
    sd = {k: v.detach() for k, v in m.state_dict().items()}
    renamed = _rename_state_dict(sd, cfg.n_layer)
    # Tied embeddings: distgpt has one tied tensor under both tok_emb and
    # lm_head; both targets must exist in the HF dict at the rename step.
    assert "model.embed_tokens.weight" in renamed
    assert "lm_head.weight" in renamed
    # Final norm
    assert "model.norm.weight" in renamed
    # Per-layer: 2 norms + 4 attn + 3 mlp = 9 keys per layer
    expected_per_layer = 9
    actual_per_layer = sum(
        1 for k in renamed if k.startswith("model.layers.0.")
    )
    assert actual_per_layer == expected_per_layer, (
        f"layer 0 has {actual_per_layer} keys, expected {expected_per_layer}"
    )
    # Total: 3 root + 9 * n_layer
    assert len(renamed) == 3 + expected_per_layer * cfg.n_layer


def test_export_drops_tied_lm_head_in_safetensors(tmp_path: Path):
    """When tie_embeddings=True, safetensors refuses to store shared
    storage twice; export must drop the lm_head duplicate. HF recreates
    the tie at load time via `tie_word_embeddings=True` in config.json.

    This pins the de-dup behaviour so a future refactor doesn't reintroduce
    the duplicate (which would silently break safetensors export)."""
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file
    cfg = _tiny_cfg(tie_embeddings=True)
    m = GPT(cfg)
    out_dir = tmp_path / "tied"
    export_to_hf(m, cfg, out_dir)
    sd = load_file(str(out_dir / "model.safetensors"))
    assert "model.embed_tokens.weight" in sd
    assert "lm_head.weight" not in sd, (
        "tied lm_head should be dropped; safetensors can't store shared storage"
    )


def test_rename_state_dict_preserves_tensor_identity():
    """The rename is a key-rename only; tensor objects must be the same
    (no copies, no dtype casts) so the export is zero-copy."""
    cfg = _tiny_cfg()
    m = GPT(cfg)
    sd = {k: v.detach() for k, v in m.state_dict().items()}
    renamed = _rename_state_dict(sd, cfg.n_layer)
    assert renamed["model.embed_tokens.weight"] is sd["tok_emb.weight"]
    assert renamed["model.layers.0.self_attn.q_proj.weight"] is \
            sd["layers.0.attn.q_proj.weight"]
    assert renamed["model.layers.0.mlp.gate_proj.weight"] is \
            sd["layers.0.ffn.w1.weight"]


def test_rename_raises_on_missing_key():
    """Drop one key from the source dict — the renamer must raise KeyError
    rather than silently produce a partial output."""
    cfg = _tiny_cfg()
    m = GPT(cfg)
    sd = {k: v.detach() for k, v in m.state_dict().items()}
    del sd["layers.0.ffn.w1.weight"]
    with pytest.raises(KeyError, match="layers.0.ffn.w1.weight"):
        _rename_state_dict(sd, cfg.n_layer)


# ---------- export_to_hf ----------


def test_export_to_hf_writes_config_and_weights(tmp_path: Path):
    """End-to-end: exporting a tiny model produces config.json +
    (model.safetensors or pytorch_model.bin) + generation_config.json."""
    cfg = _tiny_cfg()
    m = GPT(cfg)
    out_dir = tmp_path / "hf_out"
    export_to_hf(m, cfg, out_dir)
    assert (out_dir / "config.json").exists()
    assert (out_dir / "generation_config.json").exists()
    has_weights = (out_dir / "model.safetensors").exists() or \
                    (out_dir / "pytorch_model.bin").exists()
    assert has_weights, "no weights file written"
    # config.json round-trip
    with open(out_dir / "config.json") as f:
        cfg_out = json.load(f)
    assert cfg_out["hidden_size"] == cfg.d_model
    assert cfg_out["num_hidden_layers"] == cfg.n_layer


def test_export_rejects_qk_norm():
    """qk_norm=True can't be represented by stock LlamaForCausalLM, so
    export must raise loudly rather than silently strip the norms."""
    cfg = _tiny_cfg(qk_norm=True)
    m = GPT(cfg)
    with pytest.raises(ValueError, match="qk_norm"):
        export_to_hf(m, cfg, "/tmp/should_not_exist_xyz")


# ---------- round-trip ----------


def test_export_then_load_round_trip(tmp_path: Path):
    """Train a model, export to HF, then load back into a fresh distgpt
    model. The two state_dicts must be exactly equal (zero-loss round-trip)."""
    cfg = _tiny_cfg()
    m1 = GPT(cfg)
    # Make weights non-trivial so a silent rename bug shows up.
    with torch.no_grad():
        for p in m1.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    out_dir = tmp_path / "hf_rt"
    export_to_hf(m1, cfg, out_dir)

    m2 = GPT(cfg)
    load_hf_state_into_distgpt(m2, out_dir)
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"round-trip mismatch at {k}"


# ---------- HF load smoke test (gated on transformers) ----------


def test_exported_model_loads_in_transformers_when_available(tmp_path: Path):
    """If transformers is installed, the exported dir must load cleanly as
    LlamaForCausalLM (this is the real win — lm-eval-harness consumes this
    interface). Skipped without transformers."""
    pytest.importorskip("transformers")
    from transformers import LlamaForCausalLM, LlamaConfig
    cfg = _tiny_cfg()
    m = GPT(cfg).eval()
    out_dir = tmp_path / "hf_check"
    export_to_hf(m, cfg, out_dir)
    # First check config loads.
    hf_cfg = LlamaConfig.from_pretrained(str(out_dir))
    assert hf_cfg.hidden_size == cfg.d_model
    # Then check full model loads (this also validates the weight rename).
    loaded = LlamaForCausalLM.from_pretrained(str(out_dir))
    assert loaded.config.num_hidden_layers == cfg.n_layer
    # A tiny forward — sanity that loaded weights are usable.
    with torch.no_grad():
        ids = torch.zeros((1, 4), dtype=torch.int64)
        out = loaded(ids)
        assert out.logits.shape == (1, 4, cfg.vocab_size)
