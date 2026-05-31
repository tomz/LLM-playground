"""Multi-head Latent Attention tests (Tier 5.13).

MLA replaces GQA when ``attn_kind="mla"``. Risk surface:

  * Forward shape: MLA must be a strict in-place replacement for GQA above
    the attention boundary (same ``[B, T, d_model]`` output, finite loss).
  * KV cache compression: the whole point of MLA. ``kv_bytes_per_token``
    must drop vs an equivalent GQA config — pinned with explicit numbers.
  * Decoupled RoPE: the rope dim is < head_dim; the cached cos/sin table
    must be sized for ``mla_rope_dim`` (not ``head_dim``).
  * QK-norm composition: combining MLA + qk_norm must not NaN.
  * Muon split: ``q_down`` / ``kv_down`` / ``k_rope`` are IO-shaped low-rank
    projections → AdamW. The full-width ``_up`` projections and ``o_proj``
    are 2D hidden weights → Muon.
  * Trainer integration: 3-step smoke with MLA on.
  * Guards: MLA + TP raises NotImplementedError; HF export refuses MLA.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT, GQAttention, MLAttention  # noqa: E402
from distgpt.training.muon import split_muon_params  # noqa: E402


def _mla_cfg(**over):
    base = dict(
        vocab_size=64, n_layer=2, n_head=4, n_kv_head=2,
        d_model=32, d_ffn=64, max_seq_len=16,
        attn_kind="mla", mla_kv_latent_dim=16, mla_rope_head_dim=4,
    )
    base.update(over)
    return ModelConfig(**base)


# ---------- block-level swap ----------


def test_mla_replaces_gqa_when_attn_kind_is_mla():
    """Every block in a ``attn_kind='mla'`` model uses MLAttention; the
    default ``attn_kind='gqa'`` keeps GQAttention. Pins the construction."""
    gqa_cfg = ModelConfig(
        vocab_size=64, n_layer=2, n_head=4, n_kv_head=2,
        d_model=32, d_ffn=64, max_seq_len=16,
    )
    m_gqa = GPT(gqa_cfg)
    assert all(isinstance(blk.attn, GQAttention) for blk in m_gqa.layers)
    m_mla = GPT(_mla_cfg())
    assert all(isinstance(blk.attn, MLAttention) for blk in m_mla.layers)


def test_mla_forward_shape_matches_gqa():
    """MLA must return the same shape as GQA so downstream norm + lm_head
    are oblivious to which attention ran."""
    cfg = _mla_cfg()
    m = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = m(x, x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(loss)


# ---------- KV cache compression (the whole point) ----------


def test_mla_kv_cache_is_smaller_than_gqa():
    """MLA caches only ``mla_kv_latent_dim + mla_rope_dim`` per token per
    layer (latent + the one decoupled-RoPE key). GQA caches
    ``2 * n_kv_head * head_dim`` (full K and V). The MLA number must be
    strictly smaller for any reasonable config — this is the design goal."""
    base = dict(vocab_size=64, n_layer=2, n_head=8, n_kv_head=4,
                 d_model=64, d_ffn=128, max_seq_len=16)
    gqa = ModelConfig(**base)
    mla = ModelConfig(**base, attn_kind="mla", mla_kv_latent_dim=16,
                      mla_rope_head_dim=4)
    g = gqa.kv_bytes_per_token()
    m = mla.kv_bytes_per_token()
    assert m < g, f"MLA cache {m}B/tok not smaller than GQA {g}B/tok"
    # And the formula matches the docstring (bf16 = 2 B).
    assert g == 2 * 2 * gqa.n_kv_head * gqa.head_dim
    assert m == 2 * (mla.mla_kv_latent_dim + mla.mla_rope_dim)


# ---------- decoupled RoPE sizing ----------


def test_mla_internal_rope_table_is_sized_to_rope_dim():
    """The ``cos, sin`` table MLA uses internally must be the *rope* dim
    (not the full head dim). Wrong sizing would silently apply RoPE only to
    half the rope slice — the cache lazily fills on first forward."""
    cfg = _mla_cfg()
    m = GPT(cfg).eval()
    blk = m.layers[0]
    assert isinstance(blk.attn, MLAttention)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    m(x, x)
    cos, sin = blk.attn._rope_cache
    # Cached cos has shape [max_seq_len, rope_dim // 2] (build_rope halves
    # the second dim because RoPE rotates pairs).
    assert cos.shape == (cfg.max_seq_len, cfg.mla_rope_dim // 2)
    assert sin.shape == cos.shape


def test_mla_rejects_rope_dim_equal_to_head_dim():
    """``nope_dim = head_dim - rope_dim`` must be positive; otherwise there
    is no content channel left. Pinned so a config typo fails at construct
    time, not on a silent NaN later."""
    # head_dim = 32 / 4 = 8; setting rope_head_dim=8 leaves nope_dim=0.
    cfg = _mla_cfg(mla_rope_head_dim=8)
    with pytest.raises(AssertionError, match="mla_rope_head_dim"):
        GPT(cfg)


# ---------- composes with qk_norm ----------


def test_mla_with_qk_norm_produces_finite_loss():
    """Both 2025 frontier knobs on at once must remain numerically clean."""
    cfg = _mla_cfg(qk_norm=True)
    m = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = m(x, x)
    assert torch.isfinite(loss)
    # And the per-head norms are actually built.
    assert m.layers[0].attn.q_norm is not None
    assert m.layers[0].attn.k_norm is not None


# ---------- Muon split ----------


def test_split_muon_params_routes_mla_down_projections_to_adamw():
    """MLA's ``q_down`` / ``kv_down`` / ``k_rope`` are IO-shaped low-rank
    projections; they must land in the AdamW group, not Muon. The full-width
    ``q_up`` / ``k_up`` / ``v_up`` / ``o_proj`` are hidden 2D weights →
    Muon."""
    cfg = _mla_cfg()
    m = GPT(cfg)
    muon, adamw = split_muon_params(m)
    name_for = {id(p): n for n, p in m.named_parameters()}
    muon_names = {name_for[id(p)] for p in muon}
    adamw_names = {name_for[id(p)] for p in adamw}
    # Down projections → AdamW.
    assert any("attn.q_down.weight" in n for n in adamw_names)
    assert any("attn.kv_down.weight" in n for n in adamw_names)
    assert any("attn.k_rope.weight" in n for n in adamw_names)
    assert not any("attn.q_down.weight" in n for n in muon_names)
    assert not any("attn.kv_down.weight" in n for n in muon_names)
    assert not any("attn.k_rope.weight" in n for n in muon_names)
    # Up projections + o_proj → Muon.
    assert any("attn.q_up.weight" in n for n in muon_names)
    assert any("attn.k_up.weight" in n for n in muon_names)
    assert any("attn.v_up.weight" in n for n in muon_names)
    assert any("attn.o_proj.weight" in n for n in muon_names)


# ---------- trainer integration ----------


def test_trainer_runs_with_mla(tmp_path: Path):
    """3-step trainer smoke with MLA on — proves the new attention plays
    with grad-clip, optimizer, checkpointing."""
    import json
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 64, size=8_000, dtype=np.uint16).tofile(
        str(data_dir / "shard_0.bin")
    )
    cfg = {
        "run_id": "smoke_mla",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 64, "n_layer": 2, "n_head": 4, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
            "attn_kind": "mla", "mla_kv_latent_dim": 16,
            "mla_rope_head_dim": 4,
        },
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 3,
        },
        "train": {"micro_batch": 2, "grad_accum": 1,
                    "log_every": 1, "eval_every": 99, "ckpt_every": 99},
    }
    train(cfg)
    log = (Path(cfg["out_dir"]) / "log.jsonl").read_text().splitlines()
    losses = [json.loads(line)["loss"] for line in log if '"loss"' in line]
    assert losses and all(np.isfinite(l) for l in losses)


# ---------- guards ----------


def test_mla_plus_tp_raises_not_implemented():
    """MLA + TP is also not yet supported. Pin the raise so when MLA-TP
    sharding lands the test fails loudly."""
    class _FakeMesh:
        def size(self): return 2
    cfg = _mla_cfg()
    m = GPT(cfg)
    from distgpt.parallel.tensor import apply_tp
    with pytest.raises(NotImplementedError, match="MLA"):
        apply_tp(m, _FakeMesh())


def test_export_to_hf_rejects_mla(tmp_path: Path):
    """No stock HF class has the MLA layout; export must raise rather than
    silently produce a model with the wrong attention."""
    from distgpt.eval.export_hf import export_to_hf
    cfg = _mla_cfg()
    m = GPT(cfg)
    with pytest.raises(NotImplementedError, match="MLA"):
        export_to_hf(m, cfg, tmp_path / "should_not_exist")
