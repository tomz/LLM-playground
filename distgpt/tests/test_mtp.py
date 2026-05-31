"""Multi-Token Prediction tests (Tier 5.14).

MTP adds train-only auxiliary lm heads that predict tokens at offsets
+2, +3, ... from the same final hidden state. Risk surface:

  * Disabled by default: ``mtp_tokens=0`` keeps the model bit-identical
    to the pre-MTP behaviour (no extra parameters, no extra loss term).
  * Construction: ``mtp_tokens=k`` adds exactly k heads, each shaped
    ``[vocab_size, d_model]``.
  * Train-only: aux loss is added when ``self.training`` and a finite,
    positive value; ``model.eval()`` returns pure next-token CE.
  * ``mtp_weight=0`` recovers the base loss even in train mode.
  * The +(k+1) head ignores the last (k+1) sequence positions (target
    slides off the end); no off-by-one against the seq dim.
  * Muon split: MTP heads are IO-shaped and route to AdamW.
  * HF export: MTP heads are train-only and silently stripped (not
    raised), so the exported model is the main-head subset.
  * Trainer integration: 3-step smoke with MTP on.
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
from distgpt.model.transformer import GPT  # noqa: E402
from distgpt.training.muon import split_muon_params  # noqa: E402


def _cfg(**over):
    base = dict(
        vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
        d_model=32, d_ffn=64, max_seq_len=16,
    )
    base.update(over)
    return ModelConfig(**base)


# ---------- defaults & construction ----------


def test_mtp_disabled_by_default():
    """Default config keeps ``mtp_tokens=0`` and so ``mtp_heads`` is empty —
    no parameter-count drift vs the baseline."""
    cfg = ModelConfig()
    assert cfg.mtp_tokens == 0
    m = GPT(_cfg())
    assert len(m.mtp_heads) == 0


def test_mtp_enabled_builds_correct_number_and_shape_of_heads():
    """``mtp_tokens=k`` must build exactly k extra heads, each shaped like
    the main lm_head (``[vocab_size, d_model]``)."""
    cfg = _cfg(mtp_tokens=3)
    m = GPT(cfg)
    assert len(m.mtp_heads) == 3
    for h in m.mtp_heads:
        assert isinstance(h, torch.nn.Linear)
        assert h.weight.shape == (cfg.vocab_size, cfg.d_model)


# ---------- loss behaviour ----------


def test_mtp_train_mode_adds_finite_positive_aux_to_loss():
    """In train mode, MTP aux must be finite, positive, and increase the
    loss vs the same model in eval mode (which skips MTP)."""
    torch.manual_seed(0)
    cfg = _cfg(mtp_tokens=2, mtp_weight=0.5)
    m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    m.train()
    _, train_loss = m(x, x)
    m.eval()
    _, eval_loss = m(x, x)
    assert torch.isfinite(train_loss) and torch.isfinite(eval_loss)
    # Eval-mode loss == base CE; train-mode loss includes MTP aux on top.
    assert train_loss.item() > eval_loss.item() + 1e-4


def test_mtp_eval_mode_loss_is_pure_next_token_ce():
    """``model.eval()`` must turn off the MTP aux entirely so logged eval
    loss is the same metric as a non-MTP run. Pin by comparing to a fresh
    model with ``mtp_tokens=0`` and the same shared weights."""
    torch.manual_seed(0)
    cfg_off = _cfg(mtp_tokens=0)
    cfg_on = _cfg(mtp_tokens=2)
    m_off = GPT(cfg_off).eval()
    m_on = GPT(cfg_on).eval()
    # Copy the shared weights (everything except mtp_heads.*) from m_off to m_on.
    m_on.load_state_dict({**m_on.state_dict(), **m_off.state_dict()},
                          strict=False)
    x = torch.randint(0, cfg_off.vocab_size, (2, 8))
    _, l_off = m_off(x, x)
    _, l_on = m_on(x, x)
    assert torch.equal(l_off, l_on), (
        f"eval-mode MTP loss diverged from base loss: {l_off.item()} vs {l_on.item()}"
    )


def test_mtp_weight_zero_recovers_base_loss_even_in_train_mode():
    """``mtp_weight=0`` scales the MTP contribution to zero, so even in
    train mode the loss must equal a model with ``mtp_tokens=0``. Pins the
    weight knob — a regression that ignored it would silently corrupt
    ablation runs."""
    torch.manual_seed(0)
    cfg_off = _cfg(mtp_tokens=0)
    cfg_zero = _cfg(mtp_tokens=2, mtp_weight=0.0)
    m_off = GPT(cfg_off).train()
    m_zero = GPT(cfg_zero).train()
    m_zero.load_state_dict({**m_zero.state_dict(), **m_off.state_dict()},
                            strict=False)
    x = torch.randint(0, cfg_off.vocab_size, (2, 8))
    _, l_off = m_off(x, x)
    _, l_zero = m_zero(x, x)
    assert torch.allclose(l_off, l_zero), (l_off.item(), l_zero.item())


# ---------- offset / mask correctness ----------


def test_mtp_head_skips_when_seq_too_short_for_offset():
    """Each head j needs ``T > j+1`` positions to predict (target slides
    off the end). With ``mtp_tokens=2`` and ``T=1``, both heads must be
    skipped (no IndexError, no NaN, and the loss equals the base loss
    because no MTP head contributed)."""
    cfg = _cfg(mtp_tokens=2, mtp_weight=1.0)
    m = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 1))
    _, loss = m(x, x)
    assert torch.isfinite(loss)
    # Compare to the same model in eval (= base CE only); they must agree.
    m.eval()
    _, base = m(x, x)
    assert torch.allclose(loss, base), (loss.item(), base.item())


# ---------- Muon split ----------


def test_split_muon_params_routes_mtp_heads_to_adamw():
    """MTP heads are 2D but IO (vocab-side projections) — they belong in
    AdamW like ``lm_head``. Pinned because the existing ``_IO_NAME_MARKERS``
    table includes ``mtp_heads`` and we don't want a name-table edit to
    silently move them onto Muon."""
    cfg = _cfg(mtp_tokens=2)
    m = GPT(cfg)
    muon, adamw = split_muon_params(m)
    name_for = {id(p): n for n, p in m.named_parameters()}
    muon_names = {name_for[id(p)] for p in muon}
    adamw_names = {name_for[id(p)] for p in adamw}
    # Both heads → AdamW.
    assert "mtp_heads.0.weight" in adamw_names
    assert "mtp_heads.1.weight" in adamw_names
    # And NOT in Muon.
    assert not any(n.startswith("mtp_heads.") for n in muon_names)


# ---------- HF export ----------


def test_export_to_hf_strips_mtp_heads_silently(tmp_path: Path, capsys):
    """MTP heads are train-only and so silently stripped by export (unlike
    MoE / MLA which would break inference and so raise). The exported
    safetensors must not contain any ``mtp_heads.*`` keys and the export
    must print a one-line notice listing the dropped count."""
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file
    from distgpt.eval.export_hf import export_to_hf

    cfg = _cfg(mtp_tokens=2)
    m = GPT(cfg)
    out_dir = tmp_path / "hf_mtp"
    export_to_hf(m, cfg, out_dir)
    msg = capsys.readouterr().out
    assert "MTP" in msg or "mtp" in msg, (
        f"export should print a notice for dropped MTP heads; got: {msg!r}"
    )
    sd = load_file(str(out_dir / "model.safetensors"))
    assert not any(k.startswith("mtp_heads") for k in sd), (
        f"MTP heads leaked into exported safetensors: "
        f"{[k for k in sd if k.startswith('mtp_heads')]}"
    )
    # And the export must still contain all the non-MTP keys (sanity).
    assert "model.embed_tokens.weight" in sd


# ---------- trainer smoke ----------


def test_trainer_runs_with_mtp_enabled(tmp_path: Path):
    """3-step trainer smoke with ``mtp_tokens=2`` — the new aux must play
    with grad-clip + the loader + checkpoint paths."""
    import json
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 64, size=8_000, dtype=np.uint16).tofile(
        str(data_dir / "shard_0.bin")
    )
    cfg = {
        "run_id": "smoke_mtp",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 64, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
            "mtp_tokens": 2, "mtp_weight": 0.3,
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
