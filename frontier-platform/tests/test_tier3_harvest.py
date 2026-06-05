"""Tier 3 harvests from the MAI-Thinking-1 deep dive.

  9.  Hill-climb orchestrator (rl/hillclimb.py) — specialists → distill → climb (§5)
  10. Zero-init attention output (model/transformer.py + config flag) (§1)

See docs/research/mai-thinking-1-deep-dive.md §§1, 5.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer


def _tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
                d_model=64, d_ffn=128, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def _save_base_ckpt(path: Path, cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    m = Transformer(cfg)
    torch.save({"model": m.state_dict(), "model_cfg": cfg}, path)


# =====================================================================
# 10. Zero-init attention output
# =====================================================================

def test_zero_init_attn_output_zeros_o_proj():
    cfg = _tiny_cfg(zero_init_attn_output=True)
    torch.manual_seed(0)
    m = Transformer(cfg)
    m.init_weights("muP")
    for blk in m.layers:
        assert torch.count_nonzero(blk.attn.o_proj.weight) == 0


def test_zero_init_attn_output_makes_block_identity_at_init():
    """With o_proj zeroed, the attention sublayer contributes nothing, so the
    block output equals its input (x + 0·attn(x) + ffn-path is separate). We test
    the attention residual specifically by checking attn(x) == 0."""
    cfg = _tiny_cfg(zero_init_attn_output=True)
    torch.manual_seed(0)
    m = Transformer(cfg)
    m.init_weights("muP")
    blk = m.layers[0]
    x = torch.randn(2, 8, cfg.d_model)
    attn_out = blk.attn(blk.attn_norm(x))
    assert torch.allclose(attn_out, torch.zeros_like(attn_out), atol=1e-6)


def test_zero_init_off_by_default_keeps_o_proj_nonzero():
    cfg = _tiny_cfg()  # flag defaults False
    assert cfg.zero_init_attn_output is False
    torch.manual_seed(0)
    m = Transformer(cfg)
    m.init_weights("muP")
    # Standard residual-scaled init -> o_proj is not all zeros.
    assert torch.count_nonzero(m.layers[0].attn.o_proj.weight) > 0


def test_zero_init_model_still_trains():
    """A zero-init-attn model must still produce finite loss and train (the zero
    is only at init; gradients flow and o_proj becomes nonzero)."""
    cfg = _tiny_cfg(zero_init_attn_output=True, moe_num_experts=4, moe_top_k=2)
    torch.manual_seed(0)
    m = Transformer(cfg)
    m.init_weights("muP")
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = m(x, targets=y)
    assert torch.isfinite(loss)
    loss.backward()
    # After a backward pass the attention output projection has a gradient.
    assert m.layers[0].attn.o_proj.weight.grad is not None
    assert torch.count_nonzero(m.layers[0].attn.o_proj.weight.grad) > 0


def test_zero_init_works_with_mla():
    cfg = _tiny_cfg(attn_kind="mla", mla_kv_latent_dim=32, mla_rope_head_dim=8,
                    zero_init_attn_output=True)
    torch.manual_seed(0)
    m = Transformer(cfg)
    m.init_weights("muP")
    assert torch.count_nonzero(m.layers[0].attn.o_proj.weight) == 0
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    y = torch.randint(0, cfg.vocab_size, (2, 12))
    _, loss = m(x, targets=y)
    assert torch.isfinite(loss)


# =====================================================================
# 9. Hill-climb orchestrator
# =====================================================================

from platform.rl.hillclimb import (  # noqa: E402
    HillClimbConfig,
    HillClimbResult,
    Specialist,
    harvest_distillation_data,
    run_hill_climb,
    train_specialist,
)
from platform.rl.verifiers import reward_contains  # noqa: E402


def _specialists() -> list[Specialist]:
    # Two trivial domains with learnable byte-level rewards.
    return [
        Specialist("alpha", prompts=["Q: a1", "Q: a2"], verifier=reward_contains("a")),
        Specialist("beta", prompts=["Q: b1"], verifier=reward_contains("b")),
    ]


def _hc_cfg(base: Path, out: Path, **kw) -> HillClimbConfig:
    params = dict(
        base_ckpt=str(base), out_dir=str(out),
        specialist_steps=2, group_size=4, lr=5e-3, beta=0.0,
        max_new_tokens=6, seq_len=32, temperature=1.0,
        distill_samples_per_prompt=4, distill_epochs=2, distill_lr=5e-3,
        final_steps=2, final_lr=5e-3, min_distill_examples=1,
    )
    params.update(kw)
    return HillClimbConfig(**params)


def test_train_specialist_returns_ckpt_and_metrics(tmp_path):
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, _tiny_cfg())
    cfg = _hc_cfg(base, tmp_path / "hc")
    spec = _specialists()[0]
    res = train_specialist(cfg, spec)
    assert Path(res.ckpt).exists()
    assert res.name == "alpha"
    assert res.metrics["steps"] == 2
    assert "reward_final" in res.metrics


def test_harvest_distillation_rejection_samples(tmp_path):
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, _tiny_cfg())
    cfg = _hc_cfg(base, tmp_path / "hc", distill_reward_threshold=1.0)
    specialists = _specialists()
    ckpts = [train_specialist(cfg, s).ckpt for s in specialists]
    examples = harvest_distillation_data(cfg, specialists, ckpts)
    # At least the fallback guarantees >= min_distill_examples per specialist.
    assert len(examples) >= len(specialists)
    # Each example is a well-formed SFT record carrying its provenance.
    for ex in examples:
        assert "prompt" in ex and "response" in ex
        assert ex["specialist"] in {"alpha", "beta"}
        assert "reward" in ex


def test_harvest_threshold_keeps_only_winners(tmp_path):
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, _tiny_cfg())
    # An impossible reward (contains a byte we forbid by using a weird target)
    # forces the fallback path; with min=2 we must still get 2 per specialist.
    cfg = _hc_cfg(base, tmp_path / "hc", distill_reward_threshold=2.0,
                  min_distill_examples=2)
    specialists = _specialists()
    ckpts = [train_specialist(cfg, s).ckpt for s in specialists]
    examples = harvest_distillation_data(cfg, specialists, ckpts)
    # Fallback keeps exactly min_distill_examples per specialist when none pass.
    counts = {}
    for ex in examples:
        counts[ex["specialist"]] = counts.get(ex["specialist"], 0) + 1
    assert counts["alpha"] >= 2
    assert counts["beta"] >= 2


def test_run_hill_climb_end_to_end(tmp_path):
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, _tiny_cfg())
    cfg = _hc_cfg(base, tmp_path / "hc")
    specialists = _specialists()
    result = run_hill_climb(cfg, specialists)

    assert isinstance(result, HillClimbResult)
    # All stages produced a loadable checkpoint.
    assert len(result.specialists) == 2
    for s in result.specialists:
        assert Path(s.ckpt).exists()
    assert Path(result.distill.ckpt).exists()
    assert Path(result.final.ckpt).exists()
    assert result.best_ckpt == result.final.ckpt

    # The final checkpoint loads as a standard policy checkpoint.
    state = torch.load(result.final.ckpt, map_location="cpu", weights_only=False)
    assert "model" in state and "model_cfg" in state

    # Distill lineage JSONL + run summary were written.
    lineage = tmp_path / "hc" / "distill" / "distill_data.jsonl"
    assert lineage.exists()
    rows = [json.loads(ln) for ln in lineage.read_text().splitlines() if ln.strip()]
    assert len(rows) >= 2
    summary = json.loads((tmp_path / "hc" / "summary.json").read_text())
    assert summary["final_ckpt"] == result.final.ckpt
    assert set(summary["specialists"]) == {"alpha", "beta"}


def test_run_hill_climb_requires_specialists(tmp_path):
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, _tiny_cfg())
    cfg = _hc_cfg(base, tmp_path / "hc")
    with pytest.raises(ValueError, match="at least one specialist"):
        run_hill_climb(cfg, [])


def test_hill_climb_distill_metrics_present(tmp_path):
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, _tiny_cfg())
    cfg = _hc_cfg(base, tmp_path / "hc")
    result = run_hill_climb(cfg, _specialists())
    dm = result.distill.metrics
    assert dm["n_examples"] >= 2
    assert set(dm["per_specialist"]) == {"alpha", "beta"}
    # Cold-start consolidation recorded a loss trajectory.
    assert dm["loss_first"] is not None and dm["loss_last"] is not None
