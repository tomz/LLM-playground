"""LoRA Without Regret pins (Schulman et al., TML 2025).

These guard the four findings of the paper against silent config drift — the
recipe is "config + docs", so the *config* is the artifact under test:

  1. **All-linear, never attention-only.** Every LoRA-bearing config must target
     the MLP projections (gate/up/down), not just attention. This is the single
     mistake the paper calls out: attention-only LoRA underperforms even at
     matched param count via higher rank, because the MLP matrices carry the
     capacity.
  2. **High-capacity SFT recipe exists and is high-rank.** `lora_hicap.yaml`
     must be r≳128 all-linear — the post-training-scale SFT rank.
  3. **LR is rank-independent.** The hicap recipe keeps the same LR as the r=16
     recipes (1/r scaling makes the optimal LoRA LR ~rank-independent), so it
     must NOT have been quietly lowered for the bigger rank.
  4. **RL stays low-rank.** GRPO/DPO adapters extract ~1 bit/episode, so they
     must stay small (r≤32) — they should be cheaper than the SFT adapter, never
     bumped toward the SFT rank.

Pure YAML reads — no model, no network, sub-second.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"

# MLP projections that *must* be present for an "all-linear" LoRA target set.
MLP_PROJECTIONS = {"gate_proj", "up_proj", "down_proj"}
ATTN_PROJECTIONS = {"q_proj", "k_proj", "v_proj", "o_proj"}


def _load(name: str) -> dict:
    with open(CONFIGS / name) as f:
        return yaml.safe_load(f)


def _lora_configs() -> list[tuple[str, dict]]:
    """Every config in configs/ that carries a `lora.target_modules` list."""
    out = []
    for path in sorted(CONFIGS.glob("*.yaml")):
        cfg = _load(path.name)
        lcfg = cfg.get("lora") or {}
        if isinstance(lcfg.get("target_modules"), list):
            out.append((path.name, cfg))
    return out


def test_there_are_lora_configs_to_check():
    """Guard the guard: if the glob finds nothing, the rest silently passes."""
    assert _lora_configs(), "no LoRA-bearing configs found — pin is vacuous"


@pytest.mark.parametrize("name,cfg", _lora_configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_lora_config_targets_all_linear_not_attention_only(name, cfg):
    """Finding #1: never narrow LoRA back to attention-only. Every LoRA config
    must include the MLP projections alongside attention."""
    targets = set(cfg["lora"]["target_modules"])
    missing = MLP_PROJECTIONS - targets
    assert not missing, (
        f"{name}: LoRA target_modules is missing MLP projections {missing} — "
        f"that's the attention-only mistake LoRA Without Regret warns against. "
        f"Add gate/up/down_proj."
    )
    # Sanity: it should still cover attention too (full all-linear set).
    assert ATTN_PROJECTIONS <= targets, f"{name}: missing attention projections"


def test_hicap_recipe_is_high_rank_all_linear():
    """Finding #2: the high-capacity SFT recipe must actually be high-rank."""
    cfg = _load("lora_hicap.yaml")
    assert cfg["method"] == "lora"
    r = cfg["lora"]["r"]
    assert r >= 128, f"lora_hicap.yaml rank r={r} is not post-training-scale (want ≳128/256)"
    targets = set(cfg["lora"]["target_modules"])
    assert MLP_PROJECTIONS <= targets and ATTN_PROJECTIONS <= targets
    # rsLoRA is the load-bearing knob at high rank (alpha/sqrt(r) scaling).
    assert cfg["lora"].get("use_rslora") is True, "high rank needs rsLoRA scaling"


def test_hicap_lr_is_rank_independent_not_lowered():
    """Finding #3: the optimal LoRA LR is ~rank-independent (1/r scaling), so the
    high-rank recipe keeps the same LR as the r=16 recipes — it must NOT have
    been quietly scaled down for the bigger rank."""
    hicap = _load("lora_hicap.yaml")
    baseline = _load("lora.yaml")  # the canonical r=16 SFT recipe
    assert float(hicap["train"]["lr"]) == pytest.approx(float(baseline["train"]["lr"])), (
        "lora_hicap LR diverged from the r=16 baseline — LoRA LR should be "
        "rank-independent, not lowered for higher rank"
    )


def test_hicap_effective_batch_under_32():
    """Finding #4: LoRA is less batch-tolerant than FullFT — keep effective
    batch (batch_size × grad_accum) under 32."""
    t = _load("lora_hicap.yaml")["train"]
    eff = int(t["batch_size"]) * int(t["grad_accum"])
    assert eff < 32, f"lora_hicap effective batch {eff} ≥ 32 — LoRA wants it smaller"


@pytest.mark.parametrize("name", ["grpo_3050.yaml", "dpo_3050.yaml"])
def test_rl_adapters_stay_low_rank(name):
    """Finding #3 (RL side): policy-gradient needs almost no rank (~1 bit/
    episode), so RL LoRAs must stay small (r≤32) — cheaper than the SFT adapter,
    never bumped toward the r=256 SFT rank."""
    cfg = _load(name)
    r = cfg["lora"]["r"]
    assert r <= 32, (
        f"{name}: RL adapter rank r={r} is too high — RL extracts ~1 bit/episode, "
        f"so r=1–32 suffices. Don't reuse the SFT high-rank recipe for RL."
    )
