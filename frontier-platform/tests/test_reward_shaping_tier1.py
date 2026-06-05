"""Tier 1 reward-shaping harvests (#3 language consistency, #4 difficulty-aware
length) from the MAI-Thinking-1 deep dive.

See docs/research/mai-thinking-1-deep-dive.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.rl.reward import (
    CompositeReward,
    RewardConfig,
    language_consistency_reward,
    soft_length_penalty,
)
from platform.rl.verifiers import MathExactVerifier, reward_contains


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
    )


def _save_base_ckpt(path: Path, cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    m = Transformer(cfg)
    torch.save({"model": m.state_dict(), "model_cfg": cfg}, path)


# =====================================================================
# 3. Language-consistency reward
# =====================================================================

def test_language_consistency_zero_for_target_language():
    # Clear English prose -> detector says 'en' -> no penalty.
    english = ("the quick brown fox jumps over the lazy dog and then it would "
               "run back to the house with all of the other animals")
    assert language_consistency_reward(english, target_lang="en", coef=0.5) == 0.0


def test_language_consistency_penalizes_off_target():
    # Force a mismatch by targeting a language the English text isn't.
    english = ("this is a perfectly ordinary english sentence with many of the "
               "most common english stopwords so the detector is confident")
    pen = language_consistency_reward(english, target_lang="zh", coef=0.5)
    assert pen == pytest.approx(-0.5)


def test_language_consistency_unknown_is_neutral():
    # Numeric / symbolic answer -> 'unk' -> never penalized (no false positives
    # on short or non-prose answers).
    assert language_consistency_reward("\\boxed{42}", target_lang="en") == 0.0
    assert language_consistency_reward("12345 + 678", target_lang="en") == 0.0


def test_language_consistency_strips_think_and_answer_markup():
    # The <think> tags / boxed markup must not themselves count as tokens; the
    # detector should see the prose inside.
    resp = ("<think>the cat sat on the mat and then the dog came over to the "
            "house with all of the others</think>\\boxed{7}")
    # Still detected as English -> no penalty for target en.
    assert language_consistency_reward(resp, target_lang="en") == 0.0


def test_language_consistency_min_conf_gate():
    # A high min_conf makes a borderline detection fall back to neutral. This
    # text has just enough stopwords for an 'en' guess but at ~0.85 confidence,
    # below the 0.95 gate -> neutral (no false-positive penalty).
    text = "the zebra xylophone quartz monkey jungle kangaroo octopus penguin walrus"
    assert language_consistency_reward(text, target_lang="zh", coef=1.0,
                                       min_conf=0.95) == 0.0
    # Without the gate, the same text *is* penalized (it's detected as 'en').
    assert language_consistency_reward(text, target_lang="zh", coef=1.0,
                                       min_conf=0.0) == pytest.approx(-1.0)


def test_composite_reward_includes_language_when_weighted():
    base = reward_contains("answer")
    cfg = RewardConfig(language_weight=0.5, target_lang="zh")
    comp = CompositeReward(base, cfg)
    english = ("the answer is that the quick brown fox would run over to the "
               "house and then it could come back again with the others")
    bd = comp.breakdown("p", english)
    assert "language" in bd
    # Targeting zh on English prose -> language component is negative.
    assert bd["language"] == pytest.approx(-0.5)


def test_composite_reward_language_off_by_default():
    base = reward_contains("x")
    comp = CompositeReward(base, RewardConfig())  # language_weight=0.0
    bd = comp.breakdown("p", "some english text here for the detector to read")
    assert bd["language"] == 0.0


# =====================================================================
# 4. Difficulty-aware length penalty
# =====================================================================

def test_length_target_for_flat_when_disabled():
    cfg = RewardConfig(length_target_tokens=512, difficulty_aware=False)
    assert cfg.length_target_for(None) == 512
    assert cfg.length_target_for(0.9) == 512  # ignored when disabled


def test_length_target_for_interpolates_by_difficulty():
    cfg = RewardConfig(difficulty_aware=True,
                       length_target_easy=200, length_target_hard=1000)
    assert cfg.length_target_for(0.0) == 200
    assert cfg.length_target_for(1.0) == 1000
    assert cfg.length_target_for(0.5) == 600
    # None difficulty falls back to flat target even when aware.
    assert cfg.length_target_for(None) == cfg.length_target_tokens


def test_length_target_for_clamps_out_of_range():
    cfg = RewardConfig(difficulty_aware=True,
                       length_target_easy=200, length_target_hard=1000)
    assert cfg.length_target_for(-5.0) == 200
    assert cfg.length_target_for(9.0) == 1000


def test_difficulty_aware_length_penalty_in_composite():
    # A long response is penalized when treated as easy, but not when treated as
    # hard (higher token budget). Difficulty comes from a per-prompt fn.
    base = reward_contains("x")
    cfg = RewardConfig(difficulty_aware=True, length_target_easy=10,
                       length_target_hard=10_000, length_max_tokens=20_000,
                       length_coef=0.5)
    long_resp = "word " * 500 + "x"

    easy = CompositeReward(base, cfg, difficulty_fn=lambda p: 0.0)
    hard = CompositeReward(base, cfg, difficulty_fn=lambda p: 1.0)
    easy_len = easy.breakdown("p", long_resp)["length"]
    hard_len = hard.breakdown("p", long_resp)["length"]
    assert easy_len < 0.0          # over the small easy budget -> penalized
    assert hard_len == 0.0         # under the large hard budget -> free
    assert hard_len > easy_len


def test_difficulty_fn_none_uses_flat_target():
    base = reward_contains("x")
    cfg = RewardConfig(difficulty_aware=True, length_target_tokens=10,
                       length_target_easy=5, length_target_hard=10_000,
                       length_max_tokens=20_000, length_coef=0.5)
    long_resp = "word " * 500 + "x"
    # No difficulty_fn -> flat length_target_tokens (10) -> penalized.
    comp = CompositeReward(base, cfg, difficulty_fn=None)
    assert comp.breakdown("p", long_resp)["length"] < 0.0


def test_soft_length_penalty_still_standalone():
    # The base function is unchanged and still usable directly.
    assert soft_length_penalty("w " * 10, target_tokens=512) == 0.0
    assert soft_length_penalty("w " * 5000, target_tokens=512, max_tokens=2048,
                               coef=0.1) == pytest.approx(-0.1)


def test_composite_with_difficulty_and_tokenizer_runs_in_grpo(tmp_path):
    """End-to-end: a difficulty-aware composite reward drives a short GRPO run."""
    from platform.rl.grpo import GRPOConfig, run_grpo
    from platform.tokenizer.bytes import BytesTokenizer

    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)

    rcfg = RewardConfig(difficulty_aware=True, length_target_easy=4,
                        length_target_hard=64, length_coef=0.1)
    comp = CompositeReward(MathExactVerifier(4), rcfg,
                           tokenizer=BytesTokenizer(),
                           difficulty_fn=lambda p: 0.2)
    gcfg = GRPOConfig(policy_ckpt=str(base), out_dir=str(tmp_path / "out"),
                      group_size=4, steps=3, lr=1e-3, beta=0.0,
                      max_new_tokens=6, seq_len=32)
    out = run_grpo(gcfg, prompts=["2+2?"], verifier=comp)
    hist = torch.load(out, map_location="cpu", weights_only=False)["history"]
    # Composite breakdown logged each step, including the new language component.
    assert all("reward_length" in h and "reward_language" in h for h in hist)
