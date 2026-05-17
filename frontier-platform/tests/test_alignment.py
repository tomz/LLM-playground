"""Tier 3 alignment tests: SFT, RM, DPO, PPO."""
from __future__ import annotations
import json
from pathlib import Path

import pytest  # noqa: F401
import torch

from platform.alignment._common import (
    compute_logps,
    tokenize_and_pack,
)
from platform.alignment.dpo import DPOConfig, dpo_loss, run_dpo
from platform.alignment.ppo import PPOConfig, ValueHead, ppo_step, rollout
from platform.alignment.reward_model import (
    RMConfig,
    bt_loss,
    load_reward_model,
    train_reward_model,
)
from platform.alignment.sft import SFTConfig, run_sft
from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.tokenizer.bytes import BytesTokenizer


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
    )


def _write_sft_jsonl(path: Path, n: int = 10) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"prompt": f"Q: item {i}", "response": f"A: ans {i}"}) + "\n")


def _write_pref_jsonl(path: Path, n: int = 8) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "prompt": f"Q: thing {i}",
                "chosen": f"A: good {i}",
                "rejected": f"BAD wrong {i}",
            }) + "\n")


def _save_base_ckpt(path: Path, cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    m = Transformer(cfg)
    torch.save({"model": m.state_dict(), "model_cfg": cfg}, path)


# ---------- common helpers ----------

def test_tokenize_and_pack_masks_user_tokens():
    tok = BytesTokenizer()
    ex = [{"prompt": "hello", "response": "world"}]
    ids, mask = tokenize_and_pack(ex, tok, seq_len=32, mask_user_tokens=True)
    # The prompt + bos region should be masked (0); response + eos unmasked (1).
    # Response "world" (5 bytes) + eos = 6 tokens at the tail of the non-pad region.
    L = (ids[0] != tok.pad_id).sum().item()
    assert L == 1 + len("hello") + len("world") + 1  # bos+prompt+resp+eos
    assert mask[0, :L - 6].sum().item() == 0          # prompt+bos all masked
    assert mask[0, L - 6:L].sum().item() == 6.0       # response+eos all unmasked
    assert mask[0, L:].sum().item() == 0              # padding masked


def test_compute_logps_matches_hand_calc():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    m = Transformer(cfg)
    m.eval()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    y = torch.tensor([[2, 3, 4, 5, 6]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

    summed = compute_logps(m, x, y, mask)
    # Hand calculation
    with torch.no_grad():
        logits, _ = m(x)
        lp = torch.log_softmax(logits.float(), dim=-1)
        manual = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    assert torch.allclose(summed, manual, atol=1e-5)

    # Masking zeros out positions
    mask2 = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    s2 = compute_logps(m, x, y, mask2)
    manual2 = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    expected = (manual2 * mask2).sum(dim=-1)
    assert torch.allclose(s2, expected, atol=1e-5)


def test_bt_loss_chosen_higher_means_lower_loss():
    s_c = torch.tensor([1.0, 2.0, 3.0])
    s_r = torch.tensor([0.0, 0.0, 0.0])
    high = bt_loss(s_c, s_r)
    low = bt_loss(s_r, s_c)  # rejected wins → big loss
    assert high < low


# ---------- SFT ----------

def test_sft_runs_and_loss_decreases(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    train = tmp_path / "train.jsonl"
    _write_sft_jsonl(train, n=5)
    eval_ = tmp_path / "eval.jsonl"
    _write_sft_jsonl(eval_, n=2)

    sftcfg = SFTConfig(
        base_ckpt=str(base), train_set=str(train), eval_set=str(eval_),
        out_dir=str(tmp_path / "out_sft"),
        steps=20, batch_size=2, seq_len=32, lr=3e-3,
    )
    out = run_sft(sftcfg)
    assert Path(out).exists()
    state = torch.load(out, map_location="cpu", weights_only=False)
    h = state["loss_history"]
    assert len(h) == 20
    assert sum(h[-5:]) / 5 < sum(h[:5]) / 5


# ---------- Reward model ----------

def test_reward_model_separates_chosen_from_rejected(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    pref = tmp_path / "pref.jsonl"
    _write_pref_jsonl(pref, n=6)

    rmcfg = RMConfig(
        base_ckpt=str(base), pref_set=str(pref),
        out_dir=str(tmp_path / "out_rm"),
        steps=50, batch_size=2, seq_len=32, lr=3e-3,
    )
    out = train_reward_model(rmcfg)
    rm = load_reward_model(out)

    tok = BytesTokenizer()
    prefs = [json.loads(line) for line in open(pref)]
    chosen = [{"prompt": r["prompt"], "response": r["chosen"]} for r in prefs]
    rejected = [{"prompt": r["prompt"], "response": r["rejected"]} for r in prefs]
    ids_c, _ = tokenize_and_pack(chosen, tok, 32, mask_user_tokens=False)
    ids_r, _ = tokenize_and_pack(rejected, tok, 32, mask_user_tokens=False)
    with torch.no_grad():
        s_c = rm(ids_c).mean().item()
        s_r = rm(ids_r).mean().item()
    assert s_c > s_r


# ---------- DPO ----------

def test_dpo_step_decreases_logp_gap(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    pref = tmp_path / "pref.jsonl"
    _write_pref_jsonl(pref, n=6)
    dcfg = DPOConfig(
        policy_ckpt=str(base), pref_set=str(pref),
        out_dir=str(tmp_path / "out_dpo"),
        steps=20, batch_size=2, seq_len=32, lr=3e-3, beta=0.1,
    )
    out = run_dpo(dcfg)
    state = torch.load(out, map_location="cpu", weights_only=False)
    hist = state["history"]
    assert len(hist) == 20
    initial = sum(h["gap"] for h in hist[:3]) / 3
    final = sum(h["gap"] for h in hist[-3:]) / 3
    assert final > initial


def test_dpo_variant_ipo_runs(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    pref = tmp_path / "pref.jsonl"
    _write_pref_jsonl(pref, n=4)
    dcfg = DPOConfig(
        policy_ckpt=str(base), pref_set=str(pref),
        out_dir=str(tmp_path / "out_dpo_ipo"),
        steps=5, batch_size=2, seq_len=32, lr=1e-4, beta=0.1,
        loss_variant="ipo",
    )
    out = run_dpo(dcfg)
    assert Path(out).exists()

    # Also sanity-check the math directly
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([0.0, 0.0])
    c = torch.tensor([0.5, 1.5])
    d = torch.tensor([0.5, 1.0])
    L = dpo_loss(a, b, c, d, beta=0.1, variant="ipo")
    assert torch.isfinite(L)


# ---------- PPO ----------

def _make_policy_and_rm(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    pref = tmp_path / "pref.jsonl"
    _write_pref_jsonl(pref, n=4)
    rmcfg = RMConfig(
        base_ckpt=str(base), pref_set=str(pref),
        out_dir=str(tmp_path / "out_rm"),
        steps=5, batch_size=2, seq_len=32, lr=1e-3,
    )
    rm_path = train_reward_model(rmcfg)
    return str(base), rm_path


def test_ppo_rollout_no_nan(tmp_path):
    base, rm_path = _make_policy_and_rm(tmp_path)
    state = torch.load(base, map_location="cpu", weights_only=False)
    policy = Transformer(state["model_cfg"])
    policy.load_state_dict(state["model"])
    rm = load_reward_model(rm_path)
    vh = ValueHead(policy.cfg.d_model)
    tok = BytesTokenizer()
    prompts = [tok.encode("Q: x"), tok.encode("Q: y"), tok.encode("Q: z"), tok.encode("Q: w")]
    cfg = PPOConfig(policy_ckpt=base, rm_ckpt=rm_path, rollout_batch=4,
                    max_new_tokens=4, seq_len=32, ppo_epochs=1)
    traj = rollout(policy, prompts, cfg, value_head=vh, rm=rm, tokenizer=tok)
    for k in ("old_logps", "ref_logps", "values", "advantages", "returns"):
        t = traj[k]
        assert torch.isfinite(t).all(), f"{k} has non-finite values"


def test_ppo_step_runs_and_kl_finite(tmp_path):
    base, rm_path = _make_policy_and_rm(tmp_path)
    state = torch.load(base, map_location="cpu", weights_only=False)
    policy = Transformer(state["model_cfg"])
    policy.load_state_dict(state["model"])
    rm = load_reward_model(rm_path)
    vh = ValueHead(policy.cfg.d_model)
    opt = torch.optim.AdamW(list(policy.parameters()) + list(vh.parameters()), lr=1e-4)
    tok = BytesTokenizer()
    prompts = [tok.encode("Q: a"), tok.encode("Q: b")]
    cfg = PPOConfig(policy_ckpt=base, rm_ckpt=rm_path, rollout_batch=2,
                    max_new_tokens=4, seq_len=32, ppo_epochs=2)
    traj = rollout(policy, prompts, cfg, value_head=vh, rm=rm, tokenizer=tok)
    metrics = ppo_step(policy, vh, traj, cfg, optimizer=opt)
    import math
    assert math.isfinite(metrics["kl"])
    assert math.isfinite(metrics["loss_pi"])
    assert math.isfinite(metrics["loss"])
