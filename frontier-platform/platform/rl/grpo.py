"""GRPO — Group-Relative Policy Optimization (DeepSeekMath / DeepSeek-R1).

The key idea vs. PPO (see `platform.alignment.ppo`): drop the value network.
For each prompt, sample a group of G responses, score each with a *verifier*
(see `platform.rl.verifiers`), and use the group's mean/std to form a baseline:

    advantage_i = (r_i - mean(group)) / (std(group) + eps)
    loss        = -(advantage_i * sum_t logp(token_t))  +  beta * KL(pi || pi_ref)

This toy implementation uses a REINFORCE-style per-sequence log-prob (summed over
generated tokens) rather than per-token PPO ratios — enough to demonstrate the
GRPO advantage/objective end-to-end on CPU. Production GRPO keeps per-token
importance ratios and a clipped objective; the advantage computation is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..alignment._common import clone_for_reference, compute_logps
from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer
from .rollout import GroupRollout, sample_group
from .verifiers import Verifier


@dataclass
class GRPOConfig:
    policy_ckpt: str = ""
    ref_ckpt: str = ""                 # defaults to a frozen copy of the policy
    out_dir: str = "out/grpo"
    group_size: int = 4               # G rollouts per prompt
    steps: int = 10
    lr: float = 1e-5
    beta: float = 0.04                # KL-to-reference coefficient
    max_new_tokens: int = 16
    seq_len: int = 256
    temperature: float = 1.0
    grad_clip: float = 1.0
    model_cfg: ModelConfig | None = None


def group_advantages(rewards: torch.Tensor, group_index: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative, standardized advantages.

    ``rewards`` and ``group_index`` are ``[N]``. Within each group (prompt), the
    advantage is ``(r - group_mean) / (group_std + eps)``. Returns ``[N]``.
    """
    adv = torch.zeros_like(rewards)
    for gi in group_index.unique():
        sel = group_index == gi
        g = rewards[sel]
        adv[sel] = (g - g.mean()) / (g.std(unbiased=False) + eps)
    return adv


def _shift(ids: torch.Tensor, mask: torch.Tensor):
    return ids[:, :-1], ids[:, 1:], mask[:, 1:]


def grpo_step(
    policy,
    ref,
    roll: GroupRollout,
    rewards: torch.Tensor,
    cfg: GRPOConfig,
    *,
    optimizer=None,
) -> dict:
    """One GRPO update over a group rollout. Returns metrics dict."""
    adv = group_advantages(rewards, roll.group_index)        # [N]
    x, y, m = _shift(roll.ids, roll.resp_mask)

    policy.train()
    logp = compute_logps(policy, x, y, m)                    # [N] sum over generated tokens
    with torch.no_grad():
        ref_logp = compute_logps(ref, x, y, m)
    kl = (logp - ref_logp)                                   # [N] surrogate KL(pi||ref)

    pg_loss = -(adv.detach() * logp).mean()
    kl_loss = cfg.beta * kl.mean()
    loss = pg_loss + kl_loss

    metrics = {
        "loss": float(loss.detach()),
        "pg_loss": float(pg_loss.detach()),
        "kl": float(kl.mean().detach()),
        "reward_mean": float(rewards.mean().detach()),
        "reward_std": float(rewards.std(unbiased=False).detach()),
        "adv_abs_mean": float(adv.abs().mean().detach()),
    }
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()
    return metrics


def _load_policy(cfg: GRPOConfig) -> tuple[Transformer, BytesTokenizer]:
    tok = BytesTokenizer()
    if cfg.policy_ckpt and Path(cfg.policy_ckpt).exists():
        state = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
        mcfg = state.get("model_cfg") or cfg.model_cfg
        if mcfg is None:
            raise ValueError("policy_ckpt has no model_cfg; pass cfg.model_cfg")
        m = Transformer(mcfg)
        if "model" in state:
            m.load_state_dict(state["model"])
        return m, tok
    if cfg.model_cfg is None:
        raise ValueError("either policy_ckpt must exist or cfg.model_cfg must be set")
    return Transformer(cfg.model_cfg), tok


def run_grpo(
    cfg: GRPOConfig,
    prompts: list[str],
    verifier: Verifier,
) -> str:
    """Run the GRPO loop against a single ``verifier`` over ``prompts``.

    For brevity this uses one verifier for all prompts; a real run carries a
    per-prompt verifier spec (the answer key / hidden tests live with the prompt).
    """
    policy, tok = _load_policy(cfg)
    if cfg.ref_ckpt and Path(cfg.ref_ckpt).exists():
        ref_state = torch.load(cfg.ref_ckpt, map_location="cpu", weights_only=False)
        ref = Transformer(ref_state["model_cfg"])
        ref.load_state_dict(ref_state["model"])
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
    else:
        ref = clone_for_reference(policy)

    prompt_ids = [tok.encode(p) for p in prompts]
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    history: list[dict] = []

    for step in range(cfg.steps):
        roll = sample_group(
            policy, prompt_ids,
            group_size=cfg.group_size,
            max_new_tokens=cfg.max_new_tokens,
            seq_len=cfg.seq_len,
            tokenizer=tok,
            temperature=cfg.temperature,
            seed=step,
        )
        rewards = torch.tensor(
            [verifier(prompts[int(gi)], txt) for gi, txt in zip(roll.group_index, roll.response_text)],
            dtype=torch.float32,
        )
        metrics = grpo_step(policy, ref, roll, rewards, cfg, optimizer=opt)
        history.append(metrics)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grpo.pt"
    torch.save(
        {"model": policy.state_dict(), "model_cfg": policy.cfg, "history": history},
        out_path,
    )
    return str(out_path)
