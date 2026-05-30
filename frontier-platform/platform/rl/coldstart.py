"""Reasoning-SFT cold-start for RLVR (see docs/15-reasoning-rl-rlvr.md).

DeepSeek-R1 found that running GRPO directly on a base model works (R1-Zero) but
produces messy, hard-to-read chains of thought. A small **cold-start** SFT on a
few thousand high-quality long-CoT traces *before* RL stabilizes the output
format (think→answer structure), so the verifier can reliably extract answers and
the RL phase spends its budget improving reasoning rather than learning syntax.

This is a thin wrapper over the standard SFT loss masking (``_common``): given
(prompt, reasoning_trace) examples — where the response already contains the
``<think>...</think>`` then final-answer format — fine-tune the policy on the
response tokens only. Output is a checkpoint the GRPO loop loads as its policy.

Toy-functional: runs on CPU with the byte tokenizer and tiny test model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..alignment._common import compute_logps, tokenize_and_pack
from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer


@dataclass
class ColdStartConfig:
    policy_ckpt: str = ""
    out_dir: str = "out/coldstart"
    epochs: int = 3
    lr: float = 1e-5
    batch_size: int = 4
    seq_len: int = 256
    grad_clip: float = 1.0
    model_cfg: ModelConfig | None = None


def format_trace(question: str, think: str, answer: str) -> dict:
    """Build a single cold-start example in the canonical reasoning format.

    The response carries the ``<think>...</think>`` block then a boxed answer,
    matching what ``platform.rl.reward.format_reward`` rewards during RL.
    """
    response = f"<think>{think}</think>\\boxed{{{answer}}}"
    return {"prompt": question, "response": response}


@dataclass
class ColdStartResult:
    out_path: str
    loss_history: list[float] = field(default_factory=list)


def _load_policy(cfg: ColdStartConfig) -> tuple[Transformer, BytesTokenizer]:
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


def run_coldstart(cfg: ColdStartConfig, examples: list[dict]) -> ColdStartResult:
    """Reasoning-SFT on long-CoT ``examples`` (each: prompt, response).

    Returns the checkpoint path + loss history. The checkpoint is in the same
    format ``run_grpo`` expects for ``policy_ckpt``.
    """
    policy, tok = _load_policy(cfg)
    policy.train()
    ids, mask = tokenize_and_pack(examples, tok, cfg.seq_len, mask_user_tokens=True)
    n = ids.shape[0]
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)

    rng = torch.Generator().manual_seed(0)
    total = cfg.epochs * max(1, n // cfg.batch_size)
    history: list[float] = []
    for _ in range(total):
        perm = torch.randint(0, n, (cfg.batch_size,), generator=rng)
        bx, bm = ids[perm], mask[perm]
        x, y, m = bx[:, :-1], bx[:, 1:], bm[:, 1:]
        logp = compute_logps(policy, x, y, m)          # [B] summed over response
        denom = m.sum().clamp_min(1.0)
        loss = -logp.sum() / denom
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        opt.step()
        history.append(float(loss.detach()))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coldstart.pt"
    torch.save(
        {"model": policy.state_dict(), "model_cfg": policy.cfg, "loss_history": history},
        out_path,
    )
    return ColdStartResult(out_path=str(out_path), loss_history=history)
