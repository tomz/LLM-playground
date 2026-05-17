"""Direct Preference Optimization. No reward model required."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer
from ._common import (
    clone_for_reference,
    compute_logps,
    load_pref_jsonl,
    tokenize_and_pack,
)


@dataclass
class DPOConfig:
    policy_ckpt: str
    pref_set: str
    ref_ckpt: str = ""               # defaults to a frozen copy of policy
    out_dir: str = "out/dpo"
    beta: float = 0.1
    lr: float = 5e-6
    steps: int = 30
    batch_size: int = 4
    seq_len: int = 256
    epochs: int = 1
    loss_variant: str = "sigmoid"   # 'sigmoid' | 'ipo' | 'kto'
    model_cfg: ModelConfig | None = None


def dpo_loss(
    policy_logps_chosen: torch.Tensor,
    policy_logps_rejected: torch.Tensor,
    ref_logps_chosen: torch.Tensor,
    ref_logps_rejected: torch.Tensor,
    beta: float,
    variant: str = "sigmoid",
) -> torch.Tensor:
    """DPO loss family.

    Args are per-sequence summed log-probs ``[B]`` for each (policy|ref)×(chosen|rejected).

    - "sigmoid": -log σ(β · ((p_c − r_c) − (p_r − r_r))).mean()
    - "ipo":     ((p_c − r_c) − (p_r − r_r) − 1/(2β))**2 .mean()
    - "kto":     simplified — sigmoid loss with extra weight on chosen logp shift.
    """
    chosen_logratio = policy_logps_chosen - ref_logps_chosen
    rejected_logratio = policy_logps_rejected - ref_logps_rejected
    diff = chosen_logratio - rejected_logratio

    if variant == "sigmoid":
        return -F.logsigmoid(beta * diff).mean()
    if variant == "ipo":
        return ((diff - 1.0 / (2.0 * beta)) ** 2).mean()
    if variant == "kto":
        # Toy KTO: encourage chosen logratio up, rejected down, independently.
        loss_c = -F.logsigmoid(beta * chosen_logratio).mean()
        loss_r = -F.logsigmoid(-beta * rejected_logratio).mean()
        return 0.5 * (loss_c + loss_r)
    raise ValueError(f"unknown DPO variant: {variant}")


def _load_policy(cfg: DPOConfig) -> tuple[Transformer, BytesTokenizer]:
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


def _shift(ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ids[:, :-1], ids[:, 1:], mask[:, 1:]


def run_dpo(cfg: DPOConfig) -> str:
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

    prefs = load_pref_jsonl(cfg.pref_set)
    ids_c, mask_c = tokenize_and_pack(
        [{"prompt": r["prompt"], "response": r["chosen"]} for r in prefs],
        tok, cfg.seq_len, mask_user_tokens=True,
    )
    ids_r, mask_r = tokenize_and_pack(
        [{"prompt": r["prompt"], "response": r["rejected"]} for r in prefs],
        tok, cfg.seq_len, mask_user_tokens=True,
    )

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    n = ids_c.shape[0]
    bs = min(cfg.batch_size, n)
    rng = torch.Generator().manual_seed(0)
    history: list[dict] = []

    for step in range(cfg.steps):
        perm = torch.randint(0, n, (bs,), generator=rng)
        xc, yc, mc = _shift(ids_c[perm], mask_c[perm])
        xr, yr, mr = _shift(ids_r[perm], mask_r[perm])

        with torch.no_grad():
            rlp_c = compute_logps(ref, xc, yc, mc)
            rlp_r = compute_logps(ref, xr, yr, mr)
        plp_c = compute_logps(policy, xc, yc, mc)
        plp_r = compute_logps(policy, xr, yr, mr)

        loss = dpo_loss(plp_c, plp_r, rlp_c, rlp_r, beta=cfg.beta, variant=cfg.loss_variant)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        history.append({"loss": float(loss.detach()), "gap": float((plp_c - plp_r).mean().detach())})

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dpo.pt"
    torch.save(
        {"model": policy.state_dict(), "model_cfg": policy.cfg, "history": history},
        out_path,
    )
    return str(out_path)
