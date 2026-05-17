"""Supervised fine-tuning on (prompt, response) pairs with assistant-token loss masking."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer
from ..training.optim import OptimConfig, build_optimizer
from ._common import load_sft_jsonl, tokenize_and_pack


@dataclass
class SFTConfig:
    base_ckpt: str
    train_set: str
    eval_set: str
    out_dir: str = "out/sft"
    epochs: int = 3
    steps: int = 0                   # 0 → epochs * (n_train // batch_size)
    lr: float = 1e-5
    batch_size: int = 4
    seq_len: int = 256
    pack_examples: bool = True
    loss_mask_user_tokens: bool = True
    model_cfg: ModelConfig | None = None  # required if base_ckpt has no model_cfg


def _load_base(cfg: SFTConfig) -> tuple[Transformer, object]:
    """Load Transformer + tokenizer from base ckpt or from cfg.model_cfg."""
    tok = BytesTokenizer()
    if cfg.base_ckpt and Path(cfg.base_ckpt).exists():
        state = torch.load(cfg.base_ckpt, map_location="cpu", weights_only=False)
        mcfg = state.get("model_cfg") or cfg.model_cfg
        if mcfg is None:
            raise ValueError("base_ckpt has no model_cfg; pass cfg.model_cfg")
        model = Transformer(mcfg)
        if "model" in state:
            model.load_state_dict(state["model"])
    else:
        if cfg.model_cfg is None:
            raise ValueError("either base_ckpt must exist or cfg.model_cfg must be set")
        model = Transformer(cfg.model_cfg)
    return model, tok


def _sft_loss(model, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy, masked. input_ids [B,T], loss_mask [B,T]."""
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    m = loss_mask[:, 1:]
    logits, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    denom = m.sum().clamp_min(1.0)
    return -(gathered * m).sum() / denom


def run_sft(cfg: SFTConfig) -> str:
    """Train SFT and return the URI of the resulting checkpoint."""
    model, tok = _load_base(cfg)
    model.train()
    train_examples = load_sft_jsonl(cfg.train_set)
    ids, mask = tokenize_and_pack(
        train_examples, tok, cfg.seq_len, mask_user_tokens=cfg.loss_mask_user_tokens
    )

    ocfg = OptimConfig(
        peak_lr=cfg.lr,
        warmup_steps=5,
        total_steps=max(10, cfg.steps or cfg.epochs * max(1, len(train_examples) // cfg.batch_size)),
        weight_decay=0.0,
    )
    opt, sched = build_optimizer(model, ocfg)
    bs = cfg.batch_size
    n = ids.shape[0]
    total = cfg.steps or cfg.epochs * max(1, n // bs)

    loss_history: list[float] = []
    rng = torch.Generator().manual_seed(0)
    for step in range(total):
        perm = torch.randint(0, n, (bs,), generator=rng)
        bx = ids[perm]
        bm = mask[perm]
        loss = _sft_loss(model, bx, bm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_history.append(float(loss.detach()))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sft.pt"
    torch.save(
        {"model": model.state_dict(), "model_cfg": model.cfg, "loss_history": loss_history},
        out_path,
    )
    return str(out_path)
