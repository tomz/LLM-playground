"""Bradley-Terry reward model: shared trunk + scalar head over the SFT model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer
from ._common import (
    forward_hidden,
    last_nonpad_index,
    load_pref_jsonl,
    tokenize_and_pack,
)


@dataclass
class RMConfig:
    base_ckpt: str
    pref_set: str
    out_dir: str = "out/rm"
    epochs: int = 1
    steps: int = 50
    lr: float = 5e-6
    batch_size: int = 4
    seq_len: int = 256
    margin: float = 0.0
    model_cfg: ModelConfig | None = None


def bt_loss(score_chosen: torch.Tensor, score_rejected: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """-log sigmoid(s_c - s_r - margin), reduced over batch."""
    return -F.logsigmoid(score_chosen - score_rejected - margin).mean()


class RewardModel(nn.Module):
    """Transformer trunk + scalar head on the last non-pad token."""

    def __init__(self, base: Transformer, pad_id: int):
        super().__init__()
        self.trunk = base
        self.pad_id = int(pad_id)
        self.head = nn.Linear(base.cfg.d_model, 1, bias=False)
        nn.init.normal_(self.head.weight, std=1.0 / (base.cfg.d_model ** 0.5))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = forward_hidden(self.trunk, input_ids)            # [B, T, D]
        idx = last_nonpad_index(input_ids, self.pad_id)      # [B]
        gathered = h[torch.arange(h.shape[0], device=h.device), idx]  # [B, D]
        return self.head(gathered).squeeze(-1)               # [B]


def _build_trunk(cfg: RMConfig) -> tuple[Transformer, BytesTokenizer]:
    tok = BytesTokenizer()
    if cfg.base_ckpt and Path(cfg.base_ckpt).exists():
        state = torch.load(cfg.base_ckpt, map_location="cpu", weights_only=False)
        mcfg = state.get("model_cfg") or cfg.model_cfg
        if mcfg is None:
            raise ValueError("base_ckpt has no model_cfg; pass cfg.model_cfg")
        trunk = Transformer(mcfg)
        if "model" in state:
            trunk.load_state_dict(state["model"])
    else:
        if cfg.model_cfg is None:
            raise ValueError("either base_ckpt must exist or cfg.model_cfg must be set")
        trunk = Transformer(cfg.model_cfg)
    return trunk, tok


def train_reward_model(cfg: RMConfig) -> str:
    trunk, tok = _build_trunk(cfg)
    rm = RewardModel(trunk, pad_id=tok.pad_id)
    rm.train()
    prefs = load_pref_jsonl(cfg.pref_set)

    chosen_examples = [{"prompt": r["prompt"], "response": r["chosen"]} for r in prefs]
    rejected_examples = [{"prompt": r["prompt"], "response": r["rejected"]} for r in prefs]
    ids_c, _ = tokenize_and_pack(chosen_examples, tok, cfg.seq_len, mask_user_tokens=False)
    ids_r, _ = tokenize_and_pack(rejected_examples, tok, cfg.seq_len, mask_user_tokens=False)

    opt = torch.optim.AdamW(rm.parameters(), lr=cfg.lr, weight_decay=0.0)
    n = ids_c.shape[0]
    bs = min(cfg.batch_size, n)
    rng = torch.Generator().manual_seed(0)

    for step in range(cfg.steps):
        perm = torch.randint(0, n, (bs,), generator=rng)
        s_c = rm(ids_c[perm])
        s_r = rm(ids_r[perm])
        loss = bt_loss(s_c, s_r, margin=cfg.margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rm.pt"
    torch.save(
        {
            "trunk": trunk.state_dict(),
            "head": rm.head.state_dict(),
            "model_cfg": trunk.cfg,
            "pad_id": tok.pad_id,
        },
        out_path,
    )
    return str(out_path)


def load_reward_model(rm_ckpt: str) -> RewardModel:
    state = torch.load(rm_ckpt, map_location="cpu", weights_only=False)
    trunk = Transformer(state["model_cfg"])
    trunk.load_state_dict(state["trunk"])
    rm = RewardModel(trunk, pad_id=int(state["pad_id"]))
    rm.head.load_state_dict(state["head"])
    rm.eval()
    return rm


def calibrate(rm_ckpt: str, probe_set: str) -> dict:
    """Compute mean/std reward, length-vs-reward correlation, refusal-rate.

    ``probe_set`` is a JSONL of ``{"prompt", "response", "label"}`` where
    label ∈ {"accept", "refuse"}.
    """
    import json

    rm = load_reward_model(rm_ckpt)
    tok = BytesTokenizer()
    rows: list[dict] = []
    with open(probe_set, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return {"mean_reward": 0.0, "std_reward": 0.0, "length_corr": 0.0, "refusal_rate": 0.0}

    examples = [{"prompt": r["prompt"], "response": r["response"]} for r in rows]
    ids, _ = tokenize_and_pack(examples, tok, seq_len=256, mask_user_tokens=False)
    with torch.no_grad():
        scores = rm(ids).float().cpu()
    lengths = torch.tensor([len(tok.encode(r["response"])) for r in rows], dtype=torch.float32)
    refusal_rate = sum(1 for r in rows if r.get("label") == "refuse") / len(rows)

    if scores.std().item() > 0 and lengths.std().item() > 0:
        s = (scores - scores.mean()) / scores.std()
        L = (lengths - lengths.mean()) / lengths.std()
        corr = float((s * L).mean())
    else:
        corr = 0.0
    return {
        "mean_reward": float(scores.mean()),
        "std_reward": float(scores.std()),
        "length_corr": corr,
        "refusal_rate": float(refusal_rate),
    }
