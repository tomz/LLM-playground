"""Supervised fine-tuning on (prompt, response) pairs with assistant-token loss masking."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SFTConfig:
    base_ckpt: str
    train_set: str
    eval_set: str
    epochs: int = 3
    lr: float = 1e-5
    seq_len: int = 8192
    pack_examples: bool = True   # multi-pack with attention masking
    loss_mask_user_tokens: bool = True


def run_sft(cfg: SFTConfig) -> str:
    """Returns URI of resulting SFT checkpoint."""
    raise NotImplementedError
