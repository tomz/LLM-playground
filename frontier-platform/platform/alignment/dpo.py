"""Direct Preference Optimization. No reward model required."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DPOConfig:
    policy_ckpt: str
    ref_ckpt: str
    pref_set: str
    beta: float = 0.1
    lr: float = 5e-7
    epochs: int = 1
    loss_variant: str = "sigmoid"   # 'sigmoid' | 'ipo' | 'kto'


def dpo_loss(policy_logps, ref_logps, chosen_mask, beta: float, variant: str):
    raise NotImplementedError


def run_dpo(cfg: DPOConfig) -> str:
    raise NotImplementedError
