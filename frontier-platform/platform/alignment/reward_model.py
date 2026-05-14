"""Bradley-Terry reward model: shared trunk + scalar head over the SFT model."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RMConfig:
    base_ckpt: str            # SFT model
    pref_set: str             # (prompt, chosen, rejected)
    epochs: int = 1
    lr: float = 5e-6
    margin: float = 0.0       # optional margin term


def bt_loss(score_chosen, score_rejected, margin: float = 0.0):
    """-log sigmoid(s_c - s_r - margin), reduced over batch."""
    raise NotImplementedError


def train_reward_model(cfg: RMConfig) -> str:
    raise NotImplementedError


def calibrate(rm_ckpt: str, probe_set: str) -> dict:
    """Report KL-vs-score, length bias, refusal bias, demographic bias."""
    raise NotImplementedError
