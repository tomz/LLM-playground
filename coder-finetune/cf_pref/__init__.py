"""Offline preference optimization (DPO / ORPO) for code models.

The cheap, stable middle rung of the post-training ladder:

    SFT (train.py)  →  DPO/ORPO (here)  →  online RLVR/GRPO (cf_rl)

DPO turns a preference pair (prompt, chosen, rejected) into a simple classification
loss against a frozen reference policy — no reward model, no sampling, no rollouts.
ORPO folds the same preference signal *into* SFT with an odds-ratio penalty, so it
needs no reference model at all. Both are far cheaper than GRPO and are the right
default before reaching for online RL. See `dpo_train.py`.
"""
from __future__ import annotations

from . import pairs

__all__ = ["pairs"]
