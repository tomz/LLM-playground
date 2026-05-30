"""RLVR / GRPO post-training for code models (verifiable rewards).

The SFT track (`train.py`) teaches format and style from demonstrations. The RL
track here optimizes the model against a *verifier* — does the generated code
pass hidden unit tests? — using TRL's GRPOTrainer. This is the DeepSeek-R1 /
RLVR recipe applied at consumer-GPU scale, and it composes with the LoRA/QLoRA
plumbing already in `train.py` (GRPO trains the same PEFT adapters).
"""
from __future__ import annotations

from .reward import code_unit_test_reward, format_reward, soft_length_penalty

__all__ = ["code_unit_test_reward", "format_reward", "soft_length_penalty"]
