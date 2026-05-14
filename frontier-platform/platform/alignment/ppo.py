"""PPO with KL-to-reference penalty. The classical RLHF setup."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PPOConfig:
    policy_ckpt: str    # init from SFT
    ref_ckpt: str       # frozen SFT for KL reference
    rm_ckpt: str
    rollout_batch: int = 512
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.05      # adaptively controlled
    target_kl: float = 6.0     # nats
    gae_lambda: float = 0.95
    gamma: float = 1.0
    max_new_tokens: int = 1024
    lr: float = 1e-6


def rollout(policy, prompts, cfg: PPOConfig):
    """Sample completions, score with RM, compute KL-to-ref, return trajectories."""
    raise NotImplementedError


def ppo_step(policy, value_head, traj, cfg: PPOConfig):
    """One PPO update over the rollout buffer."""
    raise NotImplementedError


def run_ppo(cfg: PPOConfig) -> str:
    raise NotImplementedError
