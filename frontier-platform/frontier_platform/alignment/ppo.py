"""PPO with KL-to-reference penalty. The classical RLHF setup.

Full PPO: clipped policy objective, a value head with GAE(λ) over per-token
KL-shaped rewards, an entropy bonus, and target-KL early stopping. The rollout
re-encodes the prefix each decode step (no KV cache) — algorithmically correct
but O(T²); production swaps in the serving Engine's incremental-cache decode for
throughput. The advantage/return math is unchanged at scale.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer
from ._common import clone_for_reference, forward_hidden
from .reward_model import load_reward_model


@dataclass
class PPOConfig:
    policy_ckpt: str
    rm_ckpt: str
    ref_ckpt: str = ""
    out_dir: str = "out/ppo"
    rollout_batch: int = 4
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    target_kl: float = 6.0
    gae_lambda: float = 0.95
    gamma: float = 1.0
    max_new_tokens: int = 16
    lr: float = 1e-5
    temperature: float = 1.0
    seq_len: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    model_cfg: ModelConfig | None = None


# ---------- value head ----------

class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)  # [B, T]


# ---------- rollout ----------

def _step_logits_and_value(policy: Transformer, value_head: ValueHead, ids: torch.Tensor):
    """Return (logits[B,T,V], values[B,T]) sharing one forward pass."""
    h = forward_hidden(policy, ids)            # [B, T, D]
    logits = policy.lm_head(h)                 # [B, T, V]
    values = value_head(h)                     # [B, T]
    return logits, values


@torch.no_grad()
def rollout(policy, prompts: list[list[int]], cfg: PPOConfig, *, value_head: ValueHead,
            ref_model=None, rm=None, tokenizer=None) -> dict:
    """Sample completions, score with RM, compute KL-shaped rewards + GAE."""
    if tokenizer is None:
        tokenizer = BytesTokenizer()
    if ref_model is None:
        ref_model = clone_for_reference(policy)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    device = next(policy.parameters()).device
    B = len(prompts)
    prompt_lens = [len(p) for p in prompts]
    max_prompt = max(prompt_lens)
    T_total = min(cfg.seq_len, max_prompt + cfg.max_new_tokens)

    # Right-pad prompts; we'll fill in generated tokens.
    ids = torch.full((B, T_total), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        L = min(len(p), T_total)
        ids[i, :L] = torch.tensor(p[:L], dtype=torch.long, device=device)

    gen_logps: list[torch.Tensor] = []  # per-step [B] (policy log-prob of sampled token)
    ref_logps: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    gen_positions: list[int] = []  # absolute position in `ids` where token was generated

    cur_lens = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)

    policy.eval()
    for step in range(cfg.max_new_tokens):
        pos = int(cur_lens.max().item())
        if pos >= T_total:
            break
        # Forward pass: take logits at position cur_lens[b]-1 for each row.
        logits, vals = _step_logits_and_value(policy, value_head, ids[:, :pos])
        ref_logits, _ = _step_logits_and_value(ref_model, value_head, ids[:, :pos])

        idx = (cur_lens - 1).clamp(min=0)
        next_logits = logits[torch.arange(B, device=device), idx].float()  # [B, V]
        next_ref = ref_logits[torch.arange(B, device=device), idx].float()
        next_value = vals[torch.arange(B, device=device), idx]              # [B]

        if cfg.temperature <= 0:
            tok = next_logits.argmax(dim=-1)
        else:
            probs = (next_logits / cfg.temperature).softmax(dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)

        logp_full = F.log_softmax(next_logits, dim=-1)
        ref_logp_full = F.log_softmax(next_ref, dim=-1)
        lp = logp_full.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        rlp = ref_logp_full.gather(-1, tok.unsqueeze(-1)).squeeze(-1)

        # Write tokens into ids for non-done rows
        write_pos = cur_lens.clamp(max=T_total - 1)
        for b in range(B):
            if not done[b]:
                ids[b, write_pos[b]] = tok[b]

        gen_logps.append(lp)
        ref_logps.append(rlp)
        values.append(next_value)
        actions.append(tok)
        gen_positions.append(int(write_pos[0].item()))  # representative

        # Advance lengths only for non-done rows.
        cur_lens = torch.where(done, cur_lens, cur_lens + 1)
        done = done | (tok == eos_id) | (cur_lens >= T_total)
        if bool(done.all().item()):
            break

    # Stack into [B, K] tensors where K = #generated steps
    if not actions:
        raise RuntimeError("rollout produced no tokens (max_new_tokens=0?)")
    actions_t = torch.stack(actions, dim=1)            # [B, K]
    gen_logps_t = torch.stack(gen_logps, dim=1)        # [B, K]
    ref_logps_t = torch.stack(ref_logps, dim=1)        # [B, K]
    values_t = torch.stack(values, dim=1)              # [B, K]

    # Compute RM score on the full completed sequence (prompt+gen).
    rm_scores = torch.zeros(B, dtype=torch.float32, device=device)
    if rm is not None:
        with torch.no_grad():
            rm_scores = rm(ids).float()

    K = actions_t.shape[1]
    # Per-token reward = -kl_coef * (logp - ref_logp); last step adds rm_score.
    kl_per_tok = gen_logps_t - ref_logps_t
    rewards = -cfg.kl_coef * kl_per_tok
    rewards[:, -1] = rewards[:, -1] + rm_scores

    # GAE with bootstrap value 0 at end.
    advs = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=device)
    next_val = torch.zeros(B, device=device)
    for t in reversed(range(K)):
        delta = rewards[:, t] + cfg.gamma * next_val - values_t[:, t]
        gae = delta + cfg.gamma * cfg.gae_lambda * gae
        advs[:, t] = gae
        next_val = values_t[:, t]
    returns = advs + values_t

    # The "states" we need for re-forwarding the policy: the prefix ids just
    # before each generated token. Build [B, K] of position indices and
    # store the full padded ids (length T_total) so ppo_step can re-run forward.
    # Prompt length per row + step index gives the absolute generation index.
    prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)
    gen_idx = prompt_lens_t.unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0)  # [B, K]
    gen_idx = gen_idx.clamp(max=T_total - 1)

    return {
        "ids": ids,                          # [B, T_total]
        "actions": actions_t,                # [B, K]
        "old_logps": gen_logps_t.detach(),   # [B, K]
        "ref_logps": ref_logps_t.detach(),   # [B, K]
        "values": values_t.detach(),         # [B, K]
        "advantages": advs.detach(),         # [B, K]
        "returns": returns.detach(),         # [B, K]
        "gen_idx": gen_idx,                  # [B, K] position in ids of each generated token
        "rm_scores": rm_scores.detach(),
        "kl_mean": float(kl_per_tok.mean().detach()),
    }


def ppo_step(policy, value_head: ValueHead, traj: dict, cfg: PPOConfig, *, optimizer=None) -> dict:
    """One PPO update over the rollout buffer."""
    ids = traj["ids"]
    actions = traj["actions"]
    old_logps = traj["old_logps"]
    advs = traj["advantages"]
    returns = traj["returns"]
    gen_idx = traj["gen_idx"]

    if advs.numel() > 1 and advs.std() > 1e-8:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    policy.train()
    metrics = {}
    for _ in range(cfg.ppo_epochs):
        # Forward once over the full sequences.
        logits, values = _step_logits_and_value(policy, value_head, ids)
        # Gather at positions (gen_idx - 1): the logits that *predicted* each generated token.
        pred_idx = (gen_idx - 1).clamp(min=0)
        B, K = actions.shape
        ar = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, K)
        sel_logits = logits[ar, pred_idx]                  # [B, K, V]
        sel_values = values[ar, pred_idx]                  # [B, K]

        logp_full = F.log_softmax(sel_logits.float(), dim=-1)
        new_logp = logp_full.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, K]

        ratio = (new_logp - old_logps).exp()
        unclipped = ratio * advs
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advs
        loss_pi = -torch.min(unclipped, clipped).mean()
        loss_v = ((sel_values - returns) ** 2).mean()
        # Entropy (categorical) at the selected positions.
        probs = logp_full.exp()
        entropy = -(probs * logp_full).sum(dim=-1).mean()

        loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy

        # Early stop on KL blow-up
        approx_kl = float((old_logps - new_logp).mean().detach())
        metrics = {
            "loss": float(loss.detach()),
            "loss_pi": float(loss_pi.detach()),
            "loss_v": float(loss_v.detach()),
            "entropy": float(entropy.detach()),
            "kl": approx_kl,
        }
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_head.parameters()), 1.0
            )
            optimizer.step()
        if approx_kl > 1.5 * cfg.target_kl:
            break
    return metrics


def _load_policy(cfg: PPOConfig) -> tuple[Transformer, BytesTokenizer]:
    tok = BytesTokenizer()
    state = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
    mcfg = state.get("model_cfg") or cfg.model_cfg
    if mcfg is None:
        raise ValueError("policy_ckpt has no model_cfg; pass cfg.model_cfg")
    m = Transformer(mcfg)
    if "model" in state:
        m.load_state_dict(state["model"])
    return m, tok


def run_ppo(cfg: PPOConfig, prompts: list[str] | None = None) -> str:
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

    rm = load_reward_model(cfg.rm_ckpt) if cfg.rm_ckpt else None
    value_head = ValueHead(policy.cfg.d_model)

    opt = torch.optim.AdamW(
        list(policy.parameters()) + list(value_head.parameters()),
        lr=cfg.lr, weight_decay=0.0,
    )

    if prompts is None:
        prompts = ["Q: hello\nA:", "Q: how are you\nA:", "Q: data\nA:", "Q: model\nA:"]
    prompt_ids = [tok.encode(p) for p in prompts]

    history: list[dict] = []
    n = len(prompt_ids)
    bs = min(cfg.rollout_batch, n)
    for cycle in range(cfg.ppo_epochs):
        batch = prompt_ids[:bs]
        traj = rollout(policy, batch, cfg, value_head=value_head, ref_model=ref, rm=rm, tokenizer=tok)
        metrics = ppo_step(policy, value_head, traj, cfg, optimizer=opt)
        history.append(metrics)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppo.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "model_cfg": policy.cfg,
            "value_head": value_head.state_dict(),
            "history": history,
        },
        out_path,
    )
    return str(out_path)
