"""Group rollout for GRPO.

GRPO samples a *group* of G completions per prompt and computes a group-relative
advantage — no value network required. This module provides a synchronous,
KV-cache-free sampler that mirrors `platform.alignment.ppo.rollout` in style but
produces G samples per prompt and a response loss-mask suitable for
`platform.alignment._common.compute_logps`.

Production note: real RLVR replaces this synchronous sampler with an *async*
inference engine (vLLM/SGLang) feeding a learner, with periodic weight sync. The
interface (prompts in, ids + response-mask + group index out) is the same.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GroupRollout:
    ids: torch.Tensor          # [N, T]  full padded sequences (prompt + generated)
    resp_mask: torch.Tensor    # [N, T]  1.0 at *generated* token positions (absolute)
    group_index: torch.Tensor  # [N]     which prompt (0..B-1) each row belongs to
    prompt_lens: torch.Tensor  # [N]     prompt length per row
    response_text: list[str]   # [N]     decoded generated text (for verifiers)
    behavior_logp: torch.Tensor | None = None  # [N, T] log-prob of each token
                               # under the *sampling* (behavior) policy, aligned
                               # to ``ids`` positions (logp for token at t lives
                               # at column t). Needed for the GRPO/PPO importance
                               # ratio; None for greedy or legacy rollouts.

    @property
    def n_rows(self) -> int:
        return self.ids.shape[0]


@torch.no_grad()
def sample_group(
    policy,
    prompts_ids: list[list[int]],
    *,
    group_size: int,
    max_new_tokens: int,
    seq_len: int,
    tokenizer,
    temperature: float = 1.0,
    seed: int | None = None,
) -> GroupRollout:
    """Sample ``group_size`` completions for each prompt.

    Returns a :class:`GroupRollout` with ``N = len(prompts_ids) * group_size``
    rows. Sampling is plain autoregressive multinomial (or greedy if
    ``temperature <= 0``), no KV cache — fine for tests and tiny models.
    """
    device = next(policy.parameters()).device
    pad_id, eos_id = tokenizer.pad_id, tokenizer.eos_id
    if seed is not None:
        torch.manual_seed(seed)

    # Expand each prompt group_size times.
    rows: list[list[int]] = []
    group_index: list[int] = []
    for gi, p in enumerate(prompts_ids):
        for _ in range(group_size):
            rows.append(list(p))
            group_index.append(gi)

    B = len(rows)
    prompt_lens = [len(p) for p in rows]
    max_prompt = max(prompt_lens)
    T_total = min(seq_len, max_prompt + max_new_tokens)

    ids = torch.full((B, T_total), pad_id, dtype=torch.long, device=device)
    resp_mask = torch.zeros((B, T_total), dtype=torch.float32, device=device)
    behavior_logp = torch.zeros((B, T_total), dtype=torch.float32, device=device)
    for i, p in enumerate(rows):
        L = min(len(p), T_total)
        ids[i, :L] = torch.tensor(p[:L], dtype=torch.long, device=device)

    cur_lens = torch.tensor(prompt_lens, dtype=torch.long, device=device).clamp(max=T_total)
    done = cur_lens >= T_total

    policy.eval()
    ar = torch.arange(B, device=device)
    for _ in range(max_new_tokens):
        pos = int(cur_lens.max().item())
        if pos >= T_total or bool(done.all().item()):
            break
        logits, _ = policy(ids[:, :pos])
        idx = (cur_lens - 1).clamp(min=0)
        next_logits = logits[ar, idx].float()
        if temperature <= 0:
            tok = next_logits.argmax(dim=-1)
            step_logp = F.log_softmax(next_logits, dim=-1)
        else:
            scaled = next_logits / temperature
            probs = F.softmax(scaled, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
            # Behavior log-prob must reflect the *sampling* distribution, i.e.
            # the temperature-scaled softmax actually used to draw ``tok``.
            step_logp = F.log_softmax(scaled, dim=-1)
        tok_logp = step_logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)  # [B]

        write_pos = cur_lens.clamp(max=T_total - 1)
        for b in range(B):
            if not bool(done[b].item()):
                ids[b, write_pos[b]] = tok[b]
                resp_mask[b, write_pos[b]] = 1.0
                behavior_logp[b, write_pos[b]] = tok_logp[b]
        cur_lens = torch.where(done, cur_lens, cur_lens + 1)
        done = done | (tok == eos_id) | (cur_lens >= T_total)

    # Decode generated text per row (drop specials by relying on byte decode).
    response_text: list[str] = []
    for b in range(B):
        gen_ids = ids[b][resp_mask[b] > 0].tolist()
        response_text.append(tokenizer.decode(gen_ids))

    return GroupRollout(
        ids=ids,
        resp_mask=resp_mask,
        group_index=torch.tensor(group_index, dtype=torch.long, device=device),
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.long, device=device),
        response_text=response_text,
        behavior_logp=behavior_logp,
    )
