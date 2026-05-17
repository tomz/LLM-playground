"""Shared helpers for SFT/RM/DPO/PPO trainers."""
from __future__ import annotations
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------- IO ----------

def load_pref_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert "prompt" in r and "chosen" in r and "rejected" in r, (
                f"pref jsonl row missing keys: {list(r)}"
            )
            out.append(r)
    return out


def load_sft_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert "prompt" in r and "response" in r
            out.append(r)
    return out


# ---------- tokenization / packing ----------

def tokenize_and_pack(
    examples: list[dict],
    tokenizer,
    seq_len: int,
    mask_user_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids[B,T], loss_mask[B,T]).

    Each example must contain ``prompt`` and ``response`` string fields.
    The packed sequence is ``[bos, *prompt, *response, eos]`` right-padded
    with ``pad_id`` to ``seq_len``. ``loss_mask[b,t]==1`` means the token at
    position t should contribute to loss when used as a *target* (i.e. when
    predicting it from positions < t). When ``mask_user_tokens=True``, only
    response tokens and the final EOS are unmasked. Sequences longer than
    ``seq_len`` are truncated (response side is preserved).
    """
    bos, eos, pad = tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id
    B = len(examples)
    ids = torch.full((B, seq_len), pad, dtype=torch.long)
    mask = torch.zeros((B, seq_len), dtype=torch.float32)
    for i, ex in enumerate(examples):
        prompt_ids = tokenizer.encode(ex["prompt"])
        resp_ids = tokenizer.encode(ex["response"])
        seq = [bos] + list(prompt_ids) + list(resp_ids) + [eos]
        # Truncate from the front if too long (keep tail / response).
        if len(seq) > seq_len:
            seq = seq[-seq_len:]
            # Best-effort: recompute response start. After truncation the
            # response region is the last len(resp_ids)+1 tokens.
        seq = seq[:seq_len]
        L = len(seq)
        ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        # Response region within [0, L): the last len(resp_ids)+1 tokens.
        resp_len = min(len(resp_ids) + 1, L)  # +1 for eos
        resp_start = L - resp_len
        if mask_user_tokens:
            mask[i, resp_start:L] = 1.0
        else:
            # Mask everything except bos.
            mask[i, 1:L] = 1.0
    return ids, mask


def compute_logps(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence sum of log P(target | input) under ``model``.

    Shapes: input_ids, target_ids, loss_mask all ``[B, T]`` (already shifted
    by the caller — target_ids[b,t] is the gold next token after input_ids[b,t]).
    Returns ``[B]``.
    """
    logits, _ = model(input_ids)
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
    return (gathered * loss_mask).sum(dim=-1)


def clone_for_reference(model: torch.nn.Module) -> torch.nn.Module:
    """Deep copy + freeze + eval mode. Returns the frozen reference."""
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref


# ---------- low-level forward helpers (hidden states) ----------

def forward_hidden(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the trunk only and return the final-normed hidden states [B,T,D]."""
    x = model.tok_emb(input_ids)
    for blk in model.layers:
        x = blk(x)
    return model.final_norm(x)


def last_nonpad_index(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return ``[B]`` long tensor with the index of the last non-pad token."""
    nonpad = (input_ids != pad_id).long()
    # cumulative: position with last 1
    lengths = nonpad.sum(dim=-1)            # [B]
    return torch.clamp(lengths - 1, min=0)
