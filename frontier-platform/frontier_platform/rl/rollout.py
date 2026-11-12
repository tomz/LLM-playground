"""Token-exact group rollouts for the CPU GRPO reference.

Both samplers admit requests before generation and pack only completed, observed
trajectories. See docs/20-sota-repair-contracts.md for the sampling and snapshot
contract; this is not a distributed actor implementation.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import torch

from ..serving.engine import GenRequest, finished_event, require_int, validated_request
from ..serving.sampling import sample_token, validate_sampling


@dataclass(frozen=True)
class RolloutProvenance:
    """Immutable receipt for a group sampled from one policy snapshot."""

    weight_version: int | None
    temperature: float
    top_p: float
    logprob_kind: str
    response_ids: tuple[tuple[int, ...], ...]
    finish_reasons: tuple[str, ...]
    stop_token_ids: tuple[int | None, ...]
    generation_limits: tuple[int, ...]


@dataclass
class GroupRollout:
    ids: torch.Tensor          # [N, T] full padded sequences (prompt + response)
    resp_mask: torch.Tensor    # [N, T] 1 at generated token positions, including EOS
    group_index: torch.Tensor  # [N] prompt index for each row
    prompt_lens: torch.Tensor  # [N] original prompt length, never cropped
    response_text: list[str]   # decode of exactly the masked response IDs
    behavior_logp: torch.Tensor | None = None  # [N, T], aligned to token positions
    # Defaults preserve old constructors; the learner still requires observed
    # behavior log-probs. Missing probabilities never mean log-probability zero.
    provenance: RolloutProvenance | None = None

    @property
    def n_rows(self) -> int:
        return self.ids.shape[0]


def _snapshot_policy(policy: torch.nn.Module) -> torch.nn.Module:
    """Copy weights/buffers without changing learner device, mode, or gradients.

    Callers must serialize learner mutation with the copy itself. Afterwards,
    learner updates cannot change the actor snapshot.
    """
    snapshot = copy.deepcopy(policy).eval()
    snapshot.requires_grad_(False)
    return snapshot


def _prepare_requests(
    policy, tokenizer, prompts_ids: list[list[int]], *, group_size: int,
    max_new_tokens: int, seq_len: int, temperature: float, top_p: float,
) -> tuple[list[GenRequest], list[int]]:
    """Validate the entire batch before work; reserve each row's context budget."""
    require_int("group_size", group_size, minimum=1)
    require_int("max_new_tokens", max_new_tokens)
    require_int("seq_len", seq_len, minimum=1)
    validate_sampling(temperature, top_p)
    model_cfg = getattr(policy, "cfg", None)
    limit = min(seq_len, getattr(model_cfg, "max_seq_len", seq_len))
    vocab_size = getattr(model_cfg, "vocab_size", getattr(tokenizer, "vocab_size", None))
    require_int("pad_id", tokenizer.pad_id)
    if vocab_size is not None and tokenizer.pad_id >= vocab_size:
        raise ValueError("pad_id is outside the model vocabulary")
    requests, groups = [], []
    for gi, prompt in enumerate(prompts_ids):
        if not isinstance(prompt, (list, tuple)) or not prompt:
            raise ValueError("prompt_ids must be a nonempty token sequence")
        if len(prompt) > limit:
            raise ValueError("prompt exceeds rollout context length; cropping is not allowed")
        request = validated_request(
            GenRequest(
                prompt_ids=list(prompt), max_new_tokens=min(max_new_tokens, limit - len(prompt)),
                temperature=temperature, top_p=top_p, logprobs=1,
                eos_id=getattr(tokenizer, "eos_id", None),
            ),
            max_model_len=limit, vocab_size=vocab_size,
        )
        for _ in range(group_size):
            requests.append(replace(request, prompt_ids=list(request.prompt_ids), stop=list(request.stop)))
            groups.append(gi)
    return requests, groups


def _behavior_logprob(value) -> float:
    """An unknown/nonfinite probability is an error, never a zero fallback."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("behavior log-probability must be present and numeric")
    if not math.isfinite(value) or value > 0:
        raise ValueError("behavior log-probability must be finite and <= 0")
    return float(value)


def _validate_finish(req: GenRequest, generated: list[int], event: dict | None) -> None:
    """Require one successful terminal receipt consistent with the exact tokens."""
    if event is None or event.get("done") is not True:
        raise ValueError("rollout stream ended without a complete event")
    if event.get("logprob_kind") != "behavior":
        raise ValueError("rollout requires behavior log-probabilities, not raw/unknown log-probs")
    usage = event.get("usage")
    expected_usage = {"prompt_tokens": len(req.prompt_ids), "completion_tokens": len(generated)}
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] != value for key, value in expected_usage.items()
    ):
        raise ValueError("rollout completion usage does not match observed tokens")
    last = generated[-1] if generated else None
    if last is not None and last == req.eos_id:
        reason, stop_id = "eos", last
    elif last is not None and last in (req.stop or ()):
        reason, stop_id = "stop", last
    else:
        reason, stop_id = "length", None
        if len(generated) != req.max_new_tokens:
            raise ValueError("length-finished rollout did not consume its generation budget")
    if event.get("finish_reason") != reason or event.get("stop_token_id") != stop_id:
        raise ValueError("rollout finish reason/stop token does not match observed tokens")
    if stop_id is not None and type(event.get("stop_token_id")) is not int:
        raise ValueError("rollout stop token must be an integer")


def _pack_rollouts(
    requests: list[GenRequest], groups: list[int],
    results: list[tuple[list[int], list[float], dict]], *, tokenizer,
    temperature: float, top_p: float, weight_version: int | None = None,
    device: torch.device | str = "cpu",
) -> GroupRollout:
    """Pack without truncation; reward text is decoded once from the same IDs."""
    if len(requests) != len(results) or len(groups) != len(results):
        raise ValueError("rollout results must cover every admitted request")
    width = max((len(req.prompt_ids) + len(gen) for req, (gen, _, _) in zip(requests, results)), default=0)
    ids = torch.full((len(requests), width), tokenizer.pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.float32)
    logp = torch.zeros_like(mask)
    texts: list[str] = []
    for i, (req, (gen, lps, event)) in enumerate(zip(requests, results)):
        if len(gen) > req.max_new_tokens or len(lps) != len(gen):
            raise ValueError("rollout token/log-probability counts violate the admitted budget")
        _validate_finish(req, gen, event)
        lps = [_behavior_logprob(lp) for lp in lps]
        start, end = len(req.prompt_ids), len(req.prompt_ids) + len(gen)
        ids[i, :end] = torch.tensor(req.prompt_ids + gen, dtype=torch.long, device=device)
        mask[i, start:end] = 1.
        logp[i, start:end] = torch.tensor(lps, dtype=torch.float32, device=device)
        texts.append(tokenizer.decode(gen))
    return GroupRollout(
        ids=ids, resp_mask=mask,
        group_index=torch.tensor(groups, dtype=torch.long, device=device),
        prompt_lens=torch.tensor([len(req.prompt_ids) for req in requests], dtype=torch.long, device=device),
        response_text=texts, behavior_logp=logp,
        provenance=RolloutProvenance(
            weight_version=weight_version, temperature=temperature, top_p=top_p,
            logprob_kind="behavior", response_ids=tuple(tuple(gen) for gen, _, _ in results),
            finish_reasons=tuple(event["finish_reason"] for _, _, event in results),
            stop_token_ids=tuple(event["stop_token_id"] for _, _, event in results),
            generation_limits=tuple(req.max_new_tokens for req in requests),
        ),
    )


@torch.no_grad()
def sample_group(
    policy, prompts_ids: list[list[int]], *, group_size: int, max_new_tokens: int,
    seq_len: int, tokenizer, temperature: float = 1.0, seed: int | None = None,
    top_p: float = 1.0,
) -> GroupRollout:
    """Sample full-forward, independent rows from an isolated eval-mode snapshot.

    Each row has its own generation ceiling, even for ragged prompts. Greedy and
    top-p generation are supported, but GRPO deliberately rejects those training
    distributions (see the design contract). Empty batches return shape [0, 0].
    """
    requests, groups = _prepare_requests(
        policy, tokenizer, prompts_ids, group_size=group_size, max_new_tokens=max_new_tokens,
        seq_len=seq_len, temperature=temperature, top_p=top_p,
    )
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    results = []
    if requests:
        snapshot = _snapshot_policy(policy)
        for req in requests:
            gen, logps = [], []
            reason, stop_id = "length", None
            for _ in range(req.max_new_tokens):
                ids = torch.tensor([req.prompt_ids + gen], dtype=torch.long, device=device)
                logits, _ = snapshot(ids)
                token, logp = sample_token(
                    logits[:, -1, :], temperature=temperature, top_p=top_p, generator=generator,
                )
                gen.append(token)
                logps.append(logp)
                if token == req.eos_id:
                    reason, stop_id = "eos", token
                    break
            results.append((gen, logps, finished_event(
                req, len(gen), reason, logprob_kind="behavior", stop_token_id=stop_id,
            )))
    return _pack_rollouts(
        requests, groups, results, tokenizer=tokenizer, temperature=temperature,
        top_p=top_p, device=device,
    )
