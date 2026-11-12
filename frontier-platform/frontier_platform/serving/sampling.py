"""Reference sampling distribution shared by CPU serving and rollout tests.

Top-p keeps the smallest descending-probability prefix whose cumulative mass
reaches p, including the crossing token. Returned log-probabilities describe the
actual temperature/nucleus distribution, not the untransformed model softmax.
"""
from __future__ import annotations

import math

import torch


def validate_sampling(temperature: float, top_p: float) -> None:
    """Reject non-finite/out-of-domain sampling parameters (zero means greedy)."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite nonnegative number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite nonnegative number")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be in (0, 1]")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def sampling_log_probs(
    logits: torch.Tensor, *, temperature: float = 1.0, top_p: float = 1.0,
) -> torch.Tensor:
    """Normalized log-probs along the last axis; greedy is a point mass.

    Negative infinity is allowed for individually masked logits, but every row
    must contain at least one finite logit. NaN/+inf and scaling overflow fail
    explicitly instead of producing fabricated probabilities.
    """
    validate_sampling(temperature, top_p)
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty vocabulary axis")
    values = logits.float()
    if torch.isnan(values).any() or torch.isposinf(values).any() or not torch.isfinite(values).any(-1).all():
        raise ValueError("logits must contain a finite candidate in each row and no NaN/+inf")
    if temperature == 0:
        result = torch.full_like(values, -torch.inf)
        return result.scatter(-1, values.argmax(-1, keepdim=True), 0.0)
    scaled = values / temperature
    if torch.isposinf(scaled).any() or not torch.isfinite(scaled).any(-1).all():
        raise ValueError("temperature scaling overflowed the logits")
    if top_p < 1:
        ordered, indices = scaled.sort(dim=-1, descending=True, stable=True)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative >= top_p
        # Keep the first token that reaches/crosses p; discard only its suffix.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, -torch.inf)
        scaled = torch.full_like(scaled, -torch.inf).scatter(-1, indices, ordered)
    return torch.log_softmax(scaled, dim=-1)


def sample_token(
    logits: torch.Tensor, *, temperature: float = 1.0, top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[int, float]:
    """Draw from one vocabulary row and return its exact behavior log-prob."""
    logp = sampling_log_probs(logits, temperature=temperature, top_p=top_p)
    if logp.numel() != logp.shape[-1]:
        raise ValueError("sample_token expects one vocabulary row")
    logp = logp.reshape(-1)
    token = int(logp.argmax().item()) if temperature == 0 else int(
        torch.multinomial(logp.exp(), 1, generator=generator).item()
    )
    return token, float(logp[token].item())
