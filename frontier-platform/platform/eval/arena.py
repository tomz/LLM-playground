"""Pairwise arena ELO. Blind A/B with judge model + sampled human ratings."""
from __future__ import annotations


def compute_elo(matches: list[tuple[str, str, float]], k: float = 16.0, init: float = 1000.0) -> dict[str, float]:
    """matches: list of (model_a, model_b, score_a) where score_a in {0, 0.5, 1}."""
    ratings: dict[str, float] = {}
    for a, b, sa in matches:
        ra = ratings.setdefault(a, init)
        rb = ratings.setdefault(b, init)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        ratings[a] = ra + k * (sa - ea)
        ratings[b] = rb + k * ((1.0 - sa) - (1.0 - ea))
    return ratings
