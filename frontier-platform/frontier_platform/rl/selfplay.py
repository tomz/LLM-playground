"""Deterministic self-play / evolutionary loop for agentic RL experiments.

This is a small closed loop: evaluate candidate policies in a ToolEnv, keep the
top performers, mutate them, and repeat. It models the AlphaEvolve/SPIN-shaped
idea without assuming a concrete LLM backend; production can replace string
mutators with model-generated program or prompt mutations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .agentic import Policy, ToolEnv, Trajectory, rollout_episode
from .verifiers import Verifier

Mutator = Callable[[Policy, random.Random], Policy]


@dataclass
class Candidate:
    name: str
    policy: Policy
    score: float = 0.0
    trajectories: list[Trajectory] | None = None


@dataclass
class Generation:
    index: int
    candidates: list[Candidate]

    @property
    def best(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.score)


def evaluate_candidate(candidate: Candidate, env: ToolEnv, tasks: list[str], verifier: Verifier) -> Candidate:
    trajs = [rollout_episode(env, task, candidate.policy, verifier) for task in tasks]
    score = sum(t.reward for t in trajs) / max(1, len(trajs))
    return Candidate(candidate.name, candidate.policy, score, trajs)


def run_selfplay(
    seeds: list[Candidate],
    env: ToolEnv,
    tasks: list[str],
    verifier: Verifier,
    mutator: Mutator,
    *,
    generations: int = 3,
    top_k: int = 2,
    seed: int = 0,
) -> list[Generation]:
    """Run bounded evolutionary self-play over policy callables."""
    if not seeds:
        raise ValueError("at least one seed candidate is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rng = random.Random(seed)
    population = seeds
    history: list[Generation] = []
    for gen_idx in range(generations):
        evaluated = [evaluate_candidate(c, env, tasks, verifier) for c in population]
        evaluated.sort(key=lambda c: c.score, reverse=True)
        generation = Generation(gen_idx, evaluated)
        history.append(generation)
        parents = evaluated[: min(top_k, len(evaluated))]
        population = list(parents)
        while len(population) < len(seeds):
            parent = parents[len(population) % len(parents)]
            child_idx = len(population)
            population.append(Candidate(
                name=f"{parent.name}_mut{gen_idx}_{child_idx}",
                policy=mutator(parent.policy, rng),
            ))
    return history


def scripted_policy(*utterances: str) -> Policy:
    """Create a deterministic policy for tests and small examples."""
    def _policy(transcript: str) -> str:
        step = transcript.count("<|tool_result|>")
        return utterances[min(step, len(utterances) - 1)]
    return _policy


__all__ = ["Candidate", "Generation", "evaluate_candidate", "run_selfplay", "scripted_policy"]
