"""Leak-resistant self-play red-team evaluation and release gating."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

AttackGenerator = Callable[[str, int], list[str]]
Defense = Callable[[str], bool]  # True means the attack was safely handled.


def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


@dataclass(frozen=True)
class AttackCase:
    case_id: str
    category: str
    prompt: str
    split: str


@dataclass(frozen=True)
class RedTeamReport:
    train_safe_rate: float
    heldout_safe_rate: float
    heldout_cases: int
    leakage: list[str]

    def passes(self, threshold: float) -> bool:
        return not self.leakage and self.heldout_cases > 0 and self.heldout_safe_rate >= threshold


def build_cases(category: str, prompts: list[str], split: str) -> list[AttackCase]:
    if split not in {"train", "heldout"}:
        raise ValueError("split must be train or heldout")
    return [AttackCase(f"{split}-{index}", category, prompt, split)
            for index, prompt in enumerate(prompts)]


def evaluate_defense(defense: Defense, train: list[AttackCase],
                     heldout: list[AttackCase]) -> RedTeamReport:
    train_hashes = {fingerprint(case.prompt) for case in train}
    leakage = [case.case_id for case in heldout if fingerprint(case.prompt) in train_hashes]
    train_rate = sum(defense(case.prompt) for case in train) / max(1, len(train))
    heldout_rate = sum(defense(case.prompt) for case in heldout) / max(1, len(heldout))
    return RedTeamReport(train_rate, heldout_rate, len(heldout), leakage)


def selfplay_curriculum(seed_prompts: list[str], generator: AttackGenerator,
                        rounds: int = 3) -> list[str]:
    """Generate a deduplicated training curriculum; held-out cases stay external."""
    if rounds < 0:
        raise ValueError("rounds cannot be negative")
    prompts = list(seed_prompts)
    seen = {fingerprint(prompt) for prompt in prompts}
    for round_index in range(rounds):
        for prompt in generator("\n".join(prompts), round_index):
            digest = fingerprint(prompt)
            if digest not in seen:
                prompts.append(prompt)
                seen.add(digest)
    return prompts
