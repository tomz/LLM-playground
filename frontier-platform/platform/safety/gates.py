"""Pre-deployment safety gates. CD pipeline blocks on PASS."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ModelCard:
    ckpt: str
    eval_report_uri: str
    redteam_report_uri: str
    intended_use: str
    known_limits: list[str]


@dataclass
class GateResult:
    passed: bool
    failed_categories: list[str]
    notes: str


CATEGORIES = ["cbrn", "cyber", "persuasion", "autonomy", "bias", "jailbreak"]


def preflight(card: ModelCard) -> GateResult:
    """Aggregate redteam scores; return PASS only if every category under threshold."""
    raise NotImplementedError
