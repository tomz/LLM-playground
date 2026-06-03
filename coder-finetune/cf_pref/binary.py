"""Binary preference examples for KTO-style objectives.

KTO consumes unary feedback — one completion plus a desirable/undesirable label
— rather than DPO's chosen/rejected pairs. The built-in adapter derives a tiny
binary dataset from the existing preference pairs so smoke tests stay offline.
"""
from __future__ import annotations

from datasets import Dataset

from .pairs import BUILTIN_PREFERENCES, _fmt, _prompt


def load_builtin(*, repeat: int = 1, use_chat: bool = False) -> Dataset:
    rows = []
    for _ in range(repeat):
        for prompt, chosen, rejected in BUILTIN_PREFERENCES:
            rows.append({
                "prompt": _prompt(prompt, use_chat),
                "completion": _fmt(chosen, use_chat),
                "label": True,
            })
            rows.append({
                "prompt": _prompt(prompt, use_chat),
                "completion": _fmt(rejected, use_chat),
                "label": False,
            })
    return Dataset.from_list(rows)


def load(cfg: dict) -> Dataset:
    src = cfg.get("source", "builtin")
    use_chat = cfg.get("format", "chat") == "chat"
    if src == "builtin":
        return load_builtin(repeat=int(cfg.get("repeat", 1)), use_chat=use_chat)
    raise ValueError(f"unknown binary preference dataset source: {src}")


__all__ = ["load", "load_builtin"]
