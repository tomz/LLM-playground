"""Automated red-team harness: jailbreak suites + agentic capability probes."""
from __future__ import annotations

SUITES = ["harmbench", "advbench", "multi_turn_social", "cybench", "metr_agent"]


def run_suite(model_endpoint: str, suite: str) -> dict:
    raise NotImplementedError


def run_all(model_endpoint: str) -> dict[str, dict]:
    return {s: run_suite(model_endpoint, s) for s in SUITES}
