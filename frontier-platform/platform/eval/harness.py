"""Eval harness. Wraps lm-evaluation-harness for academic benchmarks; adds an
internal arena ELO and a contamination report.
"""
from __future__ import annotations
from dataclasses import dataclass, field

FAST_TASKS = ["hellaswag", "arc_easy", "piqa", "boolq", "openbookqa"]
FULL_TASKS = [
    "mmlu", "gpqa", "bbh", "math", "gsm8k",
    "humaneval", "mbpp", "ifeval", "truthfulqa", "arc_challenge",
]


@dataclass
class EvalRequest:
    ckpt: str
    tasks: list[str] = field(default_factory=lambda: list(FAST_TASKS))
    few_shot: int = 0
    decoding: dict = field(default_factory=lambda: {"temperature": 0.0})
    seed: int = 0


@dataclass
class EvalReport:
    ckpt: str
    metrics: dict[str, float]
    contamination: dict[str, float]
    harness_sha: str
    duration_s: float


class Evaluator:
    def __init__(self, cluster):  # eval cluster, separate from training cluster
        self.cluster = cluster
    def run(self, req: EvalRequest) -> EvalReport:
        raise NotImplementedError
    def run_fast(self, engine, step: int) -> EvalReport:
        """In-cluster, in-process fast eval used during training."""
        raise NotImplementedError
