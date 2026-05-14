"""Simulated eval harness + arena ELO + safety probes."""
from __future__ import annotations
import random
from .clock import Clock
from .cluster import Cluster, GPU_SPECS
from .economy import CostBook
from .events import EventBus
from .scaling import (
    predict_mmlu, predict_humaneval, predict_gsm8k, predict_arena_elo,
)


def simulate_eval(
    n_params: float,
    n_tokens: float,
    sft_q: float,
    rlhf_q: float,
    eval_cluster: Cluster,
    clock: Clock,
    cost: CostBook,
    bus: EventBus,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    # Full release-eval suite ~ 2h on 64 H100s. Scale by GPU count.
    base_h = 2.0 * 64 / max(1, eval_cluster.total_gpus)
    clock.advance(base_h * 3600)
    cost.charge("eval", f"gpu_{eval_cluster.gpu_type}",
                eval_cluster.total_gpus * base_h * GPU_SPECS[eval_cluster.gpu_type]["price"])

    mmlu = predict_mmlu(n_params, n_tokens) * sft_q * (0.95 + 0.07 * rlhf_q)
    he = predict_humaneval(n_params, n_tokens) * sft_q
    gsm = predict_gsm8k(n_params, n_tokens) * sft_q
    elo = predict_arena_elo(mmlu, he, gsm, sft_q, rlhf_q)
    # add small noise so multiple runs aren't identical
    nz = lambda x: max(0.0, min(1.0, x * (1 + rng.gauss(0, 0.01))))
    out = {
        "mmlu": nz(mmlu),
        "humaneval": nz(he),
        "gsm8k": nz(gsm),
        "arena_elo": elo + rng.gauss(0, 8),
    }
    bus.emit("eval.done", **out, hours=base_h)
    return out


def simulate_safety(
    eval_scores: dict,
    bus: EventBus,
    seed: int = 0,
    cbrn_threshold: float = 0.30,
    cyber_threshold: float = 0.40,
    persuasion_threshold: float = 0.50,
    autonomy_threshold: float = 0.35,
    bias_threshold: float = 0.10,
    jailbreak_threshold: float = 0.20,
) -> dict:
    """Score each safety category; refuse to release if any threshold exceeded."""
    rng = random.Random(seed)
    cap = (eval_scores["mmlu"] + eval_scores["humaneval"] + eval_scores["gsm8k"]) / 3.0
    # uplift scores generally rise with capability
    scores = {
        "cbrn":       max(0, cap * 0.55 + rng.gauss(0, 0.04)),
        "cyber":      max(0, cap * 0.65 + rng.gauss(0, 0.05)),
        "persuasion": max(0, cap * 0.70 + rng.gauss(0, 0.04)),
        "autonomy":   max(0, cap * 0.50 + rng.gauss(0, 0.05)),
        "bias":       max(0, 0.05 + rng.gauss(0, 0.02)),
        "jailbreak":  max(0, 0.30 - 0.20 * cap + rng.gauss(0, 0.04)),
    }
    thresholds = {
        "cbrn": cbrn_threshold, "cyber": cyber_threshold,
        "persuasion": persuasion_threshold, "autonomy": autonomy_threshold,
        "bias": bias_threshold, "jailbreak": jailbreak_threshold,
    }
    failed = [k for k, v in scores.items() if v > thresholds[k]]
    verdict = "PASS" if not failed else "BLOCK"
    bus.emit("safety.done", verdict=verdict, failed=failed, scores=scores, thresholds=thresholds)
    return {"verdict": verdict, "failed": failed, "scores": scores}
