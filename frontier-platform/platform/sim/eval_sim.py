"""Simulated eval harness + arena ELO + safety probes."""
from __future__ import annotations
import random
from .clock import Clock
from .cluster import Cluster, GPU_SPECS
from .economy import CostBook
from .events import EventBus
from .scaling import (
    predict_mmlu, predict_humaneval, predict_gsm8k, predict_arena_elo,
    predict_swebench, predict_arc_agi2, predict_hle, predict_mmmu,
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
    reasoning_quality: float = 1.0,
    agentic_quality: float = 1.0,
    multimodal: bool = False,
    mm_data_frac: float = 0.0,
    emit: bool = True,
) -> dict:
    rng = random.Random(seed)
    # Full release-eval suite ~ 2h on 64 H100s. Scale by GPU count.
    # When emit=False we're computing a *base-capability probe* (no reasoning
    # lift, no clock/cost charge) to feed the reasoning-RL model.
    base_h = 2.0 * 64 / max(1, eval_cluster.total_gpus)
    if emit:
        clock.advance(base_h * 3600)
        cost.charge("eval", f"gpu_{eval_cluster.gpu_type}",
                    eval_cluster.total_gpus * base_h * GPU_SPECS[eval_cluster.gpu_type]["price"])

    mmlu = predict_mmlu(n_params, n_tokens) * sft_q * (0.95 + 0.07 * rlhf_q)
    he = predict_humaneval(n_params, n_tokens) * sft_q
    gsm = predict_gsm8k(n_params, n_tokens) * sft_q
    # Reasoning-RL lifts code/math/reasoning more than broad knowledge (MMLU).
    he = min(1.0, he * (1.0 + (reasoning_quality - 1.0) * 1.5))
    gsm = min(1.0, gsm * (1.0 + (reasoning_quality - 1.0) * 2.0))
    mmlu = min(1.0, mmlu * (1.0 + (reasoning_quality - 1.0) * 0.5))
    elo = predict_arena_elo(mmlu, he, gsm, sft_q, rlhf_q, reasoning_quality)
    # add small noise so multiple runs aren't identical
    nz = lambda x: max(0.0, min(1.0, x * (1 + rng.gauss(0, 0.01))))
    # --- 2026 frontier suite: post-training- and modality-aware, harder ---
    swebench = predict_swebench(n_params, n_tokens, reasoning_quality=reasoning_quality,
                                agentic_quality=agentic_quality)
    arc = predict_arc_agi2(n_params, n_tokens, reasoning_quality=reasoning_quality)
    hle = predict_hle(n_params, n_tokens, reasoning_quality=reasoning_quality)
    mmmu = predict_mmmu(n_params, n_tokens, multimodal=multimodal,
                        mm_data_frac=mm_data_frac, sft_quality=sft_q)
    out = {
        "mmlu": nz(mmlu),
        "humaneval": nz(he),
        "gsm8k": nz(gsm),
        "arena_elo": elo + rng.gauss(0, 8),
        # 2026 frontier benchmarks
        "swebench_verified": nz(swebench),
        "arc_agi2": nz(arc),
        "hle": nz(hle),
        "mmmu": nz(mmmu),
    }
    if emit:
        bus.emit("eval.done", **out, hours=base_h, reasoning_quality=reasoning_quality,
                 agentic_quality=agentic_quality, multimodal=multimodal)
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
