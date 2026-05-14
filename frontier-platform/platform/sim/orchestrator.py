"""Top-level orchestrator. Glues every subsystem into a single end-to-end run."""
from __future__ import annotations
from dataclasses import dataclass

from .clock import Clock
from .cluster import Cluster
from .economy import CostBook
from .events import EventBus
from .data_sim import simulate_data_pipeline
from .tokenizer_sim import simulate_tokenizer_training
from .pretrain_sim import PretrainSpec, simulate_pretrain
from .alignment_sim import AlignmentSpec, simulate_alignment
from .eval_sim import simulate_eval, simulate_safety
from .serving_sim import ServingTier, simulate_serving


@dataclass
class ProgramSpec:
    name: str
    n_params: float
    total_tokens: float
    seq_len: int
    global_batch_tokens: int
    pretrain_cluster: Cluster
    eval_cluster: Cluster
    alignment: AlignmentSpec
    serving_tiers: list[ServingTier]
    serving_qps: dict[str, float]
    out_dir: str = "out/sim"
    seed: int = 0


def run_program(spec: ProgramSpec) -> dict:
    import os
    os.makedirs(spec.out_dir, exist_ok=True)
    clock = Clock()
    cost = CostBook()
    bus = EventBus(out_path=f"{spec.out_dir}/events.jsonl")

    bus.emit("program.start", name=spec.name, n_params=spec.n_params,
             total_tokens=spec.total_tokens,
             pretrain_gpus=spec.pretrain_cluster.total_gpus,
             pretrain_gpu_type=spec.pretrain_cluster.gpu_type)

    data = simulate_data_pipeline(spec.total_tokens, clock, cost, bus)
    tok = simulate_tokenizer_training(sample_gb=200.0, vocab_size=100_352,
                                      clock=clock, cost=cost, bus=bus)
    pre = simulate_pretrain(
        PretrainSpec(
            n_params=spec.n_params, total_tokens=spec.total_tokens,
            seq_len=spec.seq_len, global_batch_tokens=spec.global_batch_tokens,
        ),
        spec.pretrain_cluster, clock, cost, bus, seed=spec.seed,
    )
    align = simulate_alignment(spec.alignment, spec.n_params,
                               spec.pretrain_cluster, clock, cost, bus, seed=spec.seed)
    evals = simulate_eval(spec.n_params, spec.total_tokens,
                          align["sft_quality"], align["rlhf_quality"],
                          spec.eval_cluster, clock, cost, bus, seed=spec.seed)
    safe = simulate_safety(evals, bus, seed=spec.seed)
    serve = simulate_serving(spec.serving_tiers, spec.serving_qps, bus)

    bus.emit("program.done",
             total_dollars=cost.total, total_days=clock.days,
             evals=evals, safety_verdict=safe["verdict"])
    bus.close()

    return {
        "clock_days": clock.days,
        "cost": cost,
        "data": data, "tokenizer": tok, "pretrain": pre,
        "alignment": align, "eval": evals, "safety": safe,
        "serving": serve,
    }
