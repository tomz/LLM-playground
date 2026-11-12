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
from .reasoning_rl_sim import ReasoningRLSpec, simulate_reasoning_rl
from .agentic_rl_sim import AgenticRLSpec, simulate_agentic_rl
from .eval_sim import simulate_eval, simulate_safety
from .serving_sim import ServingTier, simulate_serving
from .scaling import moe_active_params, compute_flops


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
    # --- sparse MoE: if moe_num_experts>1, training/inference FLOPs are driven
    # by *active* params, not total params. This is the frontier default.
    moe_num_experts: int = 0
    moe_top_k: int = 2
    moe_shared_experts: int = 1
    # --- training numeric format: 'bf16' | 'fp8' | 'nvfp4'. Speeds up pretrain.
    precision: str = "bf16"
    # --- reasoning-RL (RLVR/GRPO) post-training phase
    reasoning_rl: ReasoningRLSpec | None = None
    # --- agentic / tool-use RL post-training phase (long-horizon, sparse reward)
    agentic_rl: AgenticRLSpec | None = None
    # --- multimodality: if True, the eval suite scores MMMU above chance and a
    # fraction of training tokens are image-text (priced via mm_data_frac).
    multimodal: bool = False
    mm_data_frac: float = 0.0
    # Optional throughput override from a real-GPU probe. We measure
    # achieved TFLOP/s on the local device (roughly invariant to model
    # size, unlike raw tok/s) and re-derive seconds_per_step at the
    # simulated model+cluster scale.
    measured_tflops_per_gpu: float | None = None
    measured_source: str | None = None

    @property
    def active_params(self) -> float:
        """Per-token active params (== n_params for dense models)."""
        return moe_active_params(
            self.n_params, self.moe_num_experts, self.moe_top_k, self.moe_shared_experts
        )


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
    measured_sec_per_step = None
    if spec.measured_tflops_per_gpu:
        # Project measured per-GPU TFLOP/s to the whole simulated cluster,
        # then convert to seconds-per-step using the model's FLOPs/step.
        # (FLOP/token ~ 6N; measured TFLOP/s is roughly model-size invariant
        # for compute-bound regimes, which is the regime we care about.)
        cluster_tflops = (
            spec.measured_tflops_per_gpu * spec.pretrain_cluster.total_gpus
        )
        flops_per_step = compute_flops(spec.n_params, spec.global_batch_tokens)
        measured_sec_per_step = flops_per_step / (cluster_tflops * 1e12)
        bus.emit("pretrain.calibrated",
                 per_gpu_tflops=spec.measured_tflops_per_gpu,
                 cluster_tflops=cluster_tflops,
                 flops_per_step=flops_per_step,
                 measured_seconds_per_step=measured_sec_per_step,
                 source=spec.measured_source or "measured")
    # Multimodality inflates effective training tokens: image-text examples carry
    # many vision (patch) tokens per image. Model this as an effective-token
    # multiplier on the image-text fraction (docs/16 §"Cost / simulator").
    mm_token_mult = 1.0 + spec.mm_data_frac * 3.0  # vision tokens ~ heavy
    effective_tokens = spec.total_tokens * mm_token_mult
    if spec.multimodal:
        bus.emit("pretrain.multimodal",
                 mm_data_frac=spec.mm_data_frac, token_multiplier=mm_token_mult,
                 effective_tokens=effective_tokens)
    pre = simulate_pretrain(
        PretrainSpec(
            n_params=spec.n_params, total_tokens=effective_tokens,
            seq_len=spec.seq_len, global_batch_tokens=spec.global_batch_tokens,
            active_params=spec.active_params, precision=spec.precision,
            measured_seconds_per_step=measured_sec_per_step,
            measured_source=spec.measured_source,
        ),
        spec.pretrain_cluster, clock, cost, bus, seed=spec.seed,
    )
    align = simulate_alignment(spec.alignment, spec.active_params,
                               spec.pretrain_cluster, clock, cost, bus, seed=spec.seed)

    # Base capability (pretraining-only) to feed the reasoning-RL lift model.
    base_eval = simulate_eval(spec.n_params, spec.total_tokens,
                              align["sft_quality"], align["rlhf_quality"],
                              spec.eval_cluster, clock, cost, bus, seed=spec.seed,
                              emit=False)
    base_cap = (base_eval["mmlu"] + base_eval["humaneval"] + base_eval["gsm8k"]) / 3.0
    pretrain_flops = compute_flops(spec.active_params, spec.total_tokens)
    rl_spec = spec.reasoning_rl or ReasoningRLSpec(enabled=False)
    rl = simulate_reasoning_rl(
        rl_spec, spec.active_params, pretrain_flops, base_cap,
        spec.pretrain_cluster, clock, cost, bus, seed=spec.seed,
    )

    # Agentic / tool-use RL builds on the reasoning model (long-horizon, sparse).
    ag_spec = spec.agentic_rl or AgenticRLSpec(enabled=False)
    ag = simulate_agentic_rl(
        ag_spec, spec.active_params, base_cap, rl["reasoning_quality"],
        spec.pretrain_cluster, clock, cost, bus, seed=spec.seed,
    )

    evals = simulate_eval(spec.n_params, spec.total_tokens,
                          align["sft_quality"], align["rlhf_quality"],
                          spec.eval_cluster, clock, cost, bus, seed=spec.seed,
                          reasoning_quality=rl["reasoning_quality"],
                          agentic_quality=ag["agentic_quality"],
                          multimodal=spec.multimodal, mm_data_frac=spec.mm_data_frac)
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
        "alignment": align, "reasoning_rl": rl, "agentic_rl": ag,
        "eval": evals, "safety": safe, "serving": serve,
    }
