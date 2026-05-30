"""Tests for the simulator's frontier extensions: MoE active params, FP8
economics, and the reasoning-RL (RLVR) phase. See platform/sim/."""
from __future__ import annotations

from platform.sim.clock import Clock
from platform.sim.cluster import GPU_SPECS, Cluster
from platform.sim.economy import CostBook
from platform.sim.events import EventBus
from platform.sim.reasoning_rl_sim import ReasoningRLSpec, simulate_reasoning_rl
from platform.sim.scaling import (
    moe_active_params,
    precision_speedup,
    reasoning_rl_quality,
)


# ---------- MoE active params ----------

def test_moe_active_params_less_than_total():
    total = 1.0e12
    active = moe_active_params(total, n_experts=256, top_k=8, shared_experts=1)
    assert active < total
    # dense (n_experts<=1) returns total unchanged
    assert moe_active_params(total, n_experts=0, top_k=2) == total
    assert moe_active_params(total, n_experts=1, top_k=1) == total


def test_moe_more_experts_fewer_active():
    total = 1.0e12
    a = moe_active_params(total, n_experts=8, top_k=2)
    b = moe_active_params(total, n_experts=256, top_k=8)
    # finer-grained routing (256/top-8) activates a smaller fraction than 8/top-2
    assert b < a < total


# ---------- FP8 economics ----------

def test_precision_speedup_ordering():
    assert precision_speedup("bf16") == 1.0
    assert precision_speedup("fp8") > 1.0
    assert precision_speedup("nvfp4") > precision_speedup("fp8")
    assert precision_speedup("unknown") == 1.0


# ---------- new GPUs we don't own ----------

def test_simulated_frontier_gpus_present():
    for sku in ("GB200", "B300"):
        assert sku in GPU_SPECS
        assert GPU_SPECS[sku]["tflops"] > GPU_SPECS["H100"]["tflops"]


# ---------- reasoning-RL quality model ----------

def test_reasoning_rl_quality_saturates_and_gated_by_capability():
    # no rollouts/steps -> no lift
    assert reasoning_rl_quality(0.7, 0, 0) == 1.0
    # more experience -> more lift
    lo = reasoning_rl_quality(0.7, 1.0e5, 1000)
    hi = reasoning_rl_quality(0.7, 4.0e6, 8000)
    assert hi > lo > 1.0
    # stronger base capability -> more lift at the same RL budget
    weak = reasoning_rl_quality(0.3, 4.0e6, 8000)
    strong = reasoning_rl_quality(0.9, 4.0e6, 8000)
    assert strong > weak


# ---------- reasoning-RL phase ----------

def test_reasoning_rl_phase_disabled_is_noop():
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    cluster = Cluster("t", n_nodes=64)
    out = simulate_reasoning_rl(
        ReasoningRLSpec(enabled=False), 7e9, 8e22, 0.6,
        cluster, clock, cost, bus, seed=0,
    )
    assert out["reasoning_quality"] == 1.0
    assert out["rl_compute_flops"] == 0.0
    assert cost.total == 0.0
    assert clock.days == 0.0


def test_reasoning_rl_phase_advances_clock_costs_and_lifts():
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    cluster = Cluster("t", n_nodes=512)
    out = simulate_reasoning_rl(
        ReasoningRLSpec(enabled=True, prompts=400_000, group_size=8,
                        steps=8000, avg_response_tokens=4000),
        356e9, 4.3e25, 0.75,
        cluster, clock, cost, bus, seed=0,
    )
    assert out["reasoning_quality"] > 1.0
    assert out["rl_compute_flops"] > 0
    assert cost.total > 0
    assert clock.days > 0
    # verifier CPU and labels both charged
    assert cost.by_phase.get("reasoning_rl.verifier", 0) > 0
    assert cost.by_phase.get("reasoning_rl.labels", 0) > 0


# ---------- end-to-end: MoE + FP8 + reasoning-RL ----------

def test_orchestrator_moe_fp8_rl_e2e_beats_dense_baseline():
    from platform.sim.alignment_sim import AlignmentSpec
    from platform.sim.orchestrator import ProgramSpec, run_program
    from platform.sim.serving_sim import ServingTier

    common = dict(
        n_params=1e12, total_tokens=2e13, seq_len=8192, global_batch_tokens=32_000_000,
        alignment=AlignmentSpec(sft_examples=10_000, pref_pairs=10_000, rlhf="dpo"),
        serving_tiers=[ServingTier("mid", 1e12, "fp8")],
        serving_qps={"mid": 5.0},
        seed=7,
    )

    # Dense bf16, no reasoning RL.
    dense = run_program(ProgramSpec(
        name="dense", pretrain_cluster=Cluster("p", n_nodes=512),
        eval_cluster=Cluster("e", n_nodes=1),
        out_dir="out/sim/_test_dense", **common,
    ))

    # Sparse MoE + FP8 + reasoning RL on GPUs we don't have.
    moe = run_program(ProgramSpec(
        name="moe", pretrain_cluster=Cluster("p", n_nodes=512, gpu_type="GB200"),
        eval_cluster=Cluster("e", n_nodes=1, gpu_type="GB200"),
        moe_num_experts=256, moe_top_k=8, precision="fp8",
        reasoning_rl=ReasoningRLSpec(enabled=True, prompts=400_000, steps=8000),
        out_dir="out/sim/_test_moe", **common,
    ))

    # MoE active params << dense total params -> much cheaper pretrain wall-clock.
    assert moe["clock_days"] < dense["clock_days"]
    # Reasoning RL lifts arena ELO above the dense, RL-free baseline.
    assert moe["eval"]["arena_elo"] > dense["eval"]["arena_elo"]
    assert moe["reasoning_rl"]["reasoning_quality"] > 1.0


# ---------- MLA serving KV compression ----------

def test_mla_serving_needs_fewer_gpus_than_gqa():
    from platform.sim.serving_sim import ServingTier, simulate_serving
    bus = EventBus(out_path=None)
    qps = {"gqa": 50.0, "mla": 50.0}
    tiers = [
        ServingTier("gqa", 1e12, "fp8", attn_kind="gqa"),
        ServingTier("mla", 1e12, "fp8", attn_kind="mla", kv_compression=4.0),
    ]
    out = simulate_serving(tiers, qps, bus)
    # MLA's KV compression -> higher effective throughput -> fewer replicas/GPUs
    # and a lower $/Mtok at the same QPS.
    assert out["mla"]["gpus"] < out["gqa"]["gpus"]
    assert out["mla"]["cost_per_mtok"] < out["gqa"]["cost_per_mtok"]
    assert out["mla"]["kv_throughput_mult"] > 1.0
    assert out["gqa"]["kv_throughput_mult"] == 1.0


# ---------- agentic RL phase ----------

def test_agentic_rl_quality_builds_on_reasoning():
    from platform.sim.agentic_rl_sim import agentic_rl_quality
    assert agentic_rl_quality(0.7, 1.2, 0, 0) == 1.0
    lo = agentic_rl_quality(0.7, 1.0, 1.0e5, 1000)
    hi = agentic_rl_quality(0.7, 1.3, 1.0e6, 8000)
    assert hi > lo > 1.0
    # higher reasoning_quality amplifies agentic gains at equal budget
    weak = agentic_rl_quality(0.7, 1.0, 1.0e6, 8000)
    strong = agentic_rl_quality(0.7, 1.4, 1.0e6, 8000)
    assert strong > weak


def test_agentic_rl_phase_charges_tools_and_lifts():
    from platform.sim.agentic_rl_sim import AgenticRLSpec, simulate_agentic_rl
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    cluster = Cluster("t", n_nodes=512)
    out = simulate_agentic_rl(
        AgenticRLSpec(enabled=True, tasks=50_000, steps=8000),
        356e9, 0.75, 1.25, cluster, clock, cost, bus, seed=0,
    )
    assert out["agentic_quality"] > 1.0
    assert cost.by_phase.get("agentic_rl.tools", 0) > 0
    assert cost.by_phase.get("agentic_rl.labels", 0) > 0
    assert clock.days > 0


def test_agentic_rl_disabled_is_noop():
    from platform.sim.agentic_rl_sim import AgenticRLSpec, simulate_agentic_rl
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    out = simulate_agentic_rl(
        AgenticRLSpec(enabled=False), 7e9, 0.6, 1.0,
        Cluster("t", n_nodes=8), clock, cost, bus,
    )
    assert out["agentic_quality"] == 1.0 and cost.total == 0.0


# ---------- 2026 eval suite ----------

def test_2026_benchmarks_present_and_post_training_sensitive():
    from platform.sim.scaling import predict_swebench, predict_arc_agi2, predict_mmmu
    # SWE-bench is dominated by agentic post-training.
    base = predict_swebench(3.5e11, 1.5e13, reasoning_quality=1.0, agentic_quality=1.0)
    agentic = predict_swebench(3.5e11, 1.5e13, reasoning_quality=1.3, agentic_quality=1.3)
    assert agentic > base
    # ARC-AGI-2 lifts with reasoning.
    assert predict_arc_agi2(3.5e11, 1.5e13, reasoning_quality=1.4) > \
        predict_arc_agi2(3.5e11, 1.5e13, reasoning_quality=1.0)
    # MMMU is at chance without multimodality, well above with it.
    assert predict_mmmu(3.5e11, 1.5e13, multimodal=False) == 0.22
    assert predict_mmmu(3.5e11, 1.5e13, multimodal=True, mm_data_frac=0.2) > 0.4


def test_eval_emits_2026_keys():
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    from platform.sim.eval_sim import simulate_eval
    out = simulate_eval(3.5e11, 1.5e13, 1.0, 1.0, Cluster("e", n_nodes=8),
                        clock, cost, bus, reasoning_quality=1.2, agentic_quality=1.2,
                        multimodal=True, mm_data_frac=0.15)
    for k in ("swebench_verified", "arc_agi2", "hle", "mmmu"):
        assert k in out and 0.0 <= out[k] <= 1.0


# ---------- multimodal training pricing ----------

def test_multimodal_inflates_training_cost_and_enables_mmmu():
    from platform.sim.alignment_sim import AlignmentSpec
    from platform.sim.orchestrator import ProgramSpec, run_program
    from platform.sim.serving_sim import ServingTier
    common = dict(
        n_params=1e11, total_tokens=5e12, seq_len=8192, global_batch_tokens=16_000_000,
        alignment=AlignmentSpec(sft_examples=5_000, pref_pairs=5_000, rlhf="dpo"),
        serving_tiers=[ServingTier("mid", 1e11, "fp8")], serving_qps={"mid": 2.0}, seed=3,
    )
    text = run_program(ProgramSpec(
        name="text", pretrain_cluster=Cluster("p", n_nodes=256),
        eval_cluster=Cluster("e", n_nodes=1), out_dir="out/sim/_test_text", **common,
    ))
    mm = run_program(ProgramSpec(
        name="mm", pretrain_cluster=Cluster("p", n_nodes=256),
        eval_cluster=Cluster("e", n_nodes=1), multimodal=True, mm_data_frac=0.2,
        out_dir="out/sim/_test_mm", **common,
    ))
    # Image-text tokens inflate pretrain wall-clock/cost.
    assert mm["clock_days"] > text["clock_days"]
    # Text-only is at MMMU chance; multimodal scores above it.
    assert text["eval"]["mmmu"] <= 0.23
    assert mm["eval"]["mmmu"] > 0.3

