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

