import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from platform.sim.scaling import chinchilla_loss, predict_mmlu, predict_arena_elo
from platform.sim.cluster import Cluster
from platform.sim.economy import CostBook
from platform.sim.clock import Clock
from platform.sim.events import EventBus
from platform.sim.pretrain_sim import PretrainSpec, simulate_pretrain
from platform.sim.alignment_sim import AlignmentSpec, simulate_alignment
from platform.sim.eval_sim import simulate_safety


def test_chinchilla_monotone_in_compute():
    # bigger model + more tokens → lower predicted loss
    a = chinchilla_loss(1e9, 1e12)
    b = chinchilla_loss(7e9, 2e12)
    c = chinchilla_loss(7e10, 5e12)
    assert a > b > c > 1.69


def test_eval_predictors_in_range_and_monotone():
    sml = predict_mmlu(1e9, 1e12)
    big = predict_mmlu(7e10, 5e12)
    assert 0.25 <= sml <= 0.95
    assert sml < big <= 0.95
    elo_lo = predict_arena_elo(0.4, 0.2, 0.2)
    elo_hi = predict_arena_elo(0.85, 0.85, 0.85)
    assert elo_hi > elo_lo > 1000


def test_cluster_failures_grow_with_size_and_time():
    import random
    rng = random.Random(0)
    small = Cluster("s", n_nodes=8); big = Cluster("b", n_nodes=512)
    fs = sum(small.sample_failures(24, rng) for _ in range(50))
    fb = sum(big.sample_failures(24, rng) for _ in range(50))
    assert fb > fs


def test_pretrain_smoke_advances_clock_and_costs():
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    cluster = Cluster("t", n_nodes=64)
    res = simulate_pretrain(
        PretrainSpec(n_params=1e9, total_tokens=1e10, seq_len=4096,
                     global_batch_tokens=1_000_000, log_every=200),
        cluster, clock, cost, bus, seed=0,
    )
    assert res["total_steps"] == 10_000
    assert clock.days > 0
    assert cost.total > 0


def test_alignment_quality_grows_with_data():
    clock = Clock(); cost = CostBook(); bus = EventBus(out_path=None)
    cluster = Cluster("t", n_nodes=8)
    a = simulate_alignment(AlignmentSpec(sft_examples=10_000, pref_pairs=10_000, rlhf="dpo"),
                           7e9, cluster, clock, cost, bus, seed=0)
    b = simulate_alignment(AlignmentSpec(sft_examples=500_000, pref_pairs=500_000, rlhf="dpo"),
                           7e9, cluster, clock, cost, bus, seed=0)
    assert b["sft_quality"] > a["sft_quality"]
    assert b["rlhf_quality"] > a["rlhf_quality"]


def test_safety_blocks_high_capability_on_high_thresholds():
    bus = EventBus(out_path=None)
    # zero-out thresholds → always BLOCK
    res = simulate_safety({"mmlu": 0.85, "humaneval": 0.85, "gsm8k": 0.9},
                          bus, cbrn_threshold=0.0, cyber_threshold=0.0,
                          persuasion_threshold=0.0, autonomy_threshold=0.0,
                          bias_threshold=0.0, jailbreak_threshold=0.0)
    assert res["verdict"] == "BLOCK" and len(res["failed"]) >= 1


def test_orchestrator_e2e():
    """Smoke-run the full program at toy scale; assert outputs sane."""
    from platform.sim.orchestrator import ProgramSpec, run_program
    from platform.sim.serving_sim import ServingTier
    spec = ProgramSpec(
        name="toy", n_params=1e9, total_tokens=1e10,
        seq_len=4096, global_batch_tokens=1_000_000,
        pretrain_cluster=Cluster("p", n_nodes=8),
        eval_cluster=Cluster("e", n_nodes=1),
        alignment=AlignmentSpec(sft_examples=10_000, pref_pairs=10_000, rlhf="dpo"),
        serving_tiers=[ServingTier("mid", 1e9, "fp8")],
        serving_qps={"mid": 5.0},
        out_dir="out/sim/_test", seed=42,
    )
    res = run_program(spec)
    assert res["clock_days"] > 0
    assert 0.25 <= res["eval"]["mmlu"] <= 1.0
    assert res["safety"]["verdict"] in ("PASS", "BLOCK")
    assert res["cost"].total > 0
