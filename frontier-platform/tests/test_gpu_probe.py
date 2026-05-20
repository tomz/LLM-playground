"""Real-GPU integration tests for the simulator. Auto-skip without CUDA."""
import pytest

from platform.sim.gpu_probe import have_cuda, probe_all, register_in_gpu_specs
from platform.sim.real_train import measure_real_throughput


pytestmark = pytest.mark.gpu


def setup_module(_):
    if not have_cuda():
        pytest.skip("CUDA not available")


def test_probe_returns_at_least_one_device():
    probes = probe_all(matmul_size=512, matmul_iters=5)
    assert len(probes) >= 1
    for p in probes:
        assert p.total_mem_gb > 0
        assert "fp32" in p.tflops and p.tflops["fp32"] > 0
        assert p.hbm_bandwidth_gb_s > 0


def test_register_in_gpu_specs_populates_cluster_table():
    from platform.sim.cluster import GPU_SPECS
    probes = probe_all(matmul_size=512, matmul_iters=5)
    keys = register_in_gpu_specs(probes)
    assert keys
    for k in keys:
        assert k in GPU_SPECS
        assert GPU_SPECS[k]["tflops"] > 0
        assert GPU_SPECS[k]["price"] > 0


def test_real_train_produces_throughput():
    r = measure_real_throughput(
        device_index=0,
        n_layer=2, d_model=128, n_head=4, n_kv_head=2,
        seq_len=128, batch=2, vocab_size=512,
        warmup_steps=1, measure_steps=3,
    )
    assert r is not None
    assert r.tokens_per_sec > 0
    assert r.ms_per_step > 0
    assert r.achieved_tflops > 0
    assert r.peak_mem_gb > 0


def test_calibration_changes_simulated_wallclock():
    """Same ProgramSpec, with vs without measured throughput, should give
    different wall-clock days."""
    from platform.sim.orchestrator import ProgramSpec, run_program
    from platform.sim.cluster import Cluster
    from platform.sim.alignment_sim import AlignmentSpec
    from platform.sim.serving_sim import ServingTier

    base = dict(
        name="cal", n_params=1e9, total_tokens=1e10,
        seq_len=4096, global_batch_tokens=1_000_000,
        pretrain_cluster=Cluster("p", n_nodes=8),
        eval_cluster=Cluster("e", n_nodes=1),
        alignment=AlignmentSpec(sft_examples=1000, pref_pairs=1000, rlhf="dpo"),
        serving_tiers=[ServingTier("mid", 1e9, "fp8")],
        serving_qps={"mid": 1.0},
        out_dir="out/sim/_cal_test", seed=0,
    )
    plain = run_program(ProgramSpec(**base))
    # Pretend each GPU is much slower than the H100 default → wall-clock grows.
    slow = run_program(ProgramSpec(
        **base, measured_tflops_per_gpu=2.0,   # ~2 TFLOP/s/GPU → glacial
        measured_source="synthetic-slow",
    ))
    assert slow["clock_days"] > plain["clock_days"]
