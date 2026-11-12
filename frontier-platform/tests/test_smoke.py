"""Smoke tests: import every module, sanity-check pure-Python helpers."""
import importlib
import math
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MODULES = [
    "platform",
    "frontier_platform.data.acquire", "frontier_platform.data.extract", "frontier_platform.data.filter",
    "frontier_platform.data.dedup", "frontier_platform.data.decontaminate", "frontier_platform.data.mix",
    "frontier_platform.data.shard", "frontier_platform.data.loader",
    "frontier_platform.tokenizer.bpe", "frontier_platform.tokenizer.bytes",
    "frontier_platform.model.config", "frontier_platform.model.transformer",
    "frontier_platform.training.optim", "frontier_platform.training.parallel",
    "frontier_platform.training.checkpoint", "frontier_platform.training.stability",
    "frontier_platform.training.trainer",
    "frontier_platform.alignment.sft", "frontier_platform.alignment.reward_model",
    "frontier_platform.alignment.ppo", "frontier_platform.alignment.dpo",
    "frontier_platform.eval.harness", "frontier_platform.eval.arena",
    "frontier_platform.safety.gates", "frontier_platform.safety.classifiers", "frontier_platform.safety.redteam",
    "frontier_platform.serving.engine", "frontier_platform.serving.router", "frontier_platform.serving.torch_engine",
    "frontier_platform.infra.cluster", "frontier_platform.infra.scheduler", "frontier_platform.infra.observability",
]


def test_imports():
    for m in MODULES:
        importlib.import_module(m)


def test_param_count_monotonic():
    from frontier_platform.model.config import ModelConfig
    sizes = [
        ModelConfig(n_layer=24, n_head=16, n_kv_head=8, d_model=2048, d_ffn=5632),
        ModelConfig(n_layer=32, n_head=32, n_kv_head=8, d_model=4096, d_ffn=11008),
        ModelConfig(n_layer=80, n_head=64, n_kv_head=8, d_model=8192, d_ffn=28672),
    ]
    counts = [c.param_count() for c in sizes]
    assert counts == sorted(counts)
    assert 1e9 < counts[0] < 3e9            # ~1B
    assert 5e9 < counts[1] < 1e10           # ~7B
    assert 5e10 < counts[2] < 1e11          # ~70B


def test_cosine_schedule():
    from frontier_platform.training.optim import OptimConfig, cosine_with_warmup
    cfg = OptimConfig(warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
    assert cosine_with_warmup(0, cfg) == 0.0
    assert cosine_with_warmup(100, cfg) == 1.0
    assert math.isclose(cosine_with_warmup(1000, cfg), 0.1, abs_tol=1e-6)
    mid = cosine_with_warmup(550, cfg)
    assert 0.1 < mid < 1.0


def test_spike_monitor():
    from frontier_platform.training.stability import SpikeMonitor
    m = SpikeMonitor(window=50, sigma=4.0)
    for _ in range(50):
        assert m.observe(2.0) is False
    assert m.observe(2.01) is False
    assert m.observe(20.0) is True   # huge spike


def test_elo():
    from frontier_platform.eval.arena import compute_elo
    r = compute_elo([("A", "B", 1.0)] * 20)
    assert r["A"] > r["B"]


def test_exact_dedup():
    from frontier_platform.data.dedup import stream_exact_dedup
    docs = [("1", "hello world"), ("2", "hello   WORLD"), ("3", "different")]
    kept = list(stream_exact_dedup(docs))
    assert [d[0] for d in kept] == ["1", "3"]
