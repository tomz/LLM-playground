"""Causality, gradients and repeatable equal-token hybrid comparison tests."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid import CausalShortConv, HybridGPT, LocalAttention
from hybrid_ablation import AblationConfig, build_data, evaluate, run_ablation, tensor_memory
from model import GPT, GPTConfig, GQAttention, build_rope_cache


@pytest.fixture(autouse=True)
def one_cpu_thread():
    """Keep tiny tensor tests bounded without leaking thread settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_config(**changes: object) -> GPTConfig:
    return GPTConfig(**(dict(vocab_size=16, block_size=8, n_layer=3, n_head=2, n_kv_head=1,
                             d_model=8, d_ffn=16) | changes))


@pytest.mark.parametrize("kind", ["conv", "local", "global"])
@pytest.mark.parametrize("length", [1, 5, 8])
def test_mixer_shapes_and_gradients(kind: str, length: int) -> None:
    torch.manual_seed(1)
    cfg = model_config(qk_norm=True)
    mixer = {"conv": lambda: CausalShortConv(8, 3), "local": lambda: LocalAttention(cfg, 3),
             "global": lambda: GQAttention(cfg)}[kind]()
    x = torch.randn(2, length, 8, requires_grad=True)
    cos, sin = build_rope_cache(length, cfg.head_dim, cfg.rope_base, x.device, x.dtype)
    y = mixer(x, cos, sin)
    assert y.shape == x.shape
    y.square().sum().backward()
    for value in [x, *mixer.parameters()]:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["conv", "local", "global"])
def test_each_mixer_has_no_future_dependency(kind: str) -> None:
    torch.manual_seed(2)
    cfg = model_config()
    mixer = {"conv": lambda: CausalShortConv(8, 3), "local": lambda: LocalAttention(cfg, 3),
             "global": lambda: GQAttention(cfg)}[kind]().eval()
    x = torch.randn(2, 8, 8, requires_grad=True)
    cos, sin = build_rope_cache(8, 4, 10000, x.device, x.dtype)
    y = mixer(x, cos, sin)
    changed = x.detach().clone()
    changed[:, 4:] = 100 * torch.randn_like(changed[:, 4:])
    torch.testing.assert_close(y[:, :4], mixer(changed, cos, sin)[:, :4], atol=1e-6, rtol=1e-6)
    y[:, 3].sum().backward()
    assert torch.count_nonzero(x.grad[:, 4:]) == 0
    assert x.grad[:, :4].abs().sum() > 0


@pytest.mark.parametrize("kind", ["conv", "local"])
@pytest.mark.parametrize("window", [1, 3, 12])
def test_exact_receptive_field(kind: str, window: int) -> None:
    torch.manual_seed(3)
    mixer = CausalShortConv(8, window) if kind == "conv" else LocalAttention(model_config(), window)
    x = torch.randn(1, 8, 8, requires_grad=True)
    cos, sin = build_rope_cache(8, 4, 10000, x.device, x.dtype)
    mixer(x, cos, sin)[:, 6].sum().backward()
    assert torch.count_nonzero(x.grad[:, 7:]) == 0
    assert torch.count_nonzero(x.grad[:, :max(0, 7 - window)]) == 0
    assert x.grad[:, max(0, 7 - window):7].abs().sum() > 0


def test_full_window_local_matches_global_attention() -> None:
    torch.manual_seed(4)
    cfg = model_config(qk_norm=True)
    local, global_ = LocalAttention(cfg, 8), GQAttention(cfg)
    local.load_state_dict(global_.state_dict())
    x = torch.randn(2, 8, 8)
    cos, sin = build_rope_cache(8, 4, 10000, x.device, x.dtype)
    torch.testing.assert_close(local(x, cos, sin), global_(x, cos, sin), atol=1e-6, rtol=1e-6)


def test_hybrid_end_to_end_causality_and_all_parameters_train() -> None:
    torch.manual_seed(5)
    model = HybridGPT(model_config(), window=3)
    x = torch.randint(0, 16, (2, 8))
    y = torch.randint(0, 16, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 16)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        original, _ = model(x)
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 16
        altered, _ = model(changed)
        prefix, _ = model(x[:, :4])
        torch.testing.assert_close(original[:, :4], altered[:, :4], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original[:, :4], prefix, atol=1e-6, rtol=1e-6)


def test_all_global_recipe_is_exact_baseline_and_default_gpt_unchanged() -> None:
    cfg = model_config()
    torch.manual_seed(8)
    baseline = GPT(cfg)
    torch.manual_seed(8)
    all_global = HybridGPT(cfg, ("global",) * 3)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, all_global.state_dict()[name])
    assert all(type(block.attn) is GQAttention for block in baseline.blocks)
    assert baseline.num_params(False) == all_global.num_params(False)


@pytest.mark.parametrize("changes", [
    {"train_tokens": 0}, {"train_tokens": -1}, {"train_tokens": 100}, {"train_tokens": 32.5},
    {"batch_size": 0}, {"block_size": True}, {"window": 0}, {"kernel_size": -2},
    {"d_model": 30}, {"n_head": 3}, {"eval_sequences": 0}, {"threads": 0},
    {"seeds": ()}, {"seeds": (1, 1)}, {"seeds": (-1,)}, {"data_seed": True},
    {"learning_rate": 0}, {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
    {"latency_repeats": 1}, {"warmup": 0},
])
def test_invalid_budgets_and_shapes_fail_early(changes: dict) -> None:
    with pytest.raises(ValueError):
        AblationConfig(**changes)


@pytest.mark.parametrize("mixers", [("conv",), ("conv", "local", "unknown")])
def test_invalid_recipe(mixers: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        HybridGPT(model_config(), mixers)


def test_text_split_cannot_leak_across_boundary(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"a" * 900 + b"b" * 100)
    cfg = AblationConfig(train_tokens=64, block_size=8, batch_size=2)
    train, heldout, vocab, info = build_data(cfg, path)
    assert vocab == 256
    assert (train == ord("a")).all() and (heldout == ord("b")).all()
    assert info["train_bytes"] == 900 and info["heldout_bytes"] == 100
    path.write_bytes(b"too short")
    with pytest.raises(ValueError, match="partition"):
        build_data(cfg, path)


def test_paired_runs_repeat_quality_and_exact_budget() -> None:
    cfg = AblationConfig(train_tokens=64, block_size=8, batch_size=2, d_model=8, n_head=2, d_ffn=16,
                         eval_sequences=3, seeds=(3, 5), latency_repeats=10, warmup=1)
    original_threads = torch.get_num_threads()
    first, second = run_ablation(cfg), run_ablation(cfg)
    assert torch.get_num_threads() == original_threads
    assert first["data"] == second["data"]
    assert first["data"]["train_sha256"] != first["data"]["heldout_sha256"]
    assert [(row["architecture"], row["seed"]) for row in first["results"]] == [
        ("baseline", 3), ("hybrid", 3), ("hybrid", 5), ("baseline", 5)]
    for left, right in zip(first["results"], second["results"]):
        assert left["val_nll"] == right["val_nll"]
        assert left["trained_tokens"] == 64 and left["train_steps"] == 4
        assert left["eval_tokens"] == 24
        assert len(left["latency"]["samples_ms"]) == 10
        assert left["latency"]["p50_ms"] > 0
        assert left["memory"]["parameter_bytes"] == left["parameters"] * 4
        assert left["memory"]["gradient_bytes"] == left["parameters"] * 4
        assert left["memory"]["optimizer_state_bytes"] > 0
        assert left["memory"]["cpu_peak_rss_bytes"] is None
        assert left["memory"]["accelerator_peak_allocated_bytes"] is None
    json.dumps(first, allow_nan=False)


def test_failed_arm_is_retained_without_survivor_only_summary(monkeypatch) -> None:
    import hybrid_ablation

    original = hybrid_ablation.run_arm

    def fail_one(cfg, architecture, seed, *args):
        if architecture == "hybrid" and seed == 5:
            raise RuntimeError("simulated training failure")
        return original(cfg, architecture, seed, *args)

    monkeypatch.setattr(hybrid_ablation, "run_arm", fail_one)
    cfg = AblationConfig(train_tokens=64, block_size=8, batch_size=2, d_model=8,
                         n_head=2, d_ffn=16, eval_sequences=3, seeds=(3, 5),
                         latency_repeats=10, warmup=1)
    report = run_ablation(cfg)
    assert len(report["results"]) == 4
    failed = [row for row in report["results"] if row["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["trained_tokens"] is None
    assert failed[0]["attempt_wall_s"] >= 0
    assert report["summary"]["hybrid"]["failed_seeds"] == 1
    assert report["summary"]["hybrid"]["val_nll_mean"] is None
    assert report["summary"]["baseline"]["completed_seeds"] == 2
    json.dumps(report, allow_nan=False)


def test_evaluation_tail_batch_is_token_weighted() -> None:
    torch.manual_seed(10)
    model = GPT(model_config())
    heldout = torch.randint(0, 16, (3, 9))
    with torch.no_grad():
        _, expected = model(heldout[:, :-1].contiguous(), heldout[:, 1:].contiguous())
    assert evaluate(model, heldout, batch_size=2) == pytest.approx(float(expected), abs=1e-6)
