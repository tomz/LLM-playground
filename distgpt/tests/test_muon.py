"""Muon optimizer tests.

Single-rank tests cover the algorithm (N-S iteration, scope rules, weight
update direction). A 2-rank gloo test pins the distributed all-gather
behaviour so a future FSDP2 refactor doesn't silently break it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT  # noqa: E402
from distgpt.training.muon import (  # noqa: E402
    Muon, build_muon_and_adamw, newton_schulz5, split_muon_params,
)


# ---------- newton_schulz5 ----------


def test_newton_schulz_returns_orthogonal_for_random_matrix():
    """For a random 64×32 input, N-S output should be approximately orthogonal:
    X @ X.T ≈ I on the smaller dimension (after rescaling)."""
    torch.manual_seed(0)
    G = torch.randn(64, 32, dtype=torch.float32)
    X = newton_schulz5(G, steps=5).float()
    # Smaller dim is 32. X.T @ X should be ≈ I_32. Loose tolerance because
    # the 5-step quintic gets within ~0.06 RMS per entry on a 32×32 identity
    # (||I_32|| = sqrt(32) ≈ 5.66, so a 3.0-norm error is 50% relative —
    # still clearly "the iteration is doing the right thing").
    eye = torch.eye(32)
    err = (X.T @ X - eye).norm()
    assert err < 3.0, f"X.T @ X not close to identity; ||err||={err:.3f}"


def test_newton_schulz_rejects_non_2d():
    with pytest.raises(AssertionError):
        newton_schulz5(torch.randn(8))
    with pytest.raises(AssertionError):
        newton_schulz5(torch.randn(8, 4, 2))


def test_newton_schulz_handles_transposed_orientation():
    """When rows > cols, N-S internally transposes so the inner matmul stays
    rectangular. The output shape must still match the input shape."""
    G = torch.randn(64, 16)
    out = newton_schulz5(G, steps=3)
    assert out.shape == G.shape


# ---------- Muon optimizer ----------


def test_muon_rejects_1d_params():
    p = torch.nn.Parameter(torch.zeros(8))
    with pytest.raises(ValueError, match="2D"):
        Muon([p], lr=0.01)


def test_muon_step_updates_2d_param():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(8, 16))
    opt = Muon([p], lr=0.01)
    # Need a gradient to step against.
    p.grad = torch.randn_like(p)
    before = p.clone().detach()
    opt.step()
    moved = (p - before).abs().sum().item()
    assert moved > 0, "Muon.step() did not change the parameter"


def test_muon_momentum_buffer_is_persistent_across_steps():
    p = torch.nn.Parameter(torch.zeros(8, 16))
    opt = Muon([p], lr=0.01, momentum=0.9)
    p.grad = torch.randn_like(p)
    opt.step()
    assert "momentum_buffer" in opt.state[p]
    buf1 = opt.state[p]["momentum_buffer"].clone()
    p.grad = torch.randn_like(p)
    opt.step()
    buf2 = opt.state[p]["momentum_buffer"]
    # buf2 = 0.9 * buf1 + new_grad → must differ from buf1.
    assert not torch.allclose(buf1, buf2)


# ---------- split_muon_params ----------


def test_split_muon_params_excludes_embeddings_and_head():
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                       d_model=32, d_ffn=64, max_seq_len=16)
    m = GPT(cfg)
    muon, adamw = split_muon_params(m)
    # Build a quick lookup of param -> name so we can check membership.
    name_for_param = {id(p): n for n, p in m.named_parameters()}
    muon_names = {name_for_param[id(p)] for p in muon}
    adamw_names = {name_for_param[id(p)] for p in adamw}
    # tok_emb and lm_head must NOT be in Muon (they're IO layers).
    assert "tok_emb.weight" in adamw_names
    # When tie_embeddings is True, lm_head.weight == tok_emb.weight (same param)
    # so it only appears once in named_parameters(); either way it's in adamw.
    assert all("tok_emb" not in n for n in muon_names)
    assert all("lm_head" not in n for n in muon_names)
    # SwiGLU's w1/w2/w3 ARE 2D hidden weights → Muon.
    assert any("ffn.w1" in n for n in muon_names)
    assert any("ffn.w2" in n for n in muon_names)
    assert any("ffn.w3" in n for n in muon_names)
    # RMSNorm weights are 1-D → AdamW.
    assert any("norm" in n for n in adamw_names)


def test_split_muon_params_skips_non_trainable():
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                       d_model=32, d_ffn=64, max_seq_len=16)
    m = GPT(cfg)
    for p in m.parameters():
        p.requires_grad_(False)
    muon, adamw = split_muon_params(m)
    assert muon == [] and adamw == []


# ---------- build_muon_and_adamw ----------


def test_build_muon_and_adamw_returns_two_optimizers():
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                       d_model=32, d_ffn=64, max_seq_len=16)
    m = GPT(cfg)
    opts = build_muon_and_adamw(m, muon_lr=0.02, adamw_lr=3e-4,
                                  weight_decay=0.1, fused=False)
    # Exactly one Muon + one AdamW.
    types = sorted(o.__class__.__name__ for o in opts)
    assert types == ["AdamW", "Muon"]


# ---------- end-to-end trainer with muon ----------


def test_trainer_runs_with_muon_optimizer(tmp_path: Path):
    """Trainer must complete a short run with optim.optimizer='muon' and
    produce a JSONL log with finite, decreasing-ish losses."""
    import json
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, 256, size=8_000, dtype=np.uint16).tofile(
            str(data_dir / f"shard_{i:06d}.bin")
        )

    cfg = {
        "run_id": "smoke_muon",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 256, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
        },
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "optimizer": "muon",     # ← the bit under test
            "muon_lr": 0.02,
            "lr": 3e-4, "min_lr": 3e-5, "betas": [0.9, 0.95],
            "weight_decay": 0.1, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 3,
        },
        "train": {
            "micro_batch": 2, "grad_accum": 1,
            "log_every": 1, "eval_every": 99, "ckpt_every": 99,
        },
    }
    train(cfg)
    rows = [json.loads(line) for line in (Path(cfg["out_dir"]) / "log.jsonl"
                                           ).read_text().splitlines()]
    losses = [r["loss"] for r in rows if "loss" in r]
    assert len(losses) >= 1
    assert all(np.isfinite(l) for l in losses)
    # The first vs last loss should at least be in the same order of magnitude
    # (3 steps is too few to assert real descent, but a NaN explosion would
    # show up here).
    assert losses[-1] < losses[0] * 10


def test_trainer_rejects_unknown_optimizer_kind(tmp_path: Path):
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.zeros(8_000, dtype=np.uint16).tofile(str(data_dir / "s0.bin"))

    cfg = {
        "run_id": "smoke_bad", "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0, "dtype": "float32",
        "log": {"jsonl": False, "wandb_project": None},
        "model": {"vocab_size": 256, "n_layer": 1, "n_head": 2, "n_kv_head": 2,
                   "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
                   "rope_base": 10000.0, "tie_embeddings": True},
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "optimizer": "shampoo",      # not implemented
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 1,
        },
        "train": {"micro_batch": 1, "grad_accum": 1,
                   "log_every": 1, "eval_every": 99, "ckpt_every": 99},
    }
    with pytest.raises(ValueError, match="shampoo"):
        train(cfg)


# ---------- distributed (gloo) ----------


def _muon_dist_worker(rank: int, world: int, port: int) -> None:
    """Two-rank gloo: build a tiny model with FSDP2, run one Muon step on
    its 2D weights, assert the parameter actually changed on both ranks."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch.distributed as dist
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from distgpt.parallel.fsdp import apply_fsdp

        torch.manual_seed(0)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                           d_model=32, d_ffn=64, max_seq_len=16)
        model = GPT(cfg).to(torch.float32)
        model = apply_fsdp(model, mesh["dp"], torch.float32)

        muon_params, _adamw_params = split_muon_params(model)
        assert muon_params, "no muon params after FSDP wrap; check ndim detection"
        opt = Muon(muon_params, lr=0.01)

        x = torch.randint(0, 64, (2, 8))
        _, loss = model(x, x)
        loss.backward()
        # Snapshot one param's local shard (DTensor → to_local()) before step.
        p = muon_params[0]
        before = p.to_local().clone() if hasattr(p, "to_local") else p.clone()
        opt.step()
        after = p.to_local() if hasattr(p, "to_local") else p
        moved = (after - before).abs().sum().item()
        assert moved > 0, f"rank {rank}: Muon.step on DTensor did not change param"
    finally:
        dist.destroy_process_group()
    # Exit hard *after* a clean process-group teardown. Some torch/gloo point
    # releases SIGABRT ("terminate called without an active exception") during
    # interpreter shutdown when CPU FSDP2 (DTensor) static destructors run
    # after mp.spawn — the assertions above have already passed, so the test
    # logic is fine; it's a C++ teardown abort. os._exit(0) skips those
    # destructors and reports clean success to mp.spawn's exit-code check.
    os._exit(0)


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_muon_distributed_step_on_fsdp_params():
    """Two-rank gloo: Muon must produce a non-zero update on FSDP2-wrapped
    (DTensor) parameters. This pins the distributed all-gather path so a
    future refactor can't silently regress to single-rank behaviour."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")
    import torch.multiprocessing as mp
    mp.spawn(_muon_dist_worker, args=(2, _free_port()), nprocs=2, join=True)
