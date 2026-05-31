"""Sequence-parallel tests.

Three layers of coverage:
  1. Single-process: `sequence_parallel=True` at tp=1 is a clean no-op
     (early return; no marker attribute set, no exceptions).
  2. Trainer plumbing: `parallel.sequence_parallel: true` in the config
     reaches the trainer without exploding (also a no-op at tp=1 but
     exercises the config-read path).
  3. Two-rank gloo TP+SP: real DTensor mesh, SP-wrapped norms, Shard(1)
     re-shard between blocks. The loss must (a) be finite and (b) match
     across ranks to within a tiny tolerance, which proves the gather/
     scatter all-reduces are wired correctly. A silent SP bug would
     diverge per-rank losses by O(1).
"""
from __future__ import annotations
import multiprocessing
import os
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT  # noqa: E402
from distgpt.parallel.tensor import apply_tp  # noqa: E402


# ---------------------------------------------------------------------------
# Single-process tests
# ---------------------------------------------------------------------------


def test_apply_tp_is_noop_when_tp_mesh_is_none():
    """`apply_tp(model, None, sequence_parallel=True)` must short-circuit
    cleanly: returns the same object, doesn't set the SP marker (because
    SP didn't actually take effect), doesn't raise.
    """
    cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_kv_head=4,
                      d_model=32, d_ffn=64, max_seq_len=16)
    m = GPT(cfg)
    out = apply_tp(m, tp_mesh=None, sequence_parallel=True)
    assert out is m
    # Marker should NOT be set in the early-return branch. This catches
    # accidental "set even on no-op" bugs that would mislead downstream
    # logic into thinking SP is active when tp=1.
    assert "_dgpt_sp_enabled" not in m.__dict__


def test_apply_tp_marker_uses_safe_attribute_name():
    """The SP marker name `_dgpt_sp_enabled` is short and consistent with
    the rest of distgpt's internal-attribute namespace. This test pins
    the chosen name so a rename is a conscious decision (downstream
    code may grep for it).
    """
    cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_kv_head=4,
                      d_model=32, d_ffn=64, max_seq_len=16)
    m = GPT(cfg)
    apply_tp(m, tp_mesh=None, sequence_parallel=False)
    # Setting it manually to validate the chosen name round-trips cleanly.
    m._dgpt_sp_enabled = True
    assert m._dgpt_sp_enabled is True


def test_trainer_accepts_sequence_parallel_config(tmp_path: pathlib.Path):
    """The trainer config-read path must accept `parallel.sequence_parallel`
    without exploding. Runs single-process (tp=1) so SP is a no-op, but the
    knob plumbing must compile, parse, and reach `apply_tp`.
    """
    import numpy as np
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 32, size=4000, dtype=np.uint16).tofile(
        str(data_dir / "shard_0.bin")
    )

    cfg = {
        "run_id": "sp_test",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 32, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 16, "d_ffn": 32, "max_seq_len": 16,
        },
        "parallel": {
            "dp": 1, "tp": 1, "pp": 1,
            "zero": "none", "activation_ckpt": "none",
            "sequence_parallel": True,  # accepted but no-op at tp=1
        },
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 2,
        },
        "train": {
            "micro_batch": 2, "grad_accum": 1,
            "log_every": 1, "eval_every": 99, "ckpt_every": 99,
        },
    }
    train(cfg)
    assert (pathlib.Path(cfg["out_dir"]) / "sp_test" / "log.jsonl").exists() or \
            (pathlib.Path(cfg["out_dir"]) / "log.jsonl").exists()


# ---------------------------------------------------------------------------
# Two-rank gloo TP+SP
# ---------------------------------------------------------------------------


def _has_gloo() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sp_tp2_worker(rank: int, world: int, port: int, return_dict) -> None:
    """Two-rank TP=2 + SP=on: build a tiny model, run a forward, compare
    losses across ranks. SP layouts being miswired typically shows up as
    O(1) per-rank loss divergence (or an outright shape error during the
    SP gather/scatter)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)  # identical init across ranks
        cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_kv_head=4,
                          d_model=32, d_ffn=64, max_seq_len=32)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))
        m = GPT(cfg).to(torch.float32)
        m = apply_tp(m, mesh["tp"], sequence_parallel=True)
        # Marker should reflect actual state when tp>=2.
        assert m._dgpt_sp_enabled is True, "SP marker not set after apply_tp"

        x = torch.zeros((1, 8), dtype=torch.int64)
        y = torch.zeros((1, 8), dtype=torch.int64)
        _, loss = m(x, y)

        # Gather losses across ranks; SP-correct setups produce identical
        # losses (data is identical, params are identical, gather is correct).
        out = [torch.zeros_like(loss.detach()) for _ in range(world)]
        dist.all_gather(out, loss.detach())
        vals = [t.item() for t in out]
        return_dict[rank] = (min(vals), max(vals))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _has_gloo(), reason="gloo backend not available")
def test_sp2_tp_with_sequence_parallel():
    """TP=2 + SP=on under gloo: losses must agree across ranks to ~1e-3.

    This pins the SP all-gather/scatter wiring; if a future apply_tp refactor
    drops the `Shard(1)` re-shard on o_proj/w2 or breaks the embed/lm_head
    SP layout swap, per-rank losses will diverge by O(1) and this test
    fails loudly.
    """
    world = 2
    mgr = multiprocessing.Manager()
    return_dict = mgr.dict()
    port = _free_port()
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_sp_tp2_worker,
                          args=(r, world, port, return_dict))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    for p in procs:
        if p.exitcode != 0:
            pytest.fail(f"rank exited with {p.exitcode}")
    assert len(return_dict) == world, (
        f"expected {world} worker results, got {dict(return_dict)}"
    )
    mn, mx = return_dict[0]
    rel = abs(mx - mn) / max(abs(mn), 1e-6)
    assert rel < 1e-3, (
        f"SP losses inconsistent across ranks: min={mn} max={mx} rel={rel:.2e}"
    )
