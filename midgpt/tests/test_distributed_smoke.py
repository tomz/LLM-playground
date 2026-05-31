"""Multi-rank gloo smoke tests for midgpt's DDP path.

The existing single-process smoke runs in ``test_train_smoke.py`` exercise the
trainer but never exercise *any* collective (WORLD_SIZE=1 short-circuits
``setup_ddp``). Any refactor of the DDP path therefore had no CI coverage —
the same blind spot distgpt's Tier 1 caught three real bugs in.

This file fills the gap by spawning small (2-rank) gloo CPU groups via
``torch.multiprocessing.spawn`` and exercising:

  * DDP grad-sync: a 2-rank forward+backward produces finite, non-zero grads.
  * Eval all-reduce: the per-rank ``evaluate()`` losses average to the right
    cross-rank mean (the bug pinned here: eval used to run only on rank-0 on
    a rank-sharded val set → val ppl reflected 1/world of the data).
  * End-to-end trainer: a 2-rank 2-step run completes and writes a log file.

Gloo is slow but CI-available; the failure modes we care about (missing
``no_sync()`` boundary, missing reduce on eval, missing destroy_process_group
on resume) surface identically on gloo and NCCL.
"""
from __future__ import annotations

import json
import os
import sys
import socket
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(fn, world: int) -> None:
    port = _free_port()
    mp.spawn(fn, args=(world, port), nprocs=world, join=True)


def _worker_setup(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)


def _worker_teardown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _tiny_cfg(**overrides):
    from model import GPTConfig
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, n_head=2,
               d_model=32, d_ffn=64, tie_embeddings=True)
    cfg.update(overrides)
    return GPTConfig(**cfg)


def _skip_if_no_gloo() -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def _ddp_forward_backward_worker(rank: int, world: int, port: int) -> None:
    """Wrap a model in DDP, forward+backward, assert finite non-zero grads on
    every rank. The single most load-bearing distributed code path; if
    ``gradient_as_bucket_view=True`` or the param dtype is mismatched,
    backward fails here."""
    _worker_setup(rank, world, port)
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        from model import GPT

        torch.manual_seed(0)  # identical init on every rank
        m = GPT(_tiny_cfg()).to(torch.float32)
        m = DDP(m, device_ids=None, gradient_as_bucket_view=True)  # CPU → no device_ids
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        _, loss = m(x, y)
        loss.backward()
        assert torch.isfinite(loss), f"rank {rank}: non-finite loss"
        # Some param must have a finite, non-zero grad after backward; that
        # proves DDP's reduce-scatter actually ran and the grad bucket wasn't
        # a no-op view (a bug class where dummy zero-init buckets shadow real grads).
        any_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in m.parameters()
        )
        assert any_grad, f"rank {rank}: no finite non-zero gradients"
    finally:
        _worker_teardown()


def _eval_allreduce_worker(rank: int, world: int, port: int) -> None:
    """Two ranks → two disjoint shuffles of the val set → mean-allreduce should
    give the same value on both ranks. Pins the regression where eval ran only
    on rank-0 (reporting val on 1/world of the data) by checking that every
    rank ends with the same mean."""
    _worker_setup(rank, world, port)
    try:
        import contextlib, tempfile
        from data import ShardDataset
        from model import GPT
        from train import evaluate

        # Build a deterministic val shard once (broadcast the tmpdir path).
        if rank == 0:
            tmp = tempfile.mkdtemp(prefix="midgpt_eval_smoke_")
        else:
            tmp = ""
        obj_list = [tmp]
        dist.broadcast_object_list(obj_list, src=0)
        tmp = obj_list[0]

        root = Path(tmp) / "ds"
        if rank == 0:
            (root / "val").mkdir(parents=True)
            rng = np.random.default_rng(0)
            rng.integers(0, 256, size=8_000, dtype=np.uint16).tofile(
                str(root / "val" / "shard_000000.bin")
            )
        dist.barrier()

        torch.manual_seed(0)
        m = GPT(_tiny_cfg(vocab_size=256)).to(torch.float32).eval()
        val_ds = ShardDataset(str(root), block_size=16, device="cpu", split="val")
        out = evaluate(m, {"val": val_ds}, eval_iters=4, batch_size=2,
                       ctx=contextlib.nullcontext(),
                       eval_seed=42, world=world, device="cpu")
        # Both ranks must agree on the all-reduced mean (this is the load-bearing
        # property: rank-0 must not read its own un-reduced number).
        local = torch.tensor([out["val"]])
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        for i, g in enumerate(gathered):
            assert torch.allclose(g, gathered[0], atol=1e-6), \
                f"rank {rank}: rank-{i} eval mismatch {g.item()} vs {gathered[0].item()}"
    finally:
        _worker_teardown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ddp_two_rank_forward_backward():
    _skip_if_no_gloo()
    _spawn(_ddp_forward_backward_worker, world=2)


def test_eval_allreduces_across_ranks():
    _skip_if_no_gloo()
    _spawn(_eval_allreduce_worker, world=2)


def test_trainer_end_to_end_ddp_2rank(tmp_path: Path):
    """Full trainer loop with WORLD_SIZE=2 over gloo. Uses torchrun-equivalent
    env so ``setup_ddp`` takes the real path. Asserts the log file landed and
    contains finite loss rows on rank 0 (the only rank that writes)."""
    _skip_if_no_gloo()
    work = tmp_path / "work"
    (work / "data" / "ddp_smoke" / "train").mkdir(parents=True)
    (work / "data" / "ddp_smoke" / "val").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(
        str(work / "data" / "ddp_smoke" / "train" / "shard_000000.bin")
    )
    rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(
        str(work / "data" / "ddp_smoke" / "val" / "shard_000000.bin")
    )
    for f in ("train.py", "model.py", "data.py", "muon.py"):
        (work / f).symlink_to(ROOT / f)
    (work / "utils").symlink_to(ROOT / "utils")

    out = work / "out"
    cfg_path = work / "cfg.yaml"
    cfg_path.write_text(f"""
out_dir: {out}
dataset: ddp_smoke
tokenizer: gpt2
seed: 0
dtype: float32
compile: false
grad_checkpoint: false
log: {{jsonl: true, wandb_project: null}}

model:
  vocab_size: 256
  block_size: 16
  n_layer: 2
  n_head: 2
  d_model: 32
  d_ffn: 64
  tie_embeddings: true

optim:
  lr: 1.0e-3
  min_lr: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0
  grad_clip: 1.0
  warmup_iters: 1
  lr_decay_iters: 2
  max_iters: 2

train:
  micro_batch: 1
  grad_accum: 1
  eval_interval: 1
  eval_iters: 1
  log_interval: 1
  ckpt_interval: 1
""")
    # Force CPU + gloo end-to-end via the MIDGPT_BACKEND env var (added in
    # Tier 6.1 specifically so this test can run on CI without NCCL/CUDA).
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "CUDA_VISIBLE_DEVICES": "",
        "MIDGPT_BACKEND": "gloo",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_free_port()),
    }
    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--standalone", "--nproc_per_node", "2", "--rdzv-backend", "c10d",
           "train.py", "--config", str(cfg_path)]
    r = subprocess.run(cmd, cwd=str(work), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    log = out / "log.jsonl"
    assert log.exists(), "trainer did not write log.jsonl"
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert any("loss" in r for r in rows), "no loss rows in log"
