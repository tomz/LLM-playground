"""CPU/Gloo numerical references with nonuniform tokens and bounded workers."""
from __future__ import annotations

from datetime import timedelta
import multiprocessing
import os
from pathlib import Path
import sys
import time

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distgpt.model.config import ModelConfig
from distgpt.model.transformer import GPT
from distgpt.parallel.tensor import apply_tp


def run_workers(worker, directory: Path, *args, world: int = 2) -> None:
    """Spawn real ranks and always reap them, including on timeout/failure."""
    ctx = multiprocessing.get_context("spawn")
    rendezvous = str(directory / "rendezvous")
    processes = [ctx.Process(target=worker, args=(r, world, rendezvous, *args))
                 for r in range(world)]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 90
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)


def init_gloo(rank: int, world: int, rendezvous: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=world,
                            timeout=timedelta(seconds=30))


def full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def tp_reference_worker(rank: int, world: int, rendezvous: str, sp: bool) -> None:
    init_gloo(rank, world, rendezvous)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        torch.manual_seed(11)
        cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_kv_head=2,
                          d_model=32, d_ffn=64, max_seq_len=16,
                          tie_embeddings=False, qk_norm=False)
        reference = GPT(cfg)
        parallel = GPT(cfg)
        parallel.load_state_dict(reference.state_dict())
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))
        apply_tp(parallel, mesh, sequence_parallel=sp)
        x = torch.tensor([[1, 5, 9, 2, 7, 3, 11, 6], [8, 2, 4, 12, 3, 6, 1, 15]])
        y = (x * 3 + 2) % cfg.vocab_size
        reference_logits, reference_loss = reference(x, y)
        parallel_logits, parallel_loss = parallel(x, y)
        torch.testing.assert_close(parallel_logits, reference_logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(parallel_loss, reference_loss, rtol=2e-6, atol=2e-6)
        reference_loss.backward()
        parallel_loss.backward()
        for (name, p), (ref_name, expected) in zip(parallel.named_parameters(), reference.named_parameters()):
            assert name == ref_name
            torch.testing.assert_close(full(p.grad), expected.grad, rtol=3e-4, atol=2e-6, msg=name)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, foreach=False)
        parallel_optimizer = torch.optim.AdamW(parallel.parameters(), lr=0.001, foreach=False)
        reference_optimizer.step()
        parallel_optimizer.step()
        for (name, p), (_, expected) in zip(parallel.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(full(p), expected, rtol=3e-4, atol=2e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("sp", [False, True], ids=["tp2", "tp2-sp"])
def test_tp_forward_gradient_optimizer_reference(tmp_path, sp):
    run_workers(tp_reference_worker, tmp_path, sp)
