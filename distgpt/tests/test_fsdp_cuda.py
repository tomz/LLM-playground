"""Tiny real two-GPU FSDP2 references; no downloads or long training.

Run under an outer 180-second timeout. Each case uses fewer than 1M parameters,
two updates, bounded process groups, and the worker harness always reaps ranks.
"""
from __future__ import annotations

import contextlib
from datetime import timedelta
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distgpt.model.config import ModelConfig
from distgpt.model.transformer import GPT
from distgpt.parallel.fsdp import apply_fsdp
from distgpt.training.checkpoint import CheckpointManager
from distgpt.training.gradients import clip_grad_norm
from distgpt.training.optim import build_optimizer
from test_checkpoint_restart import Cursor
from test_reference_parallel import full, run_workers


def plain(value):
    """Materialize test-only snapshots, never used in the training hot path."""
    if isinstance(value, torch.Tensor):
        return full(value).detach().cpu().clone()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def assert_state(left, right, *, rtol: float = 0, atol: float = 0) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state(left[key], right[key], rtol=rtol, atol=atol)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state(a, b, rtol=rtol, atol=atol)
    else:
        assert left == right


def fsdp_reference_worker(rank: int, world: int, rendezvous: str,
                          root: str, bf16: bool, reshard: bool) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=45))
    try:
        device = torch.device("cuda", rank)
        torch.manual_seed(19)
        cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=4, n_kv_head=2,
                          d_model=32, d_ffn=64, max_seq_len=8, tie_embeddings=True)
        reference = GPT(cfg).to(device)
        assert sum(p.numel() for p in reference.parameters()) < 1_000_000
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

        def make_parallel() -> GPT:
            model = GPT(cfg).to(device)
            model.load_state_dict(reference.state_dict())
            # Keep FP32 parameter/optimizer storage. Autocast alone chooses the
            # compute dtype, exactly as in the trainer and unsharded reference.
            return apply_fsdp(model, mesh, torch.float32, reshard_after_forward=reshard)

        parallel = make_parallel()
        reference_opt = build_optimizer(reference, 0.001, (0.9, 0.95), 0.1, fused=True)
        optimizer = build_optimizer(parallel, 0.001, (0.9, 0.95), 0.1, fused=True)
        loader = Cursor()
        loader.position = rank + 7
        manager = CheckpointManager(root, "fsdp", topology={"dp": world, "tp": 1, "pp": 1})
        models, optimizers = [parallel], [optimizer]
        fresh = None
        rtol, atol = (2e-2, 2e-3) if bf16 else (3e-4, 2e-6)

        def autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()

        for update in range(2):
            reference_opt.zero_grad(set_to_none=True)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            tokens = (torch.arange(world * 2 * 8, device=device).reshape(world, 2, 8) * 7 + 3 + update) % 32
            # Same micro-batch shapes on both sides; reference covers ALL ranks.
            for peer in range(world):
                for micro in range(2):
                    x = tokens[peer, micro:micro + 1]
                    with autocast():
                        _, loss = reference(x, (x * 3 + 5) % 32)
                    (loss / (world * 2)).backward()
            for model in models:
                for micro in range(2):
                    model.set_requires_gradient_sync(micro == 1)
                    x = tokens[rank, micro:micro + 1]
                    with autocast():
                        _, loss = model(x, (x * 3 + 5) % 32)
                    (loss / 2).backward()
                for (name, p), (expected_name, ref) in zip(model.named_parameters(), reference.named_parameters()):
                    assert name == expected_name
                    torch.testing.assert_close(full(p.grad), ref.grad, rtol=rtol, atol=atol, msg=name)
                norm = clip_grad_norm(model, 0.05)
                expected_norm = torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in reference.parameters()]))
                assert norm > 0.05  # exercise real clipping, not just norm reporting
                assert norm == pytest.approx(expected_norm.item(), rel=rtol, abs=atol)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.05)
            for model in models:
                for (name, p), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
                    torch.testing.assert_close(full(p.grad), ref.grad, rtol=rtol, atol=atol, msg=name)
            reference_opt.step()
            for opt in optimizers:
                opt.step()
            for model in models:
                for (name, p), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
                    torch.testing.assert_close(full(p), ref, rtol=rtol, atol=atol, msg=name)
            if update == 0:
                expected = plain(get_optimizer_state_dict(parallel, optimizer))
                manager.save(parallel, optimizer, loader, 1)
                fresh = make_parallel()
                fresh_opt = build_optimizer(fresh, 0.001, (0.9, 0.95), 0.1, fused=True)
                assert not fresh_opt.state
                fresh_loader = Cursor()
                assert manager.load(fresh, fresh_opt, fresh_loader) == 1
                assert fresh_loader.position == loader.position
                assert_state(expected, plain(get_optimizer_state_dict(fresh, fresh_opt)))
                assert_state(plain(dict(parallel.named_parameters())), plain(dict(fresh.named_parameters())))
                models.append(fresh)
                optimizers.append(fresh_opt)
        # Restored FSDP and uninterrupted FSDP must match exactly, not merely
        # within the looser bf16-vs-FP32-gradient-reference tolerance above.
        assert_state(plain(dict(parallel.named_parameters())), plain(dict(fresh.named_parameters())))
        assert_state(plain(get_optimizer_state_dict(parallel, optimizer)),
                     plain(get_optimizer_state_dict(fresh, optimizers[-1])))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < 2,
                    reason="requires two CUDA GPUs and NCCL; CPU is not FSDP evidence")
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-autocast"])
@pytest.mark.parametrize("reshard", [True, False], ids=["reshard", "keep-unsharded"])
def test_fsdp_gradient_clipping_fresh_adam_restart(tmp_path, bf16, reshard):
    if bf16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 unsupported")
    run_workers(fsdp_reference_worker, tmp_path, str(tmp_path), bf16, reshard)
