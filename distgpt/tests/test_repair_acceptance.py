"""Collective discovery and tiny integrated CUDA FSDP restart acceptance."""
from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh

from distgpt.data.streaming import StreamingLoader
from distgpt.model.config import ModelConfig
from distgpt.model.transformer import GPT
from distgpt.parallel.fsdp import apply_fsdp
from distgpt.training.checkpoint import CheckpointManager
from distgpt.training.optim import build_optimizer
from distgpt.training.trainer import train, validate_config
from test_fsdp_cuda import assert_state, plain
from test_reference_parallel import init_gloo, run_workers
from test_trainer_contracts import make_data, tiny_config


def discovery_worker(rank, world, rendezvous, root):
    init_gloo(rank, world, rendezvous)
    try:
        manager = CheckpointManager(root, "visibility")
        manager.latest = lambda: 3 if rank == 0 else None
        with pytest.raises(RuntimeError, match="visibility differs"):
            manager.discover_latest()
        # Both ranks must have raised, leaving the group usable rather than hung.
        marker = torch.tensor(1)
        dist.all_reduce(marker)
        assert marker.item() == world
    finally:
        dist.destroy_process_group()


def test_discovery_disagreement_fails_collectively(tmp_path):
    run_workers(discovery_worker, tmp_path, str(tmp_path))


def test_validated_fsdp_configuration_and_explicit_gates(tmp_path):
    cfg = tiny_config(tmp_path, "configuration")
    cfg["parallel"].update(dp=2, zero="fsdp")
    assert validate_config(cfg, 2, "cuda") == (2, 1, 1)
    with pytest.raises(NotImplementedError, match="CUDA"):
        validate_config(cfg, 2, "cpu")
    cfg["parallel"]["activation_ckpt"] = "full"
    with pytest.raises(NotImplementedError, match="activation-checkpoint"):
        validate_config(cfg, 2, "cuda")
    cfg["parallel"]["activation_ckpt"] = "none"
    cfg["optim"]["optimizer"] = "muon"
    with pytest.raises(NotImplementedError, match="AdamW"):
        validate_config(cfg, 2, "cuda")


def fsdp_trainer_worker(rank, world, rendezvous, root, dtype):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    base = Path(root)

    def config(name, stop=None):
        cfg = tiny_config(base, name)
        cfg["parallel"].update(dp=world, zero="fsdp")
        cfg["dtype"] = dtype
        cfg["optim"]["total_steps"] = 3
        if stop is not None:
            cfg["train"]["stop_after"] = stop
        return cfg

    def connect(index):
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}-{index}",
                                rank=rank, world_size=world, timeout=timedelta(seconds=45))

    continuous = config("continuous-fsdp")
    interrupted = config("interrupted-fsdp", 1)
    resumed = config("interrupted-fsdp")
    try:
        for i, cfg in enumerate((continuous, interrupted, resumed)):
            connect(i)
            train(cfg)  # real train() deliberately destroys the process group
            assert not dist.is_initialized()
        connect(3)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

        def load(cfg):
            model = GPT(ModelConfig(**cfg["model"])).to(rank)
            model = apply_fsdp(model, mesh, torch.float32)
            opt = cfg["optim"]
            optimizer = build_optimizer(model, opt["lr"], opt["betas"], opt["weight_decay"], fused=True)
            assert not optimizer.state
            loader = StreamingLoader(cfg["data"]["dir"], cfg["data"]["seq_len"],
                                     cfg["train"]["micro_batch"], rank, world, cfg["seed"], f"cuda:{rank}")
            manager = CheckpointManager(cfg["out_dir"], cfg["run_id"],
                                        topology={"dp": world, "tp": 1, "pp": 1})
            runtime = {}
            assert manager.load(model, optimizer, loader, 3, runtime_state=runtime) == 3
            saved = Path(manager.root) / "step_000000003" / f"rank_{rank:05d}.json"
            return {"weights": plain(dict(model.named_parameters())),
                    "optimizer": plain(get_optimizer_state_dict(model, optimizer)),
                    "loader": loader.state_dict(), "runtime": runtime,
                    "rng": json.loads(saved.read_text())["rng"]}

        assert_state(load(continuous), load(resumed))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < 2,
                    reason="requires two CUDA GPUs and NCCL; no CPU substitute")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_fsdp_trainer_interrupted_restart(tmp_path, dtype):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 unavailable")
    make_data(tmp_path)
    run_workers(fsdp_trainer_worker, tmp_path, str(tmp_path), dtype)
