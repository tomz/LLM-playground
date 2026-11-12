"""Real trainer acceptance: held-out cadence, fixed schedules, and exact restart."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distgpt.data.streaming import StreamingLoader
from distgpt.model.config import ModelConfig
from distgpt.model.transformer import GPT
from distgpt.training.checkpoint import CheckpointManager
from distgpt.training.optim import build_optimizer, cosine_lr
from distgpt.training.trainer import train, validate_config
from test_checkpoint_restart import assert_tree_equal


def make_data(root: Path) -> None:
    """Small, distinct shards; updates cross shard boundaries on both DP ranks."""
    rng = np.random.default_rng(413)
    for split in ("train", "validation"):
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            rng.integers(0, 32, size=83, dtype=np.uint16).tofile(directory / f"{i}.bin")


def tiny_config(root: Path, run_id: str) -> dict:
    """A fixed four-update schedule, never shortened to simulate interruption."""
    return {
        "run_id": run_id, "out_dir": str(root / run_id), "seed": 31,
        "dtype": "float32", "log": {"jsonl": True, "wandb_project": None},
        "data": {"dir": str(root / "train"), "val_dir": str(root / "validation"), "seq_len": 8},
        "model": {"vocab_size": 32, "n_layer": 2, "n_head": 4, "n_kv_head": 2,
                  "d_model": 32, "d_ffn": 64, "max_seq_len": 8, "tie_embeddings": True},
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none", "activation_ckpt": "none"},
        "optim": {"lr": 0.001, "min_lr": 0.0001, "betas": [0.9, 0.95],
                  "weight_decay": 0.1, "grad_clip": 0.05, "warmup_steps": 1, "total_steps": 4},
        "train": {"micro_batch": 2, "grad_accum": 2, "log_every": 1,
                  "eval_every": 1, "eval_batches": 2, "ckpt_every": 1},
    }


def rows(cfg: dict) -> list[dict]:
    return [json.loads(line) for line in (Path(cfg["out_dir"]) / "log.jsonl").read_text().splitlines()]


def checkpoint_state(cfg: dict, step: int) -> dict:
    """Read the actual DCP into a fresh model/AdamW, including saved RNG/cursor."""
    model = GPT(ModelConfig(**cfg["model"]))
    optim = cfg["optim"]
    optimizer = build_optimizer(model, optim["lr"], optim["betas"], optim["weight_decay"], fused=False)
    assert not optimizer.state
    loader = StreamingLoader(cfg["data"]["dir"], seq_len=cfg["data"]["seq_len"],
                             micro_batch=cfg["train"]["micro_batch"], rank=0,
                             world_size=1, seed=cfg["seed"], device="cpu")
    manager = CheckpointManager(cfg["out_dir"], cfg["run_id"])
    runtime = {}
    assert manager.load(model, optimizer, loader, step, runtime_state=runtime) == step
    path = Path(manager.root) / f"step_{step:09d}"
    rank_state = json.loads((path / "rank_00000.json").read_text())
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "loader": loader.state_dict(), "rng": rank_state["rng"], "runtime": runtime}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """These tests are CPU acceptance; GPU coverage has its own bounded workers."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def test_real_train_evaluation_cadence_does_not_change_updates(tmp_path, monkeypatch):
    """Consume all three RNGs in real GPT forwards to expose evaluation leakage."""
    import distgpt.training.trainer as trainer

    class StochasticGPT(GPT):
        def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
            logits, loss = super().forward(tokens, targets)
            if loss is not None:
                # Probe RNG isolation, not a new model/quality claim. Evaluation
                # consumes RNG too; the next training forward must not see it.
                factor = 1 + 0.01 * (random.random() + np.random.rand() + torch.rand((), device=loss.device))
                loss = loss * factor
            return logits, loss

    monkeypatch.setattr(trainer, "GPT", StochasticGPT)
    make_data(tmp_path)
    states, traces = [], []
    for cadence in (0, 1, 2):
        cfg = tiny_config(tmp_path, f"eval_{cadence}")
        cfg["train"]["eval_every"] = cadence
        train(cfg)
        log = rows(cfg)
        assert len([r for r in log if "eval_loss" in r]) == (4 // cadence if cadence else 0)
        traces.append([{k: r[k] for k in ("step", "next_step", "loss", "lr", "grad_norm")}
                       for r in log if "loss" in r])
        state = checkpoint_state(cfg, 4)
        # Evaluation intentionally changes best_val; training state must not.
        state.pop("runtime")
        states.append(state)
    for state, trace in zip(states[1:], traces[1:]):
        assert_tree_equal(states[0], state)
        assert traces[0] == trace


def test_real_train_interrupted_resume_matches_each_next_update(tmp_path):
    make_data(tmp_path)
    continuous = tiny_config(tmp_path, "continuous")
    train(continuous)
    interrupted = tiny_config(tmp_path, "interrupted")
    interrupted["train"]["stop_after"] = 2
    train(interrupted)
    assert_tree_equal(checkpoint_state(continuous, 2), checkpoint_state(interrupted, 2))
    for boundary in (3, 4):
        interrupted["train"]["stop_after"] = boundary
        train(interrupted)
        assert_tree_equal(checkpoint_state(continuous, boundary), checkpoint_state(interrupted, boundary))
    log = rows(interrupted)
    assert [r["next_step"] for r in log if r.get("event") == "init"] == [0, 2, 3]
    updates = [r for r in log if "loss" in r]
    assert [r["step"] for r in updates] == [0, 1, 2, 3]
    assert [r["next_step"] for r in updates] == [1, 2, 3, 4]
    optim = interrupted["optim"]
    assert [r["lr"] for r in updates] == [cosine_lr(i, 1, 4, optim["lr"], optim["min_lr"]) for i in range(4)]
    assert [{k: r[k] for k in ("loss", "lr", "grad_norm")} for r in updates] == [
        {k: r[k] for k in ("loss", "lr", "grad_norm")} for r in rows(continuous) if "loss" in r]


def test_native_resume_rejects_changed_schedule(tmp_path):
    make_data(tmp_path)
    cfg = tiny_config(tmp_path, "schedule")
    cfg["train"]["stop_after"] = 2
    train(cfg)
    cfg["optim"]["total_steps"] = 5
    cfg["train"]["stop_after"] = 3
    with pytest.raises(RuntimeError, match="training restart contract changed"):
        train(cfg)
    assert CheckpointManager(cfg["out_dir"], cfg["run_id"]).latest() == 2


@pytest.mark.parametrize("mode,world,error", [
    ({"pp": 2}, 2, "PP training is disabled"),
    ({"dp": 2, "tp": 2}, 4, "DP×TP training is disabled"),
    ({"dp": 2, "zero": "fsdp"}, 2, "FSDP"),
])
def test_unvalidated_modes_fail_before_allocation(tmp_path, monkeypatch, mode, world, error):
    import distgpt.training.trainer as trainer
    cfg = tiny_config(tmp_path, "unsupported")
    cfg["parallel"].update(mode)
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setattr(trainer, "dist_init", lambda: pytest.fail("initialized an unsupported topology"))
    with pytest.raises(NotImplementedError, match=error):
        train(cfg)
    assert not Path(cfg["out_dir"]).exists()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "not-a-dtype"])
def test_unsupported_cpu_precision_fails_before_allocation(tmp_path, dtype):
    cfg = tiny_config(tmp_path, "precision")
    cfg["dtype"] = dtype
    with pytest.raises(ValueError):
        train(cfg)
    assert not Path(cfg["out_dir"]).exists()


def test_evaluation_requires_distinct_held_out_data(tmp_path):
    cfg = tiny_config(tmp_path, "validation")
    cfg["data"].pop("val_dir")
    with pytest.raises(ValueError, match="evaluation requires"):
        validate_config(cfg, 1, "cpu")
    make_data(tmp_path)
    cfg["data"]["val_dir"] = cfg["data"]["dir"]
    with pytest.raises(RuntimeError, match="validation overlaps training"):
        train(cfg)
