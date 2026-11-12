"""Checkpoint acceptance tests: fresh optimizers, not already-used destinations."""
from __future__ import annotations

import copy
import pathlib
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distgpt.training.checkpoint import CheckpointManager
from distgpt.training.muon import Muon


class Cursor:
    """Small stateful loader stand-in; real shard manifests are tested separately."""

    def __init__(self) -> None:
        self.position = 0

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]


def assert_tree_equal(left, right) -> None:
    """Compare all optimizer leaves, including counters and parameter groups."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def make_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))


def make_optimizers(model: torch.nn.Module, dual: bool) -> list[torch.optim.Optimizer]:
    if dual:
        matrices = [p for p in model.parameters() if p.ndim == 2]
        biases = [p for p in model.parameters() if p.ndim != 2]
        return [Muon(matrices, lr=0.005), torch.optim.AdamW(biases, lr=0.003)]
    return [torch.optim.AdamW(model.parameters(), lr=0.003, amsgrad=True)]


def update(model: torch.nn.Module, optimizers: list, x: torch.Tensor) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    for optimizer in optimizers:
        optimizer.step()


@pytest.mark.parametrize("dual", [False, True], ids=["adamw", "muon-adamw"])
def test_fresh_optimizer_restores_all_moments_and_next_update(tmp_path, dual):
    torch.manual_seed(7)
    model = make_model()
    optimizers = make_optimizers(model, dual)
    x = torch.randn(5, 4)
    for _ in range(3):
        update(model, optimizers, x)
    saved = [copy.deepcopy(o.state_dict()) for o in optimizers]
    loader = Cursor()
    loader.position = 15
    manager = CheckpointManager(str(tmp_path), "resume")
    manager.save(model, optimizers, loader, step=3)

    fresh = make_model()
    fresh_optimizers = make_optimizers(fresh, dual)
    assert all(not o.state for o in fresh_optimizers)
    fresh_loader = Cursor()
    next_step = manager.load(fresh, fresh_optimizers, fresh_loader)
    for expected, restored in zip(saved, fresh_optimizers):
        assert_tree_equal(expected, restored.state_dict())
    assert next_step == 3  # three completed updates; execute update index 3 next
    assert fresh_loader.position == loader.position

    update(model, optimizers, x + 0.25)
    update(fresh, fresh_optimizers, x + 0.25)
    assert_tree_equal(model.state_dict(), fresh.state_dict())
    for expected, restored in zip(optimizers, fresh_optimizers):
        assert_tree_equal(expected.state_dict(), restored.state_dict())


def test_rng_and_runtime_state_roundtrip(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters())
    update(model, [optimizer], torch.ones(2, 4))
    manager = CheckpointManager(str(tmp_path), "rng")
    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)
    runtime = {"best_val": 1.25, "rewind": {"n_rewinds": 2, "cooldown_left": 17}}
    manager.save(model, optimizer, Cursor(), 1, runtime_state=runtime)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    loaded = {}
    fresh = make_model()
    manager.load(fresh, torch.optim.AdamW(fresh.parameters()), Cursor(),
                 runtime_state=loaded)
    assert loaded == runtime
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
