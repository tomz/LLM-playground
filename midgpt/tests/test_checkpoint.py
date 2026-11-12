"""Full-state restart and publication regressions, with real CPU/Gloo ranks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import save_ckpt
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint


def _equal(a, b) -> None:
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            _equal(left, right)
    else:
        assert a == b


def _bundle(*, scaled: bool = False) -> dict:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.25))
    return {
        "model": model,
        "optimizers": [torch.optim.AdamW(model.parameters(), lr=0.01)],
        "scaler": torch.amp.GradScaler("cpu", enabled=scaled, init_scale=16, growth_interval=1),
        "generator": torch.Generator().manual_seed(7),
        "device": "cpu",
    }


def _update(bundle: dict) -> None:
    model, opt, scaler = bundle["model"], bundle["optimizers"][0], bundle["scaler"]
    opt.zero_grad(set_to_none=True)
    x = torch.rand(3, 4, generator=bundle["generator"])
    loss = model(x).square().mean()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


def test_fresh_optimizer_rng_and_enabled_cpu_scaler_next_update(tmp_path: Path) -> None:
    torch.manual_seed(29)
    original = _bundle(scaled=True)
    _update(original)
    _update(original)
    path = str(tmp_path / "ckpt.pt")
    control = {"rewinder": {"n_rewinds": 2, "scale": 0.25, "cooldown_left": 3}}
    save_training_checkpoint(path, **original, iteration=1, cfg={}, best_val=1.23, control=control)
    _update(original)
    expected_rng = torch.get_rng_state().clone()
    restored = _bundle(scaled=True)  # fresh Adam has no moments yet
    result = load_training_checkpoint(path, **restored, cfg={})
    assert result == {"next_iter": 2, "best_val": 1.23, "control": control}
    _update(restored)
    _equal(original["model"].state_dict(), restored["model"].state_dict())
    _equal(original["optimizers"][0].state_dict(), restored["optimizers"][0].state_dict())
    _equal(original["scaler"].state_dict(), restored["scaler"].state_dict())
    _equal(original["generator"].get_state(), restored["generator"].get_state())
    _equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("corruption,match", [
    ("topology", "topology differs"),
    ("ranks", "per-rank"),
    ("optimizer", "optimizer count"),
    ("schema", "unsupported checkpoint schema"),
    ("configuration", "configuration changed"),
])
def test_reject_incompatible_checkpoint_before_mutation(tmp_path: Path, corruption: str, match: str) -> None:
    bundle = _bundle()
    path = str(tmp_path / "ckpt.pt")
    save_training_checkpoint(path, **bundle, iteration=0, cfg={}, best_val=1.0)
    state = torch.load(path, weights_only=False)
    if corruption == "topology":
        state["topology"]["world_size"] = 2
    elif corruption == "ranks":
        state["rank_states"] = []
    elif corruption == "optimizer":
        state["optims"] = []
    elif corruption == "schema":
        state["checkpoint_schema"] = "future/v99"
    else:
        state["resume_contract"]["dtype"] = "float16"
    torch.save(state, path)
    fresh = _bundle()
    before = {key: value.clone() for key, value in fresh["model"].state_dict().items()}
    with pytest.raises(RuntimeError, match=match):
        load_training_checkpoint(path, **fresh, cfg={})
    _equal(before, fresh["model"].state_dict())


def test_legacy_single_rank_read_and_known_ddp_rejection(tmp_path: Path) -> None:
    original = _bundle()
    _update(original)
    path = str(tmp_path / "ckpt.pt")
    save_training_checkpoint(path, **original, iteration=0, cfg={}, best_val=1.0)
    state = torch.load(path, weights_only=False)
    for key in ("checkpoint_schema", "topology", "rank_states", "resume_contract", "optimizer_types"):
        state.pop(key)
    state["optim"] = state.pop("optims")[0]
    torch.save(state, path)
    fresh = _bundle()
    assert load_training_checkpoint(path, **fresh)["next_iter"] == 1
    _equal(original["optimizers"][0].state_dict(), fresh["optimizers"][0].state_dict())
    state["world_size"] = 2
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="legacy checkpoints support single-rank"):
        load_training_checkpoint(path, **fresh)


def test_atomic_write_failure_keeps_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ckpt.pt"
    save_ckpt(str(path), {"previous": torch.tensor([1, 2])})
    before = path.read_bytes()

    def fail(_state, fh) -> None:
        fh.write(b"partial new checkpoint")
        raise OSError("injected disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected disk failure"):
        save_ckpt(str(path), {"new": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def _cpu_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MIDGPT_BACKEND": "gloo",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def _torchrun(work: Path, script: str, args: list[str], env: dict[str, str]) -> str:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", script, *args]
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


TRACE_RUNNER = '''from pathlib import Path
import json, os
from data import ShardDataset
import train

original = ShardDataset.get_batch

def traced(self, *args, **kwargs):
    x, y = original(self, *args, **kwargs)
    if Path(self.shards[0].filename).parent.name == "train":
        path = Path(os.environ["TRACE_DIR"]) / ("rank" + os.environ["RANK"] + ".jsonl")
        with path.open("a") as fh:
            fh.write(json.dumps({"x": x.tolist(), "y": y.tolist()}) + "\\n")
    return x, y

ShardDataset.get_batch = traced
train.main()
'''


def _work(tmp: Path) -> Path:
    tmp.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for split in ("train", "val"):
        directory = tmp / "data" / "restart" / split
        directory.mkdir(parents=True)
        rng.integers(0, 64, size=8192, dtype=np.uint16).tofile(directory / "tokens.bin")
    (tmp / "traces").mkdir()
    (tmp / "traced_train.py").write_text(TRACE_RUNNER)
    return tmp


def _run_training(work: Path, *, steps: int, optimizer: str, resume: bool = False) -> None:
    cfg = {
        "out_dir": str(work / "out"), "dataset": "restart", "tokenizer": "gpt2",
        "seed": 31, "dtype": "float32", "compile": False, "grad_checkpoint": False,
        "log": {"jsonl": True, "wandb_project": None},
        "model": {"vocab_size": 64, "block_size": 8, "n_layer": 2, "n_head": 2,
                  "d_model": 32, "d_ffn": 64, "dropout": 0.2, "tie_embeddings": True},
        "optim": {"optimizer": optimizer, "muon_lr": 0.02, "lr": 0.001, "min_lr": 0.0001,
                  "betas": [0.9, 0.95], "weight_decay": 0.01, "grad_clip": 1.0,
                  "warmup_iters": 1, "lr_decay_iters": 4, "max_iters": steps},
        "train": {"micro_batch": 2, "grad_accum": 2, "eval_interval": 1,
                  "eval_iters": 2, "log_interval": 1, "ckpt_interval": 1},
        "stability": {"spike_monitor": True, "spike_window": 2, "spike_min_abs_jump": 100},
    }
    cfg_path = work / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    args = ["--config", str(cfg_path)] + (["--resume"] if resume else [])
    _torchrun(work, "traced_train.py", args, {**_cpu_env(), "TRACE_DIR": str(work / "traces")})


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_two_rank_dropout_resume_batch_and_next_update_parity(tmp_path: Path, optimizer: str) -> None:
    continuous = _work(tmp_path / "continuous")
    interrupted = _work(tmp_path / "interrupted")
    _run_training(continuous, steps=3, optimizer=optimizer)
    _run_training(interrupted, steps=2, optimizer=optimizer)
    boundary = torch.load(interrupted / "out" / "ckpt.pt", weights_only=False)
    assert boundary["iter"] == 1
    assert len(boundary["rank_states"]) == 2
    assert not torch.equal(boundary["rank_states"][0]["gen_state"], boundary["rank_states"][1]["gen_state"])
    assert not torch.equal(boundary["rank_states"][0]["rng_state"], boundary["rank_states"][1]["rng_state"])
    _run_training(interrupted, steps=3, optimizer=optimizer, resume=True)
    expected = torch.load(continuous / "out" / "ckpt.pt", weights_only=False)
    actual = torch.load(interrupted / "out" / "ckpt.pt", weights_only=False)
    assert expected["iter"] == actual["iter"] == 2
    for key in ("model", "optims", "rank_states", "best_val", "control", "topology"):
        _equal(expected[key], actual[key])
    traces = []
    for rank in range(2):
        left = (continuous / "traces" / f"rank{rank}.jsonl").read_text().splitlines()
        right = (interrupted / "traces" / f"rank{rank}.jsonl").read_text().splitlines()
        assert left == right
        assert len(left) == 6  # 3 completed updates, 2 micro-batches each
        traces.append([json.loads(line) for line in left])
    assert traces[0] != traces[1]
    assert len(actual["rank_states"][1]["control"]["spike_monitor"]["history"]) == 2
    # The best checkpoint is also a full collective snapshot, not rank zero only.
    best = torch.load(interrupted / "out" / "ckpt_best.pt", weights_only=False)
    assert len(best["rank_states"]) == 2
    assert best["rank_states"][0]["best_val"] == best["rank_states"][1]["best_val"]


def test_two_rank_checkpoint_failures_are_collective(tmp_path: Path) -> None:
    script = tmp_path / "failure_probe.py"
    script.write_text('''from datetime import timedelta
import os
import torch
import torch.distributed as dist
from utils import checkpoint as cp

dist.init_process_group("gloo", timeout=timedelta(seconds=30))
rank = dist.get_rank()
model = torch.nn.Linear(2, 2)
bundle = dict(model=model, optimizers=[torch.optim.AdamW(model.parameters())],
              scaler=torch.amp.GradScaler("cpu", enabled=False),
              generator=torch.Generator().manual_seed(rank), device="cpu")
writer = cp.save_ckpt

def fail(*args, **kwargs):
    raise OSError("injected publication failure")

cp.save_ckpt = fail
try:
    cp.save_training_checkpoint("ckpt.pt", **bundle, iteration=0, cfg={}, best_val=1.0)
    raise AssertionError("save unexpectedly succeeded")
except RuntimeError as exc:
    assert "injected publication failure" in str(exc)
cp.save_ckpt = writer
cp.save_training_checkpoint("ckpt.pt", **bundle, iteration=0, cfg={}, best_val=1.0)
# A missing file on just one rank must make both ranks reject the restore.
try:
    cp.load_training_checkpoint("missing.pt" if rank else "ckpt.pt", **bundle)
    raise AssertionError("rank-local missing file was not collective")
except RuntimeError as exc:
    assert "missing.pt" in str(exc)
if rank == 0:
    state = torch.load("ckpt.pt", weights_only=False)
    for key in ("checkpoint_schema", "topology", "rank_states"):
        state.pop(key)
    torch.save(state, "legacy.pt")
dist.barrier()
try:
    cp.load_training_checkpoint("legacy.pt", **bundle)
    raise AssertionError("legacy DDP resume unexpectedly succeeded")
except RuntimeError as exc:
    assert "legacy checkpoints support single-rank" in str(exc)
print(f"collective-errors-ok rank={rank}")
dist.destroy_process_group()
''')
    output = _torchrun(tmp_path, script.name, [], _cpu_env())
    assert "collective-errors-ok rank=0" in output
    assert "collective-errors-ok rank=1" in output
