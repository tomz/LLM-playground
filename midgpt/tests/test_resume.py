"""End-to-end resumability: train N steps, checkpoint, resume → same final state.

The trainer is careful to round-trip RNG state, optimizer state, scaler state,
and the sampling generator. This test exercises the whole path with both the
AdamW and Muon optimizers and asserts that a single 4-step run produces the
same final loss as a 2-step run that's checkpointed and resumed for 2 more.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys
import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_dummy(d: pathlib.Path) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        sub = d / "resume_smoke" / split
        sub.mkdir(parents=True, exist_ok=True)
        rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(
            str(sub / "shard_000000.bin")
        )


def _config(out_dir: pathlib.Path, max_iters: int, optimizer: str = "adamw",
            ckpt_int: int = 1) -> str:
    return f"""
out_dir: {out_dir}
dataset: resume_smoke
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
  dropout: 0.0
  bias: false
  tie_embeddings: true

optim:
  optimizer: {optimizer}
  muon_lr: 0.02
  muon_momentum: 0.95
  lr: 1.0e-3
  min_lr: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0
  grad_clip: 1.0
  warmup_iters: 1
  lr_decay_iters: 4
  max_iters: {max_iters}

train:
  micro_batch: 2
  grad_accum: 1
  eval_interval: 100
  eval_iters: 2
  log_interval: 1
  ckpt_interval: {ckpt_int}
"""


def _run(work: pathlib.Path, cfg: str, resume: bool) -> int:
    cfg_path = work / "cfg.yaml"
    cfg_path.write_text(cfg)
    cmd = [sys.executable, "train.py", "--config", str(cfg_path)]
    if resume:
        cmd.append("--resume")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "WORLD_SIZE": "1"}
    r = subprocess.run(cmd, cwd=str(work), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r.returncode


def _setup(tmp_path: pathlib.Path) -> pathlib.Path:
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    _make_dummy(work / "data")
    for f in ("train.py", "model.py", "data.py", "muon.py"):
        (work / f).symlink_to(ROOT / f)
    (work / "utils").symlink_to(ROOT / "utils")
    return work


def _final_loss(work: pathlib.Path) -> float:
    log = (work / "out" / "log.jsonl").read_text().splitlines()
    # The last "loss" row (we skip eval rows that only have eval_val).
    losses = [json.loads(l).get("loss") for l in log]
    losses = [v for v in losses if v is not None]
    return losses[-1]


def _final_state(work: pathlib.Path) -> dict:
    sd = torch.load(work / "out" / "ckpt.pt", map_location="cpu", weights_only=False)
    return sd


def _model_state_equal(a: dict, b: dict, tol: float = 1e-5) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        if not torch.allclose(a[k], b[k], atol=tol, rtol=tol):
            return False
    return True


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_resume_continues_run(tmp_path: pathlib.Path, optimizer: str):
    """Train 2 → ckpt → resume to 4 produces the same weights as a single 4-step run.

    This is the load-bearing assertion: every piece of state that survives the
    checkpoint barrier (model, optimizer momentum, RNG, scaler, sampling
    generator) must round-trip exactly for the resumed run to land at the same
    point as the uninterrupted one.
    """
    # Run A: straight through 4 steps.
    work_a = _setup(tmp_path / "a")
    _run(work_a, _config(work_a / "out", max_iters=4, optimizer=optimizer), resume=False)
    state_a = _final_state(work_a)

    # Run B: 2 steps, then resume for 2 more steps (to max_iters=4).
    work_b = _setup(tmp_path / "b")
    # First leg: max_iters=2, ckpt at iter 1 so iter 1 saved, plus the
    # end-of-loop save at iter=max_iters-1=1.
    _run(work_b, _config(work_b / "out", max_iters=2, optimizer=optimizer, ckpt_int=1),
         resume=False)
    # Second leg: resume, run to max_iters=4.
    _run(work_b, _config(work_b / "out", max_iters=4, optimizer=optimizer, ckpt_int=1),
         resume=True)
    state_b = _final_state(work_b)

    # Pointer: iter counter must match.
    assert state_a["iter"] == state_b["iter"], (state_a["iter"], state_b["iter"])
    # Model weights must round-trip. Tolerance lets small float-order noise
    # through (none expected on CPU fp32, but adamw fused=True can reorder).
    assert _model_state_equal(state_a["model"], state_b["model"]), \
        "resumed model weights diverged from uninterrupted run"


def test_resume_back_compat_legacy_single_optim_key(tmp_path: pathlib.Path):
    """Old midgpt checkpoints stored a single ``optim`` key (pre-Muon). The
    resume path falls back to ``optim`` when ``optims`` is absent; verify
    that loading such a legacy checkpoint doesn't crash.
    """
    work = _setup(tmp_path)
    # Run one normal step.
    _run(work, _config(work / "out", max_iters=2, optimizer="adamw"), resume=False)
    # Mutate the checkpoint into the legacy shape: drop "optims", add "optim".
    ckpt = work / "out" / "ckpt.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "optims" in sd  # the new shape
    sd["optim"] = sd["optims"][0]
    del sd["optims"]
    torch.save(sd, ckpt)
    # Resume must not crash on the legacy shape.
    _run(work, _config(work / "out", max_iters=4, optimizer="adamw"), resume=True)
