"""End-to-end CPU smoke test for midgpt: prepare-equivalent + 1 train step."""
import os, sys, pathlib, subprocess, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_dummy(d: pathlib.Path) -> None:
    """Write data/smoke/{train,val}/shard_000000.bin with random uint16 tokens."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        sub = d / "smoke" / split
        sub.mkdir(parents=True, exist_ok=True)
        rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(str(sub / "shard_000000.bin"))


def _config(out_dir: pathlib.Path) -> str:
    return f"""
out_dir: {out_dir}
dataset: smoke
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
  lr: 1.0e-3
  min_lr: 1.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0
  grad_clip: 1.0
  warmup_iters: 1
  lr_decay_iters: 2
  max_iters: 2

train:
  micro_batch: 2
  grad_accum: 1
  eval_interval: 1
  eval_iters: 2
  log_interval: 1
  ckpt_interval: 1
"""


def test_train_one_step_cpu(tmp_path: pathlib.Path):
    # midgpt expects data/<dataset>/<split>/*.bin relative to cwd.
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    _make_dummy(work / "data")
    out = work / "out"
    cfg_path = work / "smoke.yaml"
    cfg_path.write_text(_config(out))

    # Symlink train.py / model.py / data.py / utils so cwd-based imports work.
    for f in ("train.py", "model.py", "data.py"):
        (work / f).symlink_to(ROOT / f)
    (work / "utils").symlink_to(ROOT / "utils")

    env = {**os.environ, "PYTHONPATH": str(ROOT), "WORLD_SIZE": "1"}
    r = subprocess.run(
        [sys.executable, "train.py", "--config", str(cfg_path)],
        cwd=str(work), env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert (out / "ckpt.pt").exists()
    assert (out / "ckpt_best.pt").exists()
    log = (out / "log.jsonl").read_text().splitlines()
    assert len(log) >= 2
    assert all(json.loads(l) for l in log)
