"""End-to-end CPU smoke test: prepare → 1-step train → resume → sample."""
import os, sys, pathlib, subprocess, json
import numpy as np
import pickle

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_dummy_dataset(d: pathlib.Path) -> None:
    """Write tiny train.bin / val.bin + meta.pkl that match nanogpt-edu's format."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        arr = rng.integers(0, 65, size=20_000, dtype=np.uint16)
        arr.tofile(str(d / f"{split}.bin"))
    chars = [chr(i) for i in range(65)]
    with open(d / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 65,
                     "stoi": {c: i for i, c in enumerate(chars)},
                     "itos": chars}, f)


def _config(tmp: pathlib.Path, max_iters: int = 2) -> pathlib.Path:
    cfg_path = tmp / "smoke_cfg.py"
    cfg_path.write_text(f"""
config = dict(
    out_dir={str(tmp / 'out')!r},
    data_dir={str(tmp / 'data')!r},
    n_layer=2, n_head=2, n_kv_head=2, d_model=32, d_ffn=64,
    block_size=16, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=4, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95),
    warmup_iters=1, lr_decay_iters={max_iters}, max_iters={max_iters},
    grad_clip=1.0,
    eval_interval=1, eval_iters=2, log_interval=1, ckpt_interval=1,
    device='cpu', dtype='float32', compile=False, seed=0,
)
""")
    return cfg_path


def test_train_one_step_cpu(tmp_path: pathlib.Path):
    _make_dummy_dataset(tmp_path / "data")
    cfg = _config(tmp_path, max_iters=2)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    assert (out / "ckpt.pt").exists()
    assert (out / "ckpt_best.pt").exists()
    # JSONL logger should have at least 2 records (one per iter, plus eval).
    lines = (out / "train.jsonl").read_text().splitlines()
    assert len(lines) >= 2
    assert all(json.loads(l) for l in lines)


def test_resume_picks_up_iter(tmp_path: pathlib.Path):
    _make_dummy_dataset(tmp_path / "data")
    cfg = _config(tmp_path, max_iters=2)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg)],
        cwd=str(ROOT), env=env, check=True, capture_output=True, text=True, timeout=120,
    )
    # Bump max_iters and resume; should not redo finished iters.
    cfg2 = _config(tmp_path, max_iters=4)
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg2), "--resume"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "resumed from iter" in r.stdout
