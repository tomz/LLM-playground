"""End-to-end CPU smoke test for distgpt.train (single-process, no NCCL)."""
import sys, pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_train_one_step_single_process(tmp_path: pathlib.Path):
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, 256, size=8_000, dtype=np.uint16).tofile(
            str(data_dir / f"shard_{i:06d}.bin")
        )

    cfg = {
        "run_id": "smoke",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 256, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
        },
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none", "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 2,
        },
        "train": {
            "micro_batch": 2, "grad_accum": 1,
            "log_every": 1, "eval_every": 1, "ckpt_every": 1,
        },
    }
    # In single-process mode dist isn't initialized; the trainer should be a
    # no-op on the dist side.
    train(cfg)

    log = pathlib.Path(cfg["out_dir"]) / "log.jsonl"
    assert log.exists()
    rows = log.read_text().splitlines()
    assert len(rows) >= 1
    for line in rows:
        json.loads(line)


def test_streaming_loader_no_overlap(tmp_path: pathlib.Path):
    """Two consecutive batches must not share any input token."""
    from distgpt.data.streaming import StreamingLoader
    arr = np.arange(10_000, dtype=np.uint16)
    arr.tofile(str(tmp_path / "shard_0.bin"))
    ld = StreamingLoader(str(tmp_path), seq_len=8, micro_batch=4,
                         rank=0, world_size=1, seed=0, device="cpu")
    x1, _ = ld.next_batch()
    x2, _ = ld.next_batch()
    s1 = set(x1.flatten().tolist())
    s2 = set(x2.flatten().tolist())
    # Tokens are unique 0..9999, so disjoint sets means non-overlapping windows.
    assert s1.isdisjoint(s2)
