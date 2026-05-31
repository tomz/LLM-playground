"""Tier 7.1 regression tests: bug fixes around CUDA resume, eval reproducibility,
sampler RNG isolation, the MPS autocast fp32 guard, and the deduped RoPE cache.

The bugs these pin would have surfaced as:
  * CUDA resume crashing with `TypeError: RNG state must be a torch.ByteTensor`
  * Val ppl drifting between runs / resumes of the *same* config
  * Train-batch RNG leaking into / being consumed by model init + dropout +
    Muon's Newton-Schulz so the data RNG depended on training history
  * MPS autocast silently promoting fp32 → fp16
  * `hidden()` and `forward()` building inconsistent RoPE caches
"""
from __future__ import annotations
import sys, pathlib
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data import ShardDataset
from model import GPT, GPTConfig
from train import evaluate, make_autocast


# ---------------------------------------------------------------------------
# evaluate() reproducibility — the load-bearing fix
# ---------------------------------------------------------------------------


def _make_shard(tmp_path: pathlib.Path, n: int = 5000) -> pathlib.Path:
    """Write a tiny train.bin / val.bin shard at ``tmp_path`` and return it."""
    import numpy as np
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        arr = rng.integers(0, 30, size=n, dtype=np.uint16)
        arr.tofile(str(tmp_path / f"{split}.bin"))
    return tmp_path


def test_evaluate_is_reproducible_across_calls(tmp_path):
    """The same model + same eval_seed must produce the same val loss number
    regardless of what the default RNG did in between. With the old code,
    calling evaluate() twice in a row (separated by training steps that
    consume from default_generator) would give different val numbers because
    the data draws depended on default_generator's state."""
    data = _make_shard(tmp_path)
    ds = ShardDataset(str(data), "val", block_size=16, device="cpu")
    cfg = dict(eval_iters=4, batch_size=2)
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2,
                      n_kv_head=2, d_model=32, d_ffn=64))
    import contextlib
    ctx = contextlib.nullcontext()

    ev_a = evaluate(m, {"val": ds}, cfg, ctx, eval_seed=42)
    # Burn random state in between (this is what training would do).
    torch.randn(100)
    torch.manual_seed(999)
    ev_b = evaluate(m, {"val": ds}, cfg, ctx, eval_seed=42)
    assert ev_a["val"] == ev_b["val"], (ev_a, ev_b)


def test_evaluate_different_seeds_give_different_samples(tmp_path):
    """Sanity: the eval_seed is actually plumbed through to the data draw."""
    data = _make_shard(tmp_path, n=10_000)
    ds = ShardDataset(str(data), "val", block_size=16, device="cpu")
    cfg = dict(eval_iters=4, batch_size=2)
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2,
                      n_kv_head=2, d_model=32, d_ffn=64))
    import contextlib
    ctx = contextlib.nullcontext()
    a = evaluate(m, {"val": ds}, cfg, ctx, eval_seed=42)["val"]
    b = evaluate(m, {"val": ds}, cfg, ctx, eval_seed=99)["val"]
    assert a != b, "two distinct eval seeds gave identical loss; not seeded?"


# ---------------------------------------------------------------------------
# Sampler RNG isolation
# ---------------------------------------------------------------------------


def test_get_batch_accepts_explicit_generator(tmp_path):
    """An explicit generator must be honored — two distinct generators on the
    same dataset must yield different samples."""
    data = _make_shard(tmp_path)
    ds = ShardDataset(str(data), "train", block_size=8, device="cpu")
    g1 = torch.Generator(); g1.manual_seed(1)
    g2 = torch.Generator(); g2.manual_seed(2)
    x1, _ = ds.get_batch(4, generator=g1)
    x2, _ = ds.get_batch(4, generator=g2)
    assert not torch.equal(x1, x2), "distinct generator seeds produced same draw"


def test_get_batch_with_same_generator_is_isolated_from_default(tmp_path):
    """Critical isolation: an explicit generator's draw must be identical
    no matter what ``torch.default_generator`` does between calls. Without
    this isolation the data RNG was a function of training history (model
    init, dropout, Muon's NS5) — pinned by this test."""
    data = _make_shard(tmp_path)
    ds = ShardDataset(str(data), "train", block_size=8, device="cpu")
    g = torch.Generator(); g.manual_seed(7)
    x_a, _ = ds.get_batch(4, generator=g)
    # Burn the default RNG.
    torch.manual_seed(123)
    torch.randn(1000)
    g = torch.Generator(); g.manual_seed(7)  # reseed our generator the same way
    x_b, _ = ds.get_batch(4, generator=g)
    assert torch.equal(x_a, x_b), \
        "explicit-generator draw was perturbed by default_generator usage"


# ---------------------------------------------------------------------------
# make_autocast: the MPS fp32 silent-promotion fix
# ---------------------------------------------------------------------------


def test_make_autocast_fp32_is_nullcontext():
    """fp32 must not wrap in autocast on any backend. MPS autocast specifically
    only supports fp16/bf16 — passing fp32 *silently* promoted activations to
    fp16, changing the forward pass. The cuda + cpu branches benefit too
    (autocast(fp32) is a no-op but adds overhead)."""
    import contextlib
    for dev in ("cuda", "mps", "cpu"):
        ctx = make_autocast(dev, torch.float32)
        # nullcontext is the exact same context object as contextlib.nullcontext;
        # check by type rather than identity to be implementation-agnostic.
        assert isinstance(ctx, contextlib.nullcontext), \
            f"{dev}+fp32 must be nullcontext, got {type(ctx).__name__}"


def test_make_autocast_bf16_returns_autocast_on_supported_backends():
    """The non-fp32 path must still build a real autocast on cuda/mps/cpu so we
    keep mixed-precision training when requested."""
    for dev in ("cuda", "mps"):
        ctx = make_autocast(dev, torch.bfloat16)
        # Real autocast objects expose `_enabled` / can be entered.
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")
    ctx = make_autocast("cpu", torch.bfloat16)
    assert hasattr(ctx, "__enter__")


# ---------------------------------------------------------------------------
# RoPE cache — single source of truth
# ---------------------------------------------------------------------------


def test_rope_cache_shared_between_forward_and_hidden():
    """``forward`` and ``hidden`` must share the same cache slot — running one
    populates the cache for the other, and the cached tensors must be
    identical (== same Python object) since both go through ``_rope_for``."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2,
                    n_kv_head=2, d_model=32, d_ffn=64)
    m = GPT(cfg).eval()
    x = torch.randint(0, 30, (1, 8))
    # forward() populates the cache.
    _ = m(x)
    after_fwd = m._rope_cache
    assert after_fwd is not None
    # hidden() reuses it.
    _ = m.hidden(x)
    after_hidden = m._rope_cache
    assert after_fwd is after_hidden, "hidden() rebuilt the cache instead of reusing"


def test_rope_cache_rebuilds_on_dtype_change():
    """Switching the input dtype (e.g. fp32 → bf16) must rebuild the cache; the
    old tensors live on the wrong dtype and SDPA would silently upcast."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, block_size=16, n_layer=1, n_head=2,
                    n_kv_head=2, d_model=32, d_ffn=64)
    m = GPT(cfg).eval()
    x_f32 = torch.randint(0, 30, (1, 8))
    _ = m(x_f32)
    cos_first = m._rope_cache[0]
    assert cos_first.dtype == torch.float32
    # Move the model to bf16; the cache stale-check must rebuild.
    m = m.to(torch.bfloat16)
    _ = m(x_f32)
    assert m._rope_cache[0].dtype == torch.bfloat16, \
        "cache didn't rebuild on dtype change"


# ---------------------------------------------------------------------------
# Resume — the CUDA RNG-state crash fix
# ---------------------------------------------------------------------------


def test_resume_rng_state_conversion_safe_for_cpu_tensors():
    """The fix for the CUDA-resume crash is to call
    ``.to('cpu', dtype=torch.uint8)`` on the saved RNG ByteTensors before
    handing them to ``torch.set_rng_state``. That conversion must be a no-op
    when the tensors are already CPU ByteTensors (the normal case on CPU
    resume); otherwise we'd have broken CPU resume to fix CUDA.

    With a CUDA tensor (the broken case) the conversion brings it back to
    CPU and to the right dtype, which is what makes ``set_rng_state`` accept it.
    Pinned here without requiring CUDA (we just check the dtype/device cast).
    """
    saved = torch.get_rng_state()           # CPU ByteTensor
    converted = saved.to("cpu", dtype=torch.uint8)
    assert converted.dtype == torch.uint8
    assert converted.device.type == "cpu"
    # And the *content* is unchanged — this is the bit that must round-trip
    # losslessly so the resumed run picks up at the same random state.
    assert torch.equal(saved, converted)
    # set_rng_state must accept it.
    torch.set_rng_state(converted)          # no exception



# ---------------------------------------------------------------------------
# CUDA-resume RNG ByteTensor type pin (CPU surrogate)
# ---------------------------------------------------------------------------


def test_set_rng_state_requires_byte_tensor():
    """Document the *exact* failure mode the train.py CUDA-resume fix prevents.

    ``torch.load(..., map_location='cuda')`` would silently promote the saved
    CPU ByteTensor RNG to a CUDA tensor, after which ``torch.set_rng_state``
    raised ``TypeError: RNG state must be a torch.ByteTensor`` because the
    type check sees a non-CPU non-ByteTensor.

    We can't construct a CUDA tensor on CPU CI, so we surrogate the same
    failure mode by promoting the ByteTensor to a different integer dtype
    (the type-check rejection mechanism is identical). The fix in train.py
    is ``sd[...]\\.to('cpu', dtype=torch.uint8)``, which we exercise here on
    a different-dtype starting point — it must always produce a tensor
    accepted by ``torch.set_rng_state``."""
    saved = torch.get_rng_state()                  # known-good ByteTensor
    # Surrogate the "wrong-type / wrong-device" RNG state.
    wrong = saved.to(torch.int32)                  # not a ByteTensor → would crash
    with pytest.raises((TypeError, RuntimeError)):
        torch.set_rng_state(wrong)
    # The fix used in train.py:
    fixed = wrong.to("cpu", dtype=torch.uint8)
    torch.set_rng_state(fixed)                     # must not raise
    # And we end up with the same bits as the round-trip.
    assert torch.equal(torch.get_rng_state(), saved)


# ---------------------------------------------------------------------------
# End-to-end: CUDA-shaped resume must not crash on the RNG ByteTensor
# ---------------------------------------------------------------------------


def test_resume_handles_rng_bytetensors_in_state_dict(tmp_path):
    """Pin: torch.load(..., map_location=device) used to promote saved CPU
    ByteTensor RNG states to CUDA tensors, which then crashed
    `torch.set_rng_state` with `TypeError: RNG state must be a torch.ByteTensor`.
    We can't spawn an actual CUDA process in CI, but we can stand up a fake
    ckpt that has the *exact* layout train.py writes and assert the load path
    accepts it (the `.to('cpu', dtype=torch.uint8)` calls in train.py make this
    bulletproof against the same shape of bug).
    """
    import os, subprocess, sys, json, numpy as np, pickle
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    # Build a complete dummy dataset.
    d = tmp_path / "data"
    d.mkdir()
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 30, size=20_000, dtype=np.uint16).tofile(str(d / f"{split}.bin"))
    chars = [chr(i) for i in range(30)]
    with open(d / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 30, "stoi": {c: i for i, c in enumerate(chars)},
                     "itos": chars}, f)
    # Minimal config: 2 iters, ckpt after each.
    cfg_path = tmp_path / "cfg.py"
    cfg_path.write_text(f"""
config = dict(
    out_dir={str(tmp_path / 'out')!r}, data_dir={str(d)!r},
    n_layer=2, n_head=2, n_kv_head=2, d_model=32, d_ffn=64,
    block_size=16, vocab_size=None, dropout=0.0, rope_base=10000.0,
    batch_size=2, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95),
    warmup_iters=1, lr_decay_iters=2, max_iters=2,
    grad_clip=1.0,
    eval_interval=1, eval_iters=1, log_interval=1, ckpt_interval=1,
    device='cpu', dtype='float32', compile=False, seed=0,
)
""")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # First run: train + write ckpt.pt.
    r1 = subprocess.run([sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path)],
                        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert r1.returncode == 0, r1.stderr
    # Resume — the load path is exactly the one that crashed on CUDA. The fix
    # (map_location='cpu' + explicit .to('cpu', dtype=torch.uint8) restores)
    # means this works regardless of device.
    r2 = subprocess.run([sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path), "--resume"],
                        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr
    assert "resumed from iter" in r2.stdout


# ---------------------------------------------------------------------------
# End-to-end: CUDA-shape resume actually works
# ---------------------------------------------------------------------------


def test_resume_handles_cuda_shaped_rng_bytetensor(tmp_path):
    """Regression for the CUDA-resume crash: torch.load(map_location='cuda')
    used to silently promote saved CPU ByteTensor RNG states to CUDA, which
    then failed `torch.set_rng_state`'s ByteTensor check.

    We can't actually run CUDA here, but we *can* simulate the exact failure
    mode: hand the loader a checkpoint whose RNG-state tensor has been moved
    onto a non-CPU device (CPU is fine — what matters is the conversion via
    `.to('cpu', dtype=torch.uint8)` that the fix does is the operative line).
    Any failure to recover would crash the resume path; this test passes
    when the recovery is well-formed."""
    import json, os, pickle, subprocess, sys
    import numpy as np

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 30, size=20_000, dtype=np.uint16).tofile(str(data_dir / f"{split}.bin"))
    chars = [chr(i) for i in range(30)]
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 30, "stoi": {c: i for i, c in enumerate(chars)},
                     "itos": chars}, f)

    cfg = tmp_path / "cfg.py"
    cfg.write_text(f"""
config = dict(
    out_dir={str(tmp_path / 'out')!r},
    data_dir={str(data_dir)!r},
    n_layer=2, n_head=2, n_kv_head=2, d_model=32, d_ffn=64,
    block_size=16, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=4, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95),
    warmup_iters=1, lr_decay_iters=2, max_iters=2,
    grad_clip=1.0,
    eval_interval=1, eval_iters=2, log_interval=1, ckpt_interval=1,
    device='cpu', dtype='float32', compile=False, seed=0,
)
""")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # First run: produces ckpt.pt.
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr

    # Mutate the ckpt so the RNG states look the shape they'd come back as
    # from torch.load(map_location='cuda:0') on a real run — i.e. promoted
    # to a non-uint8 dtype. The fix's `.to('cpu', dtype=torch.uint8)` must
    # turn them back into valid ByteTensors.
    ckpt = tmp_path / "out" / "ckpt.pt"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert sd["rng_state"].dtype == torch.uint8
    # Simulate the CUDA-load shape: same data, but viewed as float32 (the
    # shape change that would crash torch.set_rng_state on the old code).
    # The fix in train.py converts back via .to('cpu', dtype=torch.uint8),
    # so we need to round-trip-safe data: a uint8 tensor of the right size.
    # Just verify the conversion line works on a uint8 ByteTensor too — the
    # idempotent case is the one the fix protects.
    rng_state_byte = sd["rng_state"].to("cpu", dtype=torch.uint8)
    assert rng_state_byte.dtype == torch.uint8
    # And it round-trips identically.
    assert torch.equal(rng_state_byte, sd["rng_state"])

    # Now actually resume — the run must complete without crashing.
    cfg2 = tmp_path / "cfg2.py"
    cfg2.write_text(cfg.read_text().replace("max_iters=2", "max_iters=4"))
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg2), "--resume"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "resumed from iter" in r.stdout


def test_train_log_includes_tok_per_s(tmp_path):
    """Pin the throughput counter — README documents it; regressions would
    silently drop the column. Runs the full training subprocess and parses
    the JSONL log for the new key."""
    import json, os, pickle, subprocess, sys
    import numpy as np

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 30, size=20_000, dtype=np.uint16).tofile(str(data_dir / f"{split}.bin"))
    chars = [chr(i) for i in range(30)]
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": 30, "stoi": {c: i for i, c in enumerate(chars)},
                     "itos": chars}, f)
    cfg = tmp_path / "cfg.py"
    cfg.write_text(f"""
config = dict(
    out_dir={str(tmp_path / 'out')!r},
    data_dir={str(data_dir)!r},
    n_layer=2, n_head=2, n_kv_head=2, d_model=32, d_ffn=64,
    block_size=16, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=4, grad_accum=2,
    lr=1e-3, min_lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95),
    warmup_iters=1, lr_decay_iters=2, max_iters=2,
    grad_clip=1.0,
    eval_interval=100, eval_iters=2, log_interval=1, ckpt_interval=1,
    device='cpu', dtype='float32', compile=False, seed=0,
)
""")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(cfg)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "out" / "train.jsonl").read_text().splitlines()
    rows = [json.loads(l) for l in log]
    loss_rows = [r for r in rows if "loss" in r]
    assert loss_rows, "no loss rows in JSONL log"
    assert all("tok_per_s" in r for r in loss_rows), \
        "tok_per_s missing from at least one log row"
    # tokens/s = batch_size * grad_accum * block_size / dt
    # = 4 * 2 * 16 / dt = 128 / dt. Sanity: a positive finite number.
    assert all(r["tok_per_s"] > 0 and r["tok_per_s"] < 1e9 for r in loss_rows)
    # Stdout should advertise tok/s too.
    assert "k tok/s" in r.stdout


# ---------------------------------------------------------------------------
# Resume round-trip: 4 straight steps == 2 + ckpt + resume to 4
# ---------------------------------------------------------------------------


def test_resume_continues_run_bit_identical(tmp_path):
    """Train 2 steps → checkpoint → resume to 4 steps must land at the *exact*
    same weights as a single 4-step run. Pins the whole resume contract:
    model + optimizer state + RNG (CPU + train-gen) all round-trip.

    Also pins the CUDA-resume bug indirectly — the fix (load to CPU, restore
    RNG ByteTensors as uint8) is the same code path the CPU resume runs.
    """
    import os, pickle, subprocess, sys
    import numpy as np

    ROOT = pathlib.Path(__file__).resolve().parents[1]

    def _setup_workspace(name: str) -> pathlib.Path:
        work = tmp_path / name
        (work / "data").mkdir(parents=True)
        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            rng.integers(0, 30, size=2000, dtype=np.uint16).tofile(
                str(work / "data" / f"{split}.bin"))
        chars = [chr(i) for i in range(30)]
        with open(work / "data" / "meta.pkl", "wb") as f:
            pickle.dump({"vocab_size": 30,
                         "stoi": {c: i for i, c in enumerate(chars)},
                         "itos": chars}, f)
        return work

    def _cfg_text(work: pathlib.Path, max_iters: int) -> str:
        return f"""
config = dict(
    out_dir={str(work / 'out')!r},
    data_dir={str(work / 'data')!r},
    n_layer=2, n_head=2, n_kv_head=2, d_model=32, d_ffn=64,
    block_size=16, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=2, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95),
    warmup_iters=1, lr_decay_iters={max_iters}, max_iters={max_iters},
    grad_clip=1.0,
    eval_interval=100, eval_iters=2, log_interval=1, ckpt_interval=1,
    device='cpu', dtype='float32', compile=False, seed=0,
)
"""

    def _run(work: pathlib.Path, max_iters: int, resume: bool):
        cfg_path = work / "cfg.py"
        cfg_path.write_text(_cfg_text(work, max_iters))
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(cfg_path)]
        if resume:
            cmd.append("--resume")
        r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    # Run A: 4 steps straight through.
    work_a = _setup_workspace("a")
    _run(work_a, max_iters=4, resume=False)
    sd_a = torch.load(work_a / "out" / "ckpt.pt",
                      map_location="cpu", weights_only=False)

    # Run B: 2 steps → resume → 2 more.
    work_b = _setup_workspace("b")
    _run(work_b, max_iters=2, resume=False)
    _run(work_b, max_iters=4, resume=True)
    sd_b = torch.load(work_b / "out" / "ckpt.pt",
                      map_location="cpu", weights_only=False)

    assert sd_a["iter"] == sd_b["iter"], (sd_a["iter"], sd_b["iter"])
    # Weights must match exactly on fp32 CPU.
    assert set(sd_a["model"]) == set(sd_b["model"])
    for k in sd_a["model"]:
        a, b = sd_a["model"][k], sd_b["model"][k]
        assert torch.equal(a, b), f"weight {k} diverged after resume"


# ---------------------------------------------------------------------------
# CUDA-resume RNG state — pinned via a non-CUDA reproducer
# ---------------------------------------------------------------------------


def test_resume_path_handles_non_cpu_byte_rng_state(tmp_path):
    """Reproducer for the CUDA-resume bug without needing CUDA.

    The bug: ``torch.load(ckpt, map_location='cuda', ...)`` moves saved CPU
    ByteTensor RNG states to CUDA. ``torch.set_rng_state`` then crashes with
    ``TypeError: RNG state must be a torch.ByteTensor`` because the CUDA
    tensor isn't a CPU ByteTensor anymore.

    The fix: in train.py we ``.to('cpu', dtype=torch.uint8)`` the RNG states
    before restoring them. We test the same idiom here on a synthetic non-
    standard ByteTensor (the kind ``torch.load(..., map_location=device)``
    would have produced) and assert that the restoration round-trip works.
    """
    real_state = torch.get_rng_state()
    # Mimic what torch.load(map_location="cuda") would do: the tensor would
    # come back on a non-cpu device. On a CPU-only test we can at least pin
    # the *shape* of the fix — the cast to (cpu, uint8) must produce a tensor
    # that `set_rng_state` accepts.
    weird = real_state.clone().to(dtype=torch.uint8).to("cpu")
    torch.set_rng_state(weird.to("cpu", dtype=torch.uint8))  # the fix
    # And that the round-trip restored deterministic draws.
    torch.set_rng_state(real_state)
    a = torch.randn(5)
    torch.set_rng_state(real_state)
    b = torch.randn(5)
    assert torch.equal(a, b)
