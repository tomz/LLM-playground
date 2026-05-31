"""SpikeMonitor (two-threshold detector) and RewindController (LR halving)."""
from __future__ import annotations
import sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from stability import RewindController, SpikeMonitor


def test_spike_monitor_quiet_below_window():
    """No spike can fire before the window is full — the running mean/std are
    undefined until then. Pinned because the old prototype dereferenced an
    empty deque before this guard existed."""
    m = SpikeMonitor(window=10, sigma=5.0, min_abs_jump=2.0)
    # First 10 observations should always return False (filling the window).
    for v in [3.5] * 10:
        assert m.observe(v) is False


def test_spike_monitor_z_score_alone_does_not_fire():
    """Even a large z-score must coexist with a large absolute jump.

    This is the load-bearing rule that prevented the distgpt-Tier-4 rewind-loop
    bug: once the loss has plateaued, the running std collapses to a few
    hundredths and a normal 0.3-loss jitter blip is ~7σ from the mean. Without
    ``min_abs_jump`` the trainer would rewind, train back to the same plateau,
    rewind, train back to the same plateau, ... in an endless loop."""
    m = SpikeMonitor(window=20, sigma=5.0, min_abs_jump=2.0)
    for _ in range(20):
        m.observe(4.0)
    # 0.3 jump (10× the post-floor std) is still less than the 2.0 abs floor.
    # That's >5σ relative — would fire by z-score alone — but doesn't here.
    assert m.observe(4.3) is False


def test_spike_monitor_fires_on_real_spike():
    """Real spike: jump of 3.0 over a stable mean of 4.0 (i.e. loss doubles).
    Both thresholds clear."""
    m = SpikeMonitor(window=20, sigma=5.0, min_abs_jump=2.0)
    for _ in range(20):
        m.observe(4.0)
    assert m.observe(7.0) is True


def test_spike_monitor_window_slides():
    """After firing on a spike, the window keeps moving — a second spike
    farther down the curve must also fire (the monitor isn't latched)."""
    m = SpikeMonitor(window=10, sigma=5.0, min_abs_jump=2.0)
    for _ in range(10):
        m.observe(4.0)
    assert m.observe(7.0) is True
    # 10 more clean steps drain the spike from the window.
    for _ in range(10):
        m.observe(4.0)
    assert m.observe(7.0) is True


def test_rewinder_caps_at_max_rewinds():
    """A chronically spiky model must eventually stop rewinding so the cosine
    schedule's planned LR can still take over. Without this cap a real run
    once spent 6 hours regressing through the same 100 steps (the bug history
    distgpt's stability.py records)."""
    loads: list[tuple[str, int]] = []
    r = RewindController(
        load_ckpt_fn=lambda p: (loads.append((p, len(loads))), 42)[1],
        last_ckpt_path_fn=lambda: "ckpt.pt",
        lr_floor=1e-3, max_rewinds=3, cooldown_steps=10,
    )
    for _ in range(10):
        r.on_spike(current_step=100)
    assert r.n_rewinds == 3
    assert len(loads) == 3


def test_rewinder_no_op_without_checkpoint():
    """Early in the run there's no ``ckpt.pt`` yet — the rewinder must not
    crash and must not consume a rewind slot."""
    r = RewindController(
        load_ckpt_fn=lambda p: 0,
        last_ckpt_path_fn=lambda: None,
        max_rewinds=3,
    )
    out = r.on_spike(current_step=42)
    assert out == 42
    assert r.n_rewinds == 0


def test_rewinder_halves_lr_for_cooldown_only():
    """On rewind: LR scale halves and stays halved for ``cooldown_steps``
    consecutive calls, then snaps back to 1.0."""
    r = RewindController(
        load_ckpt_fn=lambda p: 0,
        last_ckpt_path_fn=lambda: "ckpt.pt",
        lr_floor=1e-4, max_rewinds=3, cooldown_steps=5,
    )
    # Before any spike: full LR.
    assert r.lr_multiplier() == 1.0
    r.on_spike(current_step=0)
    # 5 cooldown steps at 0.5x.
    for _ in range(5):
        assert r.lr_multiplier() == pytest.approx(0.5)
    # Cooldown over: snap back.
    assert r.lr_multiplier() == 1.0


def test_rewinder_lr_floor_caps_halving():
    """Successive rewinds halve the LR scale further but stop at ``lr_floor``,
    so the model never trains at 1e-12 of the schedule's LR."""
    r = RewindController(
        load_ckpt_fn=lambda p: 0,
        last_ckpt_path_fn=lambda: "ckpt.pt",
        lr_floor=0.25, max_rewinds=10, cooldown_steps=1,
    )
    r.on_spike(0)          # → 0.5
    assert r.scale == pytest.approx(0.5)
    r.on_spike(0)          # → 0.25 (floor)
    assert r.scale == pytest.approx(0.25)
    r.on_spike(0)          # would be 0.125 but clamped at floor.
    assert r.scale == pytest.approx(0.25)


def test_spike_monitor_disabled_by_default_in_trainer(tmp_path):
    """End-to-end: when ``stability.spike_monitor`` is absent from the YAML,
    the trainer must not touch the new code path (default-off). A 2-step
    smoke run should land at the same place as before this tier added the
    spike machinery."""
    import json, os, subprocess, sys
    import numpy as np

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    (work / "data" / "stab_smoke" / "train").mkdir(parents=True)
    (work / "data" / "stab_smoke" / "val").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(
        str(work / "data" / "stab_smoke" / "train" / "s.bin"))
    rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(
        str(work / "data" / "stab_smoke" / "val" / "s.bin"))
    for f in ("train.py", "model.py", "data.py", "muon.py", "stability.py"):
        (work / f).symlink_to(ROOT / f)
    (work / "utils").symlink_to(ROOT / "utils")

    out = work / "out"
    cfg = work / "cfg.yaml"
    cfg.write_text(f"""
out_dir: {out}
dataset: stab_smoke
tokenizer: gpt2
seed: 0
dtype: float32
compile: false
grad_checkpoint: false
log: {{jsonl: true, wandb_project: null}}
model: {{vocab_size: 256, block_size: 16, n_layer: 2, n_head: 2,
         d_model: 32, d_ffn: 64, tie_embeddings: true}}
optim: {{lr: 0.001, min_lr: 0.0001, betas: [0.9, 0.95], weight_decay: 0.0,
         grad_clip: 1.0, warmup_iters: 1, lr_decay_iters: 2, max_iters: 2}}
train: {{micro_batch: 2, grad_accum: 1, eval_interval: 100, eval_iters: 1,
         log_interval: 1, ckpt_interval: 1}}
""")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "WORLD_SIZE": "1"}
    p = subprocess.run([sys.executable, "train.py", "--config", str(cfg)],
                       cwd=str(work), env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    # No spike-machinery messages: the default-off path must not announce
    # SpikeMonitor / RewindController setup, and the spike-detection line
    # (which begins with the warning marker) must not fire.
    assert "stability: SpikeMonitor" not in p.stdout
    assert "⚠ spike at iter" not in p.stdout
    # And the run wrote a log file with loss rows.
    rows = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert any("loss" in r for r in rows)
