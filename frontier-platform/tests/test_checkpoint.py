from __future__ import annotations
import torch

from platform.model.transformer import Transformer
from platform.model.config import ModelConfig
from platform.training.optim import OptimConfig, build_optimizer
from platform.training.checkpoint import CheckpointManager
from platform.training.parallel import ParallelEngine, ParallelConfig
from platform.training.stability import SpikeMonitor, RewindController


def _engine():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=32)
    torch.manual_seed(0)
    model = Transformer(cfg)
    opt, _ = build_optimizer(model, OptimConfig(peak_lr=1e-3))
    pcfg = ParallelConfig()
    pcfg.grad_clip = 1.0
    return model, ParallelEngine(model, opt, pcfg)


def test_save_load_roundtrip(tmp_path):
    model, eng = _engine()
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    for _ in range(10):
        eng.forward_backward((x, y))
        eng.step()
    mgr = CheckpointManager(str(tmp_path), "r1")
    mgr.save_async(eng, None, step=10)

    with torch.no_grad():
        _, loss_before = model(x, targets=y)

    # Fresh model+optim, then load.
    model2, eng2 = _engine()
    mgr.load_into(eng2, None, step="latest")
    with torch.no_grad():
        _, loss_after = model2(x, targets=y)
    assert abs(float(loss_before) - float(loss_after)) < 1e-4


def test_keep_last_gc(tmp_path):
    _, eng = _engine()
    mgr = CheckpointManager(str(tmp_path), "r2", keep_last=3)
    for s in range(1, 6):
        mgr.save_async(eng, None, step=s)
    remaining = sorted(p.name for p in (tmp_path / "r2" / "ckpts").iterdir())
    assert len(remaining) == 3
    assert remaining == ["step_000000003", "step_000000004", "step_000000005"]


def test_spike_triggers_rewind(tmp_path):
    _, eng = _engine()
    for pg in eng.optimizer.param_groups:
        pg["lr"] = 1e-3
    mgr = CheckpointManager(str(tmp_path), "r3")
    mgr.save_async(eng, None, step=1)
    rc = RewindController(mgr, lr_floor=1e-9)
    rc.on_spike(eng, current_step=42)
    assert abs(eng.optimizer.param_groups[0]["lr"] - 5e-4) < 1e-9

    sm = SpikeMonitor(window=20, sigma=4.0)
    for _ in range(20):
        assert not sm.observe(1.0)
    assert sm.observe(50.0) is True
