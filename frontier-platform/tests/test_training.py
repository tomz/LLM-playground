from __future__ import annotations
import pytest
import torch

from platform.model.transformer import Transformer, RMSNorm
from platform.training.optim import OptimConfig, build_optimizer
from platform.training.trainer import Trainer, TrainConfig
from platform.training.parallel import ParallelConfig


class _SyntheticLoader:
    """Deterministic infinite (x, y) generator for toy pretraining tests."""

    def __init__(self, vocab=256, batch=4, seq=32, seed=0):
        self.vocab, self.batch, self.seq = vocab, batch, seq
        self.g = torch.Generator().manual_seed(seed)
        # Bias the distribution so loss can actually drop below uniform.
        # Pattern: y[t] = (x[t] + 1) % vocab
        self._cache = None

    def __iter__(self):
        while True:
            x = torch.randint(0, self.vocab, (self.batch, self.seq + 1), generator=self.g)
            yield x[:, :-1].numpy(), ((x[:, :-1] + 1) % self.vocab).numpy()

    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass


def _make(seq=32, vocab=256):
    from platform.model.config import ModelConfig
    cfg = ModelConfig(
        vocab_size=vocab, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=seq,
    )
    torch.manual_seed(0)
    model = Transformer(cfg)
    return cfg, model


def test_tiny_pretrain_loss_decreases():
    _, model = _make()
    ocfg = OptimConfig(peak_lr=3e-3, warmup_steps=5, total_steps=200, weight_decay=0.0)
    opt, sched = build_optimizer(model, ocfg)
    tcfg = TrainConfig(
        run_id="unit", seq_len=32, micro_batch=4, total_tokens=10**12,
        log_every=10, eval_every=0, ckpt_every=0,
        optim=ocfg, parallel=ParallelConfig(),
    )
    tcfg.optim.total_steps = 100
    loader = _SyntheticLoader(vocab=256, batch=4, seq=32)
    tr = Trainer(model, loader, None, None, tcfg, optimizer=opt, scheduler=sched)
    tr.fit()
    h = tr.loss_history
    first = sum(h[:10]) / 10
    last = sum(h[-10:]) / 10
    assert last < first, (first, last)


@pytest.mark.gpu
def test_tiny_pretrain_on_cuda(gpu_or_skip):
    device = gpu_or_skip
    _, model = _make()
    model = model.to(device).to(torch.float16)
    ocfg = OptimConfig(peak_lr=1e-4, warmup_steps=5, total_steps=50, weight_decay=0.0, grad_clip=0.5)
    opt, sched = build_optimizer(model, ocfg)
    tcfg = TrainConfig(
        run_id="unit-gpu", seq_len=32, micro_batch=4, total_tokens=10**12,
        log_every=10, eval_every=0, ckpt_every=0,
        optim=ocfg, parallel=ParallelConfig(),
    )
    loader = _SyntheticLoader(vocab=256, batch=4, seq=32)
    tr = Trainer(model, loader, None, None, tcfg, optimizer=opt, scheduler=sched)
    tr.fit()
    assert len(tr.loss_history) == 50
    assert all(torch.isfinite(torch.tensor(x)) for x in tr.loss_history)


def test_optim_excludes_norms_from_wd(tiny_model):
    ocfg = OptimConfig(weight_decay=0.1)
    opt, _ = build_optimizer(tiny_model, ocfg)
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    # RMSNorm.weight is 1-D so must be in the no-decay group.
    rms_params = [p for m in tiny_model.modules() if isinstance(m, RMSNorm) for p in m.parameters()]
    for p in rms_params:
        assert any(p is q for q in no_decay_group["params"])
        assert not any(p is q for q in decay_group["params"])


def test_lr_scheduler_warmup_then_cosine(tiny_model):
    ocfg = OptimConfig(peak_lr=1.0, min_lr_ratio=0.1, warmup_steps=10, total_steps=100)
    opt, sched = build_optimizer(tiny_model, ocfg)
    lrs = []
    for _ in range(101):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == 0.0
    assert abs(lrs[10] - 1.0) < 1e-6
    assert abs(lrs[100] - 0.1) < 1e-3
