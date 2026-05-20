"""Run a few honest-to-goodness training steps on a local GPU.

This isn't training a real frontier model — it's training the *tiny*
transformer from ``platform.model.transformer`` for a handful of steps so
we can measure actual tokens/sec, ms/step, and peak memory on the local
silicon. The simulator then uses those measurements to recalibrate its
``seconds_per_step`` (instead of relying on ``peak_tflops * target_mfu``).

The result feels more "real" because:
  * Wall-clock for the simulated pretrain run is now derived from a number
    we actually observed on this machine.
  * The console report shows measured vs spec TFLOP/s and the implied MFU.
  * The events.jsonl gets a ``real_train.*`` event with the raw numbers,
    which downstream notebooks can plot.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
from typing import Optional


@dataclass
class RealTrainResult:
    device: str
    dtype: str
    n_params: int
    batch: int
    seq_len: int
    steps_measured: int
    ms_per_step: float
    tokens_per_sec: float
    achieved_tflops: float
    peak_mem_gb: float
    loss_first: float
    loss_last: float

    def as_dict(self) -> dict:
        return asdict(self)


def _pick_dtype(device_index: int):
    """Pick the most realistic dtype the device supports (bf16 > fp16 > fp32)."""
    import torch
    cap = torch.cuda.get_device_capability(device_index)
    if cap >= (8, 0):
        try:
            torch.zeros(1, device=f"cuda:{device_index}", dtype=torch.bfloat16) + 1
            return torch.bfloat16, "bf16"
        except Exception:
            pass
    if cap >= (5, 3):
        try:
            torch.zeros(1, device=f"cuda:{device_index}", dtype=torch.float16) + 1
            return torch.float16, "fp16"
        except Exception:
            pass
    return torch.float32, "fp32"


def measure_real_throughput(
    device_index: int = 0,
    n_layer: int = 4,
    d_model: int = 256,
    n_head: int = 4,
    n_kv_head: int = 2,
    seq_len: int = 512,
    batch: int = 4,
    vocab_size: int = 4096,
    warmup_steps: int = 2,
    measure_steps: int = 6,
    use_amp: bool = True,
) -> Optional[RealTrainResult]:
    """Train a tiny transformer for a few steps; return measured throughput.

    Returns ``None`` if CUDA is unavailable or the device can't be used.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        return None
    # If the device can't even run a tiny op (e.g. unsupported compute cap),
    # don't bother trying to train on it.
    try:
        x = torch.zeros(4, device=f"cuda:{device_index}") + 1
        torch.cuda.synchronize(device_index)
        del x
    except Exception:
        return None

    from ..model.config import ModelConfig
    from ..model.transformer import Transformer

    dev = torch.device(f"cuda:{device_index}")
    dtype, dtype_name = _pick_dtype(device_index)

    # Build a tiny model that comfortably fits on a 4GB card.
    d_ffn = 4 * d_model
    cfg = ModelConfig(
        vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head,
        d_model=d_model, d_ffn=d_ffn, max_seq_len=max(seq_len, 512),
        rope_base=10000.0,
    )
    torch.manual_seed(0)
    model = Transformer(cfg).to(dev)
    # For non-fp32 dtypes we use autocast around forward only (params stay fp32).
    if not use_amp:
        model = model.to(dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    n_params = sum(p.numel() for p in model.parameters())

    # Random token batches (we don't care about loss going down — only timing).
    def _batch():
        x = torch.randint(0, vocab_size, (batch, seq_len), device=dev)
        y = torch.randint(0, vocab_size, (batch, seq_len), device=dev)
        return x, y

    torch.cuda.reset_peak_memory_stats(dev)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if use_amp and dtype != torch.float32
        else _NullCtx()
    )

    losses: list[float] = []
    # Warmup
    for _ in range(warmup_steps):
        x, y = _batch()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx:
            _, loss = model(x, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(dev)

    start = time.perf_counter()
    for s in range(measure_steps):
        x, y = _batch()
        opt.zero_grad(set_to_none=True)
        with autocast_ctx:
            _, loss = model(x, y)
        losses.append(loss.detach().float().item())
        loss.backward()
        opt.step()
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - start

    ms_per_step = (elapsed / measure_steps) * 1000.0
    tokens_per_sec = (batch * seq_len * measure_steps) / elapsed
    # ~6 N D FLOPs/token (fwd+bwd) per Kaplan/Chinchilla rule of thumb
    flops = 6.0 * n_params * batch * seq_len * measure_steps
    achieved_tflops = flops / elapsed / 1e12
    peak_mem_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)

    return RealTrainResult(
        device=torch.cuda.get_device_name(device_index),
        dtype=dtype_name,
        n_params=n_params,
        batch=batch,
        seq_len=seq_len,
        steps_measured=measure_steps,
        ms_per_step=ms_per_step,
        tokens_per_sec=tokens_per_sec,
        achieved_tflops=achieved_tflops,
        peak_mem_gb=peak_mem_gb,
        loss_first=losses[0] if losses else 0.0,
        loss_last=losses[-1] if losses else 0.0,
    )


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def format_real_train_report(r: RealTrainResult) -> str:
    lines = [" MEASURED REAL TRAINING (this machine)"]
    lines.append(f"   device:       {r.device}  ({r.dtype})")
    lines.append(f"   model:        {r.n_params/1e6:.2f} M params, "
                 f"batch={r.batch}  seq={r.seq_len}")
    lines.append(f"   step time:    {r.ms_per_step:.1f} ms  →  "
                 f"{r.tokens_per_sec:,.0f} tok/s  "
                 f"({r.achieved_tflops:.2f} TFLOP/s)")
    lines.append(f"   peak memory:  {r.peak_mem_gb:.2f} GB")
    lines.append(f"   loss:         {r.loss_first:.3f} → {r.loss_last:.3f}")
    return "\n".join(lines)
