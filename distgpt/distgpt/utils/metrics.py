"""Training-time observability: grad/param norms, MFU, throughput accounting.

These metrics are cheap (microseconds per step at the scales we care about)
and turn the JSONL log into the single source of truth a frontier-scale run
needs for postmortems. Grad-norm in particular is the single best early-
warning signal for divergence — a diverging run usually announces itself in
grad-norm 100+ steps before loss spikes.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Norm computations
# ---------------------------------------------------------------------------


def compute_grad_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """L2 norm of all gradients, computed identically to ``clip_grad_norm_``
    but without applying any clipping. Use this on the no-clip path so the
    log still has the diagnostic. Returns 0.0 if no grads are populated."""
    grads = [
        p.grad for p in model.parameters()
        if p.grad is not None and p.requires_grad
    ]
    if not grads:
        return 0.0
    # Match torch.nn.utils.clip_grad_norm_'s computation exactly so the
    # values are comparable across runs that do/don't clip.
    device = grads[0].device
    total = torch.stack(
        [torch.linalg.vector_norm(g.detach().to(torch.float32), norm_type)
         for g in grads]
    ).to(device)
    return float(torch.linalg.vector_norm(total, norm_type))


def compute_param_norm(model: torch.nn.Module, norm_type: float = 2.0) -> float:
    """L2 of all model parameters. Slow on big models so sample sparsely
    (the trainer takes one sample every 10× log_every)."""
    params = [p.detach().to(torch.float32) for p in model.parameters()
              if p.requires_grad]
    if not params:
        return 0.0
    return float(torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p, norm_type) for p in params]),
        norm_type,
    ))


# ---------------------------------------------------------------------------
# MFU
# ---------------------------------------------------------------------------


# Vendor-published peak (dense, non-sparse) TFLOPs at each precision. These
# are theoretical peaks — real MFU caps around 50-60% on a well-tuned run.
# Source: NVIDIA datasheets. Conservative when unsure.
#
# Format: (name_substring, {dtype: tflops})
_GPU_PEAK_TFLOPS = [
    ("H100",  {torch.bfloat16: 989, torch.float16: 989, torch.float32: 67}),
    ("H200",  {torch.bfloat16: 989, torch.float16: 989, torch.float32: 67}),
    ("A100",  {torch.bfloat16: 312, torch.float16: 312, torch.float32: 19.5}),
    ("V100",  {torch.float16: 125, torch.float32: 15.7}),
    ("L40S",  {torch.bfloat16: 362, torch.float16: 362, torch.float32: 91.6}),
    ("L4",    {torch.bfloat16: 121, torch.float16: 121, torch.float32: 30.3}),
    ("B200",  {torch.bfloat16: 2250, torch.float16: 2250, torch.float32: 70}),
    # RTX consumer cards (sm_120 / Blackwell, sm_89 / Ada).
    ("RTX 5060", {torch.bfloat16: 178, torch.float16: 178, torch.float32: 22}),
    ("RTX 5090", {torch.bfloat16: 838, torch.float16: 838, torch.float32: 105}),
    ("RTX 4090", {torch.bfloat16: 330, torch.float16: 330, torch.float32: 82.6}),
    ("RTX 3090", {torch.float16: 142, torch.float32: 35.6}),
    # Pascal P100 — no bf16, no Tensor Cores.
    ("P100", {torch.float16: 19.05, torch.float32: 9.5}),
]


def peak_tflops_for_device(dtype: torch.dtype) -> float | None:
    """Best-effort peak TFLOPs at ``dtype`` for the active CUDA device.

    Returns ``None`` on CPU or for a GPU we don't recognize (so the trainer
    can omit MFU from the log rather than report nonsense). The table is
    extended via PRs as new SKUs appear.
    """
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for sub, table in _GPU_PEAK_TFLOPS:
        if sub in name and dtype in table:
            return float(table[dtype])
    return None


def model_flops_per_token(cfg) -> float:
    """6N approximation for forward+backward (Chinchilla / PaLM).

    N is the active parameter count, so for MoE models pass the *active*
    config. The 6× includes: 2× for forward (one mat * vec), 4× for the
    backward grad-of-output + grad-of-weight passes. Attention's quadratic
    term is folded into the constant for short context; long-context runs
    should add a separate ``+12 L H D T^2`` term — for now we keep it simple.
    """
    return 6.0 * float(cfg.param_count())


def estimate_mfu(cfg, tokens_per_step: int, dt_seconds: float,
                  peak_tflops: float, world_size: int) -> float | None:
    """Model-FLOPs-utilization, in [0, 1]. Returns ``None`` when inputs
    are malformed (so the caller can suppress the log entry rather than
    print 'inf MFU')."""
    if dt_seconds <= 0 or peak_tflops <= 0 or world_size <= 0:
        return None
    flops = model_flops_per_token(cfg) * tokens_per_step
    achieved_tflops = (flops / dt_seconds) / 1e12 / world_size
    return achieved_tflops / peak_tflops


__all__ = [
    "compute_grad_norm",
    "compute_param_norm",
    "peak_tflops_for_device",
    "model_flops_per_token",
    "estimate_mfu",
]
