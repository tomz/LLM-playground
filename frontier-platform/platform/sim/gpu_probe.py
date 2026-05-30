"""Probe local CUDA GPUs and micro-benchmark them.

When torch + CUDA are available, this enumerates devices, runs a square
matmul benchmark to measure achieved TFLOP/s in fp32 (and fp16/bf16 if
supported), and runs a large device-to-device copy to estimate HBM
bandwidth. Results plug straight back into :data:`platform.sim.cluster.GPU_SPECS`
so the rest of the simulator can use measured numbers instead of vendor
spec-sheet peaks.

All torch imports are lazy — importing this module on a machine without
torch is a no-op.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GPUProbeResult:
    index: int
    name: str
    compute_capability: tuple[int, int]
    total_mem_gb: float
    # measured TFLOP/s per dtype (only those we ran)
    tflops: dict[str, float] = field(default_factory=dict)
    # device-to-device copy throughput, GB/s
    hbm_bandwidth_gb_s: float = 0.0
    # vendor spec-sheet TFLOP/s (BF16) for the closest matching SKU, if known
    spec_tflops_bf16: Optional[float] = None
    # implied MFU = best_measured_tflops / spec_tflops_bf16
    implied_mfu: Optional[float] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["compute_capability"] = list(self.compute_capability)
        return d

    @property
    def best_tflops(self) -> float:
        return max(self.tflops.values()) if self.tflops else 0.0

    @property
    def synthetic_gpu_key(self) -> str:
        """A unique key like ``LOCAL_P100`` we can register in GPU_SPECS."""
        base = self.name.replace(" ", "_").replace("-", "_")
        # Strip vendor prefixes for brevity
        for prefix in ("NVIDIA_", "Tesla_", "GeForce_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
        return f"LOCAL_{base}"


# Loose mapping from "name contains X" to spec-sheet BF16/FP16 TFLOP/s.
# Used purely for the "implied MFU" column in the report.
_KNOWN_SPECS: list[tuple[str, float, float]] = [
    # (substring, spec BF16/FP16 TFLOP/s, $/GPU-hr est.)
    ("B200",     2250.0, 4.50),
    ("H200",      989.0, 2.50),
    ("H100",      989.0, 2.00),
    ("A100",      312.0, 1.20),
    ("L40",       181.0, 1.00),
    ("RTX 4090",  165.0, 0.70),
    ("RTX 4080",  113.0, 0.50),
    ("RTX 3090",   71.0, 0.40),
    ("RTX 3080",   59.0, 0.35),
    ("RTX 3050",   18.0, 0.15),
    ("V100",      125.0, 0.80),
    ("P100",       21.0, 0.30),   # FP16 peak (Pascal has no BF16/TF32)
    ("T4",         65.0, 0.35),
]


def _guess_spec(name: str) -> tuple[Optional[float], Optional[float]]:
    for sub, tfl, price in _KNOWN_SPECS:
        if sub.lower() in name.lower():
            return tfl, price
    return None, None


def have_cuda() -> bool:
    """True only if CUDA is available *and* a trivial kernel actually runs.

    `torch.cuda.is_available()` can return True on a card whose compute
    capability isn't supported by the installed torch build (e.g. an sm_60 Pascal
    with a torch compiled for sm_75+), where every kernel then raises. We verify
    a real elementwise kernel executes so callers (and the gpu-probe tests'
    skip guard) get an honest answer.
    """
    try:
        import torch
        if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
            return False
        x = torch.zeros(8, device="cuda") + 1.0
        torch.cuda.synchronize()
        return bool(float(x.sum().item()) == 8.0)
    except Exception:
        return False


def _supports_dtype(dev_index: int, dtype) -> bool:
    """Quick capability check via a 16×16 matmul."""
    import torch
    try:
        a = torch.zeros((16, 16), device=f"cuda:{dev_index}", dtype=dtype)
        (a @ a).sum().item()
        return True
    except Exception:
        return False


def _benchmark_matmul(dev_index: int, dtype, size: int, iters: int) -> float:
    """Return measured TFLOP/s for an NxN square matmul at the given dtype."""
    import torch
    dev = torch.device(f"cuda:{dev_index}")
    a = torch.randn((size, size), device=dev, dtype=dtype)
    b = torch.randn((size, size), device=dev, dtype=dtype)
    # warm up
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize(dev)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize(dev)
    elapsed_s = start.elapsed_time(end) / 1000.0
    flops = 2.0 * size * size * size * iters
    del a, b, c
    torch.cuda.empty_cache()
    return flops / elapsed_s / 1e12


def _benchmark_hbm(dev_index: int, mb: int = 256, iters: int = 20) -> float:
    """Device-to-device copy bandwidth (GB/s)."""
    import torch
    dev = torch.device(f"cuda:{dev_index}")
    n = (mb * 1024 * 1024) // 4   # fp32 elements
    src = torch.empty(n, device=dev, dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize(dev)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize(dev)
    elapsed_s = start.elapsed_time(end) / 1000.0
    # one read + one write per copy
    bytes_moved = 2.0 * n * 4 * iters
    del src, dst
    torch.cuda.empty_cache()
    return bytes_moved / elapsed_s / 1e9


def _device_usable(dev_index: int) -> bool:
    """Run the cheapest possible kernel; if torch refuses, we can't bench this card."""
    import torch
    try:
        x = torch.zeros(4, device=f"cuda:{dev_index}") + 1
        torch.cuda.synchronize(dev_index)
        del x
        return True
    except Exception:
        return False


def probe_all(matmul_size: int = 2048, matmul_iters: int = 30,
              skip_unusable: bool = True) -> list[GPUProbeResult]:
    """Probe every visible CUDA device. Returns [] if no CUDA.

    Devices that torch can see but can't actually run on (e.g. an sm_60
    Pascal card with a torch built only for sm_75+) are skipped when
    ``skip_unusable=True``.
    """
    if not have_cuda():
        return []
    import torch
    out: list[GPUProbeResult] = []
    n = torch.cuda.device_count()
    for i in range(n):
        if skip_unusable and not _device_usable(i):
            continue
        props = torch.cuda.get_device_properties(i)
        cap = (props.major, props.minor)
        res = GPUProbeResult(
            index=i,
            name=props.name,
            compute_capability=cap,
            total_mem_gb=props.total_memory / (1024 ** 3),
        )
        # FP32 is universal
        try:
            res.tflops["fp32"] = _benchmark_matmul(i, torch.float32, matmul_size, matmul_iters)
        except Exception:
            pass
        # FP16: anything sm_53+
        if cap >= (5, 3) and _supports_dtype(i, torch.float16):
            try:
                res.tflops["fp16"] = _benchmark_matmul(i, torch.float16, matmul_size, matmul_iters)
            except Exception:
                pass
        # BF16: Ampere+ (sm_80+) really, but try anyway
        if cap >= (8, 0) and _supports_dtype(i, torch.bfloat16):
            try:
                res.tflops["bf16"] = _benchmark_matmul(i, torch.bfloat16, matmul_size, matmul_iters)
            except Exception:
                pass
        try:
            res.hbm_bandwidth_gb_s = _benchmark_hbm(i)
        except Exception:
            pass
        spec_tfl, _ = _guess_spec(props.name)
        res.spec_tflops_bf16 = spec_tfl
        if spec_tfl and res.best_tflops:
            res.implied_mfu = min(1.0, res.best_tflops / spec_tfl)
        out.append(res)
    return out


def register_in_gpu_specs(probes: list[GPUProbeResult]) -> list[str]:
    """Insert measured devices into :data:`platform.sim.cluster.GPU_SPECS`.

    Returns the list of newly-registered keys. Price is taken from the closest
    known SKU; if unknown, defaults to $0.50/GPU-hr.
    """
    from .cluster import GPU_SPECS
    keys: list[str] = []
    for p in probes:
        key = p.synthetic_gpu_key
        spec_tfl, price = _guess_spec(p.name)
        GPU_SPECS[key] = {
            "tflops": p.best_tflops or (spec_tfl or 50.0),
            "hbm":    int(round(p.total_mem_gb)),
            "price":  price if price is not None else 0.50,
        }
        keys.append(key)
    return keys


def format_probe_report(probes: list[GPUProbeResult]) -> str:
    if not probes:
        return " (no CUDA devices visible — skipping real-GPU probe)"
    lines = []
    lines.append(" MEASURED LOCAL GPUs")
    lines.append(f"   {'idx':>3s} {'name':<28s} {'sm':>5s} {'mem':>7s}  "
                 f"{'fp32':>7s} {'fp16':>7s} {'bf16':>7s}  {'HBM GB/s':>9s}  {'spec':>7s} {'MFU':>5s}")
    for p in probes:
        sm = f"{p.compute_capability[0]}.{p.compute_capability[1]}"
        mem = f"{p.total_mem_gb:.1f}GB"
        fp32 = f"{p.tflops.get('fp32', 0):.2f}" if 'fp32' in p.tflops else "  -  "
        fp16 = f"{p.tflops.get('fp16', 0):.2f}" if 'fp16' in p.tflops else "  -  "
        bf16 = f"{p.tflops.get('bf16', 0):.2f}" if 'bf16' in p.tflops else "  -  "
        hbm = f"{p.hbm_bandwidth_gb_s:.1f}"
        spec = f"{p.spec_tflops_bf16:.0f}" if p.spec_tflops_bf16 else "  -  "
        mfu = f"{p.implied_mfu*100:.0f}%" if p.implied_mfu else "  -  "
        lines.append(f"   {p.index:>3d} {p.name[:28]:<28s} {sm:>5s} {mem:>7s}  "
                     f"{fp32:>7s} {fp16:>7s} {bf16:>7s}  {hbm:>9s}  {spec:>7s} {mfu:>5s}")
    return "\n".join(lines)
