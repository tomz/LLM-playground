"""Bounded CPU, equal-token hybrid ablation; run python hybrid_ablation.py --help.

This is an educational architecture analogue, not LFM3 weights or a reproduction.
The default delayed-copy task is a plumbing check, not a language benchmark.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from hybrid import HybridGPT, positive_int, validate_config
from model import GPT, GPTConfig

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationConfig:
    """One fixed budget per architecture per seed; CPU FP32 only."""

    train_tokens: int = 4096
    batch_size: int = 4
    block_size: int = 32
    d_model: int = 32
    n_head: int = 4
    d_ffn: int = 64
    window: int = 8
    kernel_size: int = 3
    eval_sequences: int = 16
    seeds: tuple[int, ...] = (7, 17, 27)
    data_seed: int = 1234
    learning_rate: float = 0.003
    latency_repeats: int = 20
    warmup: int = 3
    threads: int = 1

    def __post_init__(self) -> None:
        for name in ("train_tokens", "batch_size", "block_size", "d_model", "n_head", "d_ffn",
                     "window", "kernel_size", "eval_sequences", "latency_repeats", "warmup", "threads"):
            positive_int(name, getattr(self, name))
        if self.train_tokens % (self.batch_size * self.block_size):
            raise ValueError("train_tokens must be divisible by batch_size * block_size; no budget rounding")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        for seed in (*self.seeds, self.data_seed):
            if type(seed) is not int or not 0 <= seed < 2**63 - 1:
                raise ValueError("seeds must be integers in [0, 2**63-1)")
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float))
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")
        if self.latency_repeats < 10:
            raise ValueError("latency_repeats must be at least 10")
        validate_config(self.model_config(32))

    def model_config(self, vocab_size: int) -> GPTConfig:
        """The two arms share all GPT settings except their sequence mixers."""
        return GPTConfig(vocab_size=vocab_size, block_size=self.block_size, n_layer=3,
                         n_head=self.n_head, n_kv_head=self.n_head, d_model=self.d_model,
                         d_ffn=self.d_ffn, dropout=0.0)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Content identity of the exact CPU int64 examples used by both arms."""
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _synthetic(count: int, length: int, seed: int) -> torch.Tensor:
    """Independent delayed-copy examples with 20% random innovations."""
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(0, 32, (count, length), generator=generator)
    lag = min(7, max(1, length // 2))
    for position in range(lag, length):
        copy = torch.rand(count, generator=generator) < 0.8
        samples[:, position] = torch.where(copy, (samples[:, position - lag] + 1) % 32, samples[:, position])
    return samples


def _byte_windows(data: bytes, count: int, length: int, seed: int) -> torch.Tensor:
    """Sample windows within one partition; no window can cross the split."""
    if len(data) < length:
        raise ValueError(f"each text partition needs at least {length} bytes")
    values = torch.tensor(list(data), dtype=torch.long)
    starts = torch.randint(len(data) - length + 1, (count,), generator=torch.Generator().manual_seed(seed))
    return values[starts[:, None] + torch.arange(length)]


def build_data(cfg: AblationConfig, text: Path | None = None) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
    """Create fixed train/held-out banks independent of model RNG and run order."""
    count, length = cfg.train_tokens // cfg.block_size, cfg.block_size + 1
    if text is None:
        train = _synthetic(count, length, cfg.data_seed)
        heldout = _synthetic(cfg.eval_sequences, length, cfg.data_seed + 1)
        vocab_size = 32
        info = {"kind": "synthetic_delayed_copy", "split": "independent examples and RNG streams",
                "caveat": "Synthetic smoke task, not natural-language quality."}
    else:
        raw = text.read_bytes()
        split = int(len(raw) * 0.9)
        train = _byte_windows(raw[:split], count, length, cfg.data_seed)
        heldout = _byte_windows(raw[split:], cfg.eval_sequences, length, cfg.data_seed + 1)
        vocab_size = 256
        info = {"kind": "local_bytes", "path": str(text), "sha256": hashlib.sha256(raw).hexdigest(),
                "train_bytes": split, "heldout_bytes": len(raw) - split,
                "split": "contiguous 90/10 bytes before sampling; no crossing windows"}
    info.update(train_sha256=tensor_sha256(train), heldout_sha256=tensor_sha256(heldout),
                train_examples=len(train), heldout_examples=len(heldout), vocab_size=vocab_size)
    return train, heldout, vocab_size, info


def tensor_memory(model: GPT, optimizer: torch.optim.Optimizer) -> dict:
    """Exact live tensor payloads, NOT peak memory or allocator/RSS estimates."""
    def size(tensors: list[torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in tensors)

    parameters = list(model.parameters())  # tied embedding is counted once
    cache = list(model._rope_cache) if model._rope_cache is not None else []
    values = {
        "parameter_bytes": size(parameters),
        "gradient_bytes": size([p.grad for p in parameters if p.grad is not None]),
        "optimizer_state_bytes": size([v for state in optimizer.state.values() for v in state.values()
                                      if isinstance(v, torch.Tensor)]),
        "buffer_and_rope_bytes": size(list(model.buffers()) + cache),
    }
    return {**values, "accounted_tensor_bytes": sum(values.values()),
            "cpu_peak_rss_bytes": None, "accelerator_peak_allocated_bytes": None,
            "note": "Live tensor payload after training, excludes activations, workspaces, data, Python and allocator overhead. "
                    "CPU peak RSS is unmeasured; CUDA unused. Dense local mask does not establish a memory saving."}


@torch.inference_mode()
def evaluate(model: GPT, heldout: torch.Tensor, batch_size: int) -> float:
    """Token-weighted NLL on every fixed held-out example, including a tail batch."""
    model.eval()
    loss_sum, tokens = 0.0, 0
    for batch in heldout.split(batch_size):
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        _, loss = model(x, y)
        loss_sum += float(loss) * y.numel()
        tokens += y.numel()
    return loss_sum / tokens


@torch.inference_mode()
def measure_latency(model: GPT, x: torch.Tensor, repeats: int, warmup: int) -> dict:
    """Warm CPU full-sequence forwards; excludes data prep, loss and sampling."""
    model.eval()
    for _ in range(warmup):
        model(x)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        samples.append((time.perf_counter() - start) * 1000)
    return {"kind": "full_sequence_forward_no_kv_cache", "batch_size": x.size(0), "sequence_length": x.size(1),
            "warmup": warmup, "samples_ms": samples, "p50_ms": statistics.median(samples),
            "p95_ms": float(torch.tensor(samples, dtype=torch.float64).quantile(0.95)),
            "stdev_ms": statistics.stdev(samples)}


def run_arm(cfg: AblationConfig, architecture: str, seed: int, train: torch.Tensor,
            heldout: torch.Tensor, vocab_size: int) -> dict:
    """Train exactly the requested tokens and measure one architecture/seed."""
    if architecture not in {"baseline", "hybrid"}:
        raise ValueError("unknown architecture")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model_cfg = cfg.model_config(vocab_size)
        model = (GPT(model_cfg) if architecture == "baseline" else
                 HybridGPT(model_cfg, window=cfg.window, kernel_size=cfg.kernel_size))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        model.train()
        trained_tokens = 0
        start = time.perf_counter()
        for batch in train.split(cfg.batch_size):
            x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in {architecture} seed={seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            trained_tokens += y.numel()
        train_wall_s = time.perf_counter() - start
        if trained_tokens != cfg.train_tokens:
            raise ValueError("training bank did not match the exact token budget")
        val_nll = evaluate(model, heldout, cfg.batch_size)
        if not math.isfinite(val_nll):
            raise RuntimeError("nonfinite held-out loss")
        latency = measure_latency(model, heldout[:cfg.batch_size, :-1].contiguous(),
                                  cfg.latency_repeats, cfg.warmup)
        return {"architecture": architecture, "seed": seed,
                "mixers": ["global"] * 3 if architecture == "baseline" else list(model.mixers),
                "parameters": model.num_params(non_embedding=False), "trained_tokens": trained_tokens,
                "train_steps": len(train) // cfg.batch_size, "train_wall_s": train_wall_s,
                "val_nll": val_nll, "val_bits_per_token": val_nll / math.log(2),
                "eval_tokens": len(heldout) * cfg.block_size,
                "latency": latency, "memory": tensor_memory(model, optimizer)}


def _git(*args: str) -> str | None:
    """Best-effort reproducibility metadata without requiring a git checkout."""
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_ablation(cfg: AblationConfig, text: Path | None = None) -> dict:
    """Run paired seeds in alternating order and retain every arm in JSON."""
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cfg.threads)
        train, heldout, vocab_size, data_info = build_data(cfg, text)
        results = []
        for index, seed in enumerate(cfg.seeds):
            order = ("baseline", "hybrid") if index % 2 == 0 else ("hybrid", "baseline")
            for architecture in order:
                LOG.info("%s seed=%d tokens=%d", architecture, seed, cfg.train_tokens)
                attempted_at = time.perf_counter()
                try:
                    row = run_arm(cfg, architecture, seed, train, heldout, vocab_size)
                    row["status"] = "completed"
                except Exception as exc:
                    # No inferred zero costs or survivor-only aggregate after a crash.
                    row = {"architecture": architecture, "seed": seed, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}"[:1000],
                           "trained_tokens": None, "train_steps": None, "val_nll": None,
                           "requested_train_tokens": cfg.train_tokens}
                row["attempt_wall_s"] = time.perf_counter() - attempted_at
                results.append(row)
        summary = {}
        for architecture in ("baseline", "hybrid"):
            arms = [row for row in results if row["architecture"] == architecture]
            losses = [row["val_nll"] for row in arms if row["status"] == "completed"]
            complete = len(losses) == len(cfg.seeds)
            summary[architecture] = {
                "seeds": len(cfg.seeds), "completed_seeds": len(losses),
                "failed_seeds": len(arms) - len(losses),
                "attempt_wall_s": sum(row["attempt_wall_s"] for row in arms),
                "val_nll_mean": statistics.mean(losses) if complete else None,
                "val_nll_stdev": statistics.stdev(losses) if complete and len(losses) > 1 else None,
            }
        root = Path(__file__).parent
        return {"schema_version": 1, "config": dataclasses.asdict(cfg), "data": data_info,
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "platform": platform.platform(), "machine": platform.machine(),
                                "processor": platform.processor(), "device": "cpu", "dtype": "float32",
                                "threads": torch.get_num_threads(), "git_revision": _git("rev-parse", "HEAD"),
                                "git_dirty": bool(status) if (status := _git("status", "--porcelain")) is not None else None,
                                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                                  for name in ("model.py", "hybrid.py", "hybrid_ablation.py")}},
                "method": "Equal tokens/data/optimizer/width/depth; not equal parameters, FLOPs or wall time. "
                          "Paired fixed seeds, alternating arm order; no tuning or checkpoint selection. "
                          "Latency is warm full-sequence forward, not cached decode. Educational analogue only.",
                "results": results, "summary": summary}
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Run a small CPU experiment without downloads, APIs, or protected harness changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = AblationConfig()
    for name in ("train_tokens", "batch_size", "block_size", "d_model", "n_head", "d_ffn", "window",
                 "kernel_size", "eval_sequences", "data_seed", "latency_repeats", "warmup", "threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.seeds)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--text", type=Path, help="optional local corpus; byte-level contiguous 90/10 split")
    parser.add_argument("--out", type=Path, default=Path("out/hybrid-ablation.json"))
    args = vars(parser.parse_args())
    text, out = args.pop("text"), args.pop("out")
    args["seeds"] = tuple(args["seeds"])
    try:
        cfg = AblationConfig(**args)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_ablation(cfg, text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOG.info("Results: %s", out)


if __name__ == "__main__":
    main()
