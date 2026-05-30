"""Training & inference numerics: a single place to control precision.

The blueprint's discipline (docs/03 §"Implementation discipline", docs/04
§"Numerics") is that *all* compute flows through one precision policy so we can
swap bf16 → FP8 (Transformer Engine) → NVFP4 globally without touching model
code. This module provides that policy as a context manager plus capability
detection, so the same code runs:

  * on CPU / old GPUs  → fp32 (no autocast), for tests today;
  * on Ampere/Hopper   → bf16 autocast;
  * on Hopper/Blackwell with Transformer Engine → FP8 autocast (te.fp8_autocast)
    wrapping the bf16 autocast, when `transformer_engine` is installed.

When the real GPUs arrive, set `precision="fp8"` and (optionally) install
Transformer Engine; nothing else changes. Until then `fp8` transparently falls
back to bf16/fp32 with a one-time warning, so the code path is exercised now.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass

import torch

_VALID = ("fp32", "bf16", "fp16", "fp8", "nvfp4")
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg, stacklevel=3)


def cuda_supports_bf16() -> bool:
    return bool(torch.cuda.is_available()) and torch.cuda.is_bf16_supported()


def has_transformer_engine() -> bool:
    from importlib.util import find_spec
    return find_spec("transformer_engine") is not None


def resolve_precision(requested: str) -> tuple[str, str]:
    """Map a requested precision to (autocast_dtype_name, effective_backend).

    Returns the precision we will *actually* run and a short backend tag, after
    accounting for hardware/library availability. This is the single source of
    truth other code consults.
    """
    if requested not in _VALID:
        raise ValueError(f"unknown precision {requested!r}; valid: {_VALID}")

    if requested in ("fp8", "nvfp4"):
        if has_transformer_engine() and cuda_supports_bf16():
            return ("bf16", "transformer_engine")  # TE does the fp8 matmuls under bf16 autocast
        if cuda_supports_bf16():
            _warn_once(requested,
                       f"precision={requested!r} requested but Transformer Engine "
                       "not available; falling back to bf16 autocast.")
            return ("bf16", "bf16_fallback")
        _warn_once(requested,
                   f"precision={requested!r} requested but no bf16-capable CUDA "
                   "device; falling back to fp32 (CPU/old GPU).")
        return ("fp32", "fp32_fallback")

    if requested == "bf16":
        if cuda_supports_bf16():
            return ("bf16", "bf16")
        _warn_once("bf16", "bf16 requested but unsupported here; using fp32.")
        return ("fp32", "fp32_fallback")

    if requested == "fp16":
        if torch.cuda.is_available():
            return ("fp16", "fp16")
        _warn_once("fp16", "fp16 requested on CPU; using fp32.")
        return ("fp32", "fp32_fallback")

    return ("fp32", "fp32")


@dataclass
class PrecisionPolicy:
    """Resolved precision policy. Build once, reuse for every forward."""

    requested: str = "bf16"
    autocast_dtype: str = "fp32"
    backend: str = "fp32"

    @classmethod
    def create(cls, requested: str = "bf16") -> "PrecisionPolicy":
        dt, backend = resolve_precision(requested)
        return cls(requested=requested, autocast_dtype=dt, backend=backend)

    @property
    def torch_dtype(self):
        return {"fp32": torch.float32, "bf16": torch.bfloat16,
                "fp16": torch.float16}[self.autocast_dtype]

    @property
    def uses_fp8(self) -> bool:
        return self.backend == "transformer_engine"

    @contextlib.contextmanager
    def autocast(self, fp8_recipe=None):
        """Context manager wrapping the forward/loss compute.

        Nests Transformer Engine's fp8_autocast inside torch.autocast when TE is
        available; otherwise just torch.autocast (bf16/fp16) or a no-op (fp32).
        """
        if self.autocast_dtype == "fp32":
            yield
            return
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                torch.autocast(device_type=device_type, dtype=self.torch_dtype)
            )
            if self.uses_fp8:
                import transformer_engine.pytorch as te  # type: ignore
                recipe = fp8_recipe or _default_fp8_recipe()
                stack.enter_context(te.fp8_autocast(enabled=True, fp8_recipe=recipe))
            yield


def _default_fp8_recipe():
    """DeepSeek-V3-style FP8 recipe (E4M3 fwd, per-tile scaling). Only imported
    when Transformer Engine is present."""
    from transformer_engine.common.recipe import DelayedScaling, Format  # type: ignore
    return DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16,
                          amax_compute_algo="max")


def autocast_context(precision: str = "bf16"):
    """Convenience: resolve + enter an autocast context in one call."""
    return PrecisionPolicy.create(precision).autocast()
