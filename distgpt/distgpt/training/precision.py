"""FP8 precision via NVIDIA Transformer Engine — optional, lazy-resolved.

What this gives you
-------------------
On Hopper (sm_90) and Blackwell (sm_100+) GPUs with Transformer Engine
installed, wrapping the forward in `autocast_fp8_context(recipe)` lets the
GEMMs inside `te.Linear` / `te.LayerNormLinear` run in FP8 with automatic
amp-scale handling. Activations stay in bf16, gradients in fp32 — so the
optimizer state is unchanged but the matmuls run at ~2× the throughput of
bf16 on Hopper, ~4× on Blackwell with NVFP4 (when available).

What this does NOT do
---------------------
The distgpt model is built from plain `nn.Linear` layers, not `te.Linear`.
The autocast wrapper still applies because TE's `fp8_autocast` patches
common ops via dispatch, but to *realise* the throughput win you'd swap
the linears in `model/transformer.py` for their TE equivalents (out of
scope for this PR; the autocast wrapper alone is a no-op for pure
`nn.Linear` modules and the model just trains in bf16). The point of
landing this now is to:

  * have a single configuration point (`train.fp8`) for users who DO
    install TE and swap their linears,
  * give a clean ImportError + helpful message when they don't,
  * keep the trainer pluggable without scattering try/except across it.

When TE is missing or the device doesn't support FP8, the context manager
falls back to a no-op `nullcontext` so the training loop is unconditional.

Recipe choice
-------------
TE exposes two recipes: `e4m3` (best accuracy, all-FP8) and `hybrid` (E5M2
for the backward pass, E4M3 forward — better dynamic range). We expose
both as string knobs; the default for new runs is `hybrid` which matches
TE's own recommendation for transformer training.

Tests
-----
* `tests/test_precision.py` checks the import-guard / fallback behaviour
  without requiring TE. The actual numerics test is gated on a Hopper+TE
  machine and skipped here.
"""
from __future__ import annotations
import contextlib
import warnings


# Compute capabilities that natively support FP8 in HW (TE will refuse to
# run otherwise). Hopper = sm_90, Blackwell datacenter = sm_100, Blackwell
# consumer (RTX 50xx) = sm_120 — all support FP8.
_FP8_CAPABLE_MAJORS = {9, 10, 12}


def device_supports_fp8(device: str = "cuda") -> bool:
    """True if the given CUDA device's compute capability supports FP8 GEMMs.

    Returns False on CPU, on older GPUs (Ampere, Ada), and when CUDA is not
    available at all. Lets the caller decide whether to plumb fp8 on/off
    without an explicit try/except.
    """
    if not str(device).startswith("cuda"):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major in _FP8_CAPABLE_MAJORS
    except Exception:
        return False


def resolve_fp8_recipe(setting: str, device: str, dtype):
    """Validate a user-facing `train.fp8` setting and return a normalised
    recipe spec the autocast context manager understands.

    `setting` is one of:
      * "off" — disable FP8, return None. Always safe.
      * "e4m3" — pure E4M3 FP8. Requires TE + FP8-capable GPU.
      * "hybrid" — TE's HYBRID recipe (E5M2 backward, E4M3 forward).
        TE's own recommendation for transformer training.

    Returns:
      None if FP8 should not be used (setting=="off", or HW doesn't support
      it and we're falling back). A string {"e4m3","hybrid"} otherwise.

    Raises:
      ValueError if `setting` is unknown.

    This function deliberately does NOT import transformer_engine — it only
    decides whether the autocast context will try to. The actual import is
    deferred to `autocast_fp8_context`.
    """
    if setting in (None, "off", False):
        return None
    if setting not in ("e4m3", "hybrid"):
        raise ValueError(
            f"unknown train.fp8 setting {setting!r}; "
            "expected one of: 'off', 'e4m3', 'hybrid'"
        )
    # bf16 master dtype is required for FP8 (TE accumulates in higher
    # precision; fp16 master is not supported by the current recipe API).
    import torch
    if dtype not in (torch.bfloat16,):
        warnings.warn(
            f"train.fp8={setting!r} but dtype={dtype} — FP8 needs bfloat16 "
            f"master dtype. Falling back to no-FP8."
        )
        return None
    if not device_supports_fp8(device):
        warnings.warn(
            f"train.fp8={setting!r} but device={device} does not support "
            f"FP8 (need sm_90+). Falling back to no-FP8."
        )
        return None
    return setting


def autocast_fp8_context(recipe: str | None):
    """Build the FP8 autocast context manager (or nullcontext if disabled).

    Lazy-imports transformer_engine only when actually needed, so distgpt's
    base deps stay torch-only. If TE is missing despite a valid recipe being
    requested, we raise ImportError with a clear pointer to the install.
    """
    if recipe is None:
        return contextlib.nullcontext()
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import (
            DelayedScaling, Format,
        )
    except ImportError as e:
        raise ImportError(
            f"train.fp8={recipe!r} requested but transformer_engine is not "
            "installed. Install with `pip install transformer-engine` (NVIDIA "
            "wheels: https://github.com/NVIDIA/TransformerEngine). Set "
            "train.fp8: off to disable."
        ) from e
    fmt = Format.E4M3 if recipe == "e4m3" else Format.HYBRID
    te_recipe = DelayedScaling(fp8_format=fmt, amax_history_len=16,
                                 amax_compute_algo="max")
    return te.fp8_autocast(enabled=True, fp8_recipe=te_recipe)


def log_fp8_choice(recipe: str | None, dtype) -> None:
    """Print a one-line trainer-startup banner describing the FP8 decision."""
    if recipe is None:
        print(f"[fp8] disabled (dtype={dtype})")
    else:
        print(f"[fp8] enabled: recipe={recipe} (dtype={dtype})")
