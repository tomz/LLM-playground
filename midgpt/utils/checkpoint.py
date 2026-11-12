"""Collective, same-topology training checkpoints at completed-step boundaries.

Every rank must call save/load. Only rank zero publishes the shared model and
optimizer state, while RNG, sampling, scaler, and controller state are rank-local.
A checkpoint is trusted pickle input, not a safe format for untrusted weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from . import save_ckpt

SCHEMA = "midgpt.training/v2"


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _topology(device: str) -> dict[str, Any]:
    _, world = _rank_world()
    return {
        "world_size": world,
        "device_type": torch.device(device).type,
        "backend": str(dist.get_backend()) if world > 1 else None,
    }


def _device_index(device: str) -> int | None:
    dev = torch.device(device)
    if dev.type == "cuda":
        return dev.index if dev.index is not None else torch.cuda.current_device()
    return dev.index


def _collective_errors(error: str | None, operation: str) -> None:
    """Make ordinary I/O/validation failures fail on all ranks, not hang peers."""
    _, world = _rank_world()
    errors = [error]
    if world > 1:
        errors = [None] * world
        dist.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message]
    if failures:
        raise RuntimeError(f"checkpoint {operation} failed: " + "; ".join(failures))


def checkpoint_exists(path: str) -> bool:
    """Use rank zero's existence decision on every rank (collective in DDP)."""
    rank, world = _rank_world()
    found = [Path(path).is_file() if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(found, src=0)
    return bool(found[0])


def _resume_contract(cfg: dict) -> dict:
    """Knobs that affect subsequent updates; max_iters may extend a run."""
    optim = dict(cfg.get("optim", {}))
    optim.pop("max_iters", None)
    return {
        key: cfg.get(key) for key in (
            "dataset", "tokenizer", "seed", "dtype", "compile", "grad_checkpoint",
            "fused_ce", "model", "stability",
        )
    } | {
        "optim": optim,
        "train": {key: cfg.get("train", {}).get(key) for key in (
            "micro_batch", "grad_accum", "eval_interval", "eval_iters",
        )},
    }


def save_training_checkpoint(
    path: str, *, model: torch.nn.Module, optimizers: list,
    scaler: torch.amp.GradScaler, generator: torch.Generator, device: str,
    iteration: int, cfg: dict, best_val: float, control: dict | None = None,
) -> None:
    """Gather rank state, then atomically publish one full training snapshot.

    Call only after a complete optimizer step, controller observation, and any
    scheduled evaluation. Partial gradient-accumulation restart is not supported.
    Publication failures are reported to all participating ranks.
    """
    rank, world = _rank_world()
    local = None
    error = None
    try:
        local = {
            "rank": rank,
            "device_index": _device_index(device),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.startswith("cuda") else None,
            "mps_rng_state": torch.mps.get_rng_state() if device == "mps" else None,
            "gen_state": generator.get_state(),
            "scaler_enabled": scaler.is_enabled(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "control": control or {},
            "best_val": best_val,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _collective_errors(error, "capture")
    rank_states = [local] if rank == 0 else None
    if world > 1:
        rank_states = [None] * world if rank == 0 else None
        dist.gather_object(local, rank_states, dst=0)

    error = None
    if rank == 0:
        try:
            state = {
                "checkpoint_schema": SCHEMA,
                "topology": _topology(device),
                "rank_states": rank_states,
                "model": model.state_dict(),
                "optims": [opt.state_dict() for opt in optimizers],
                "optimizer_types": [type(opt).__name__ for opt in optimizers],
                "iter": iteration, "cfg": cfg, "best_val": best_val,
                "resume_contract": _resume_contract(cfg),
                # Preserve the old single-rank inspection API. New resume code
                # always selects rank_states[rank], never these rank-zero aliases.
                **{key: local[key] for key in (
                    "rng_state", "cuda_rng_state", "gen_state", "scaler", "control",
                )},
            }
            save_ckpt(path, state)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    _collective_errors(error, "publication")


def _validate(
    state: dict, *, optimizers: list, scaler: torch.amp.GradScaler,
    device: str, cfg: dict | None,
) -> tuple[dict, list]:
    rank, world = _rank_world()
    schema = state.get("checkpoint_schema")
    if schema is None:
        # Old checkpoints have one RNG stream with no rank identity. They must
        # never be copied into every DDP rank or treated as a topology migration.
        saved_world = state.get("world_size", state.get("cfg", {}).get("world_size", 1))
        if world != 1 or saved_world != 1 or state.get("topology", {}).get("world_size", 1) != 1:
            raise ValueError("legacy checkpoints support single-rank resume only; DDP rank state is unavailable")
        local = state
    else:
        if schema != SCHEMA:
            raise ValueError(f"unsupported checkpoint schema: {schema!r}")
        if state.get("topology") != _topology(device):
            raise ValueError("checkpoint topology differs; full restart requires the same world size, device type and backend")
        ranks = state.get("rank_states", [])
        if len(ranks) != world or [item.get("rank") for item in ranks] != list(range(world)):
            raise ValueError("missing or misordered per-rank checkpoint states")
        local = ranks[rank]
        if local.get("device_index") != _device_index(device):
            raise ValueError("checkpoint rank-to-device mapping differs")
        if local.get("scaler_enabled") != scaler.is_enabled():
            raise ValueError("checkpoint GradScaler mode differs")
        if cfg is not None and state.get("resume_contract") != _resume_contract(cfg):
            raise ValueError("training configuration changed across resume (only run length/log/output settings may change)")
        if state.get("optimizer_types") != [type(opt).__name__ for opt in optimizers]:
            raise ValueError("checkpoint optimizer types differ")
        for key in ("rng_state", "gen_state", "control", "best_val"):
            if key not in local:
                raise ValueError(f"missing rank-local {key}")
        if device.startswith("cuda") and local.get("cuda_rng_state") is None:
            raise ValueError("missing rank-local CUDA RNG state")
        if device == "mps" and local.get("mps_rng_state") is None:
            raise ValueError("missing rank-local MPS RNG state")
    opt_states = state.get("optims")
    if opt_states is None and "optim" in state:
        opt_states = [state["optim"]]
    if not isinstance(opt_states, list) or len(opt_states) != len(optimizers):
        raise ValueError("checkpoint optimizer count differs or optimizer state is missing")
    if scaler.is_enabled() and not local.get("scaler"):
        raise ValueError("checkpoint GradScaler state is missing")
    return local, opt_states


def load_training_checkpoint(
    path: str, *, model: torch.nn.Module, optimizers: list,
    scaler: torch.amp.GradScaler, generator: torch.Generator, device: str,
    cfg: dict | None = None,
) -> dict:
    """Restore a trusted checkpoint collectively and return step/control state.

    Validate topology and update-affecting configuration before mutating state.
    Legacy single-optimizer/single-rank checkpoints remain readable; legacy DDP
    and topology changes are rejected rather than cloning rank zero's randomness.
    """
    error = None
    state = local = opt_states = None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        local, opt_states = _validate(
            state, optimizers=optimizers, scaler=scaler, device=device, cfg=cfg,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _collective_errors(error, "validation")

    error = None
    try:
        model.load_state_dict(state["model"])
        for opt, saved in zip(optimizers, opt_states):
            opt.load_state_dict(saved)
        if local.get("scaler") is not None:
            scaler.load_state_dict(local["scaler"])
        # Restore RNG last: construction/loading must not perturb the next draw.
        if local.get("rng_state") is not None:
            torch.set_rng_state(local["rng_state"].to("cpu", dtype=torch.uint8))
        if device.startswith("cuda") and local.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(local["cuda_rng_state"].to("cpu", dtype=torch.uint8), device)
        if device == "mps" and local.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(local["mps_rng_state"].to("cpu", dtype=torch.uint8))
        if local.get("gen_state") is not None:
            generator.set_state(local["gen_state"].to("cpu", dtype=torch.uint8))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _collective_errors(error, "restore")
    return {
        "next_iter": state["iter"] + 1,
        "best_val": local.get("best_val", state.get("best_val", float("inf"))),
        "control": local.get("control", {}),
    }
