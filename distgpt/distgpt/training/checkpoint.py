"""Synchronous DCP restart checkpoints with collective, atomic publication.

Schema v2 stores canonical model/optimizer FQNs, rank-local loader/RNG/runtime
state, and a hashed completion manifest. ``step`` always means COMPLETED updates
(and therefore the next zero-based update index). Legacy unmarked checkpoints
are weights-only imports, explicitly opted into; they are never auto-resumed.

Publication requires a shared POSIX filesystem and every world rank. It is not
an asynchronous/object-store protocol. A failed stage is retained for diagnosis
and never selected by ``latest`` or garbage-collected as a valid checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
from typing import Callable, TypeVar
import uuid
import warnings

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)

from ..utils.state import (
    atomic_json, atomic_text, capture_rng_state, fsync_dir, restore_rng_state,
    sha256_file,
)

SCHEMA = "distgpt.checkpoint/v2"
COMPLETE = "COMPLETED.json"
_STEP = re.compile(r"step_[0-9]{9,}\Z")
_T = TypeVar("_T")


def _rank_world() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _collective_call(action: Callable[[], _T], phase: str) -> _T:
    """Propagate ordinary rank-local failures before the next collective.

    Process death/transport failure still relies on the process-group timeout;
    it cannot be converted into a successful checkpoint by this helper.
    """
    result = None
    error = None
    try:
        result = action()
    except (Exception, dcp.CheckpointException) as exc:
        error = f"{type(exc).__name__}: {exc}"
    _, world = _rank_world()
    errors = [error]
    if world > 1:
        errors = [None] * world
        dist.all_gather_object(errors, error)
    if any(errors):
        details = "; ".join(f"rank {i}: {e}" for i, e in enumerate(errors) if e)
        raise RuntimeError(f"checkpoint {phase} failed: {details}")
    return result


def _as_list(optimizers) -> list[torch.optim.Optimizer]:
    opts = list(optimizers) if isinstance(optimizers, (tuple, list)) else [optimizers]
    if not opts or any(not isinstance(o, torch.optim.Optimizer) for o in opts):
        raise ValueError("at least one optimizer is required for a training checkpoint")
    parameters = [id(p) for o in opts for g in o.param_groups for p in g["params"]]
    if len(parameters) != len(set(parameters)):
        raise ValueError("checkpoint optimizers must own disjoint parameter sets")
    return opts


def _optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     *, materialize: bool = False) -> dict:
    """Use the public DCP FQN API without a hidden optimizer update.

    DCP materializes an empty optimizer by taking a zero-LR step. We isolate
    that temporary state and restore gradients/state afterwards. On save an
    actually empty optimizer remains empty; on load the allocated tensors are
    the destination for saved moments. Partial (e.g. unused expert) states are
    filtered using the saved schema, not silently initialized on restore.
    """
    if optimizer.state and not materialize:
        return get_optimizer_state_dict(model, optimizer)
    state = optimizer.state
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    grads = [p.grad for p in parameters]
    lrs = [g["lr"] for g in optimizer.param_groups]
    optimizer.state = defaultdict(dict)
    try:
        for p in parameters:
            p.grad = None
        result = get_optimizer_state_dict(model, optimizer)
        if not materialize:
            result["state"] = {}
        return result
    finally:
        optimizer.state = state
        for p, grad in zip(parameters, grads):
            p.grad = grad
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr


def _optimizer_schema(optimizer: torch.optim.Optimizer, state: dict) -> dict:
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "groups": [g["params"] for g in state["param_groups"]],
        "group_keys": [sorted(g) for g in state["param_groups"]],
        "state_fields": {name: sorted(values) for name, values in state["state"].items()},
    }


def _model_schema(state: dict) -> dict:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()}


def _step_name(step: int) -> str:
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint step must be a nonnegative completed-update count")
    return f"step_{step:09d}"


def _canonical_step(name: str) -> int | None:
    if not _STEP.fullmatch(name):
        return None
    step = int(name[5:])
    return step if _step_name(step) == name else None


def _read_checkpoint(path: Path, *, verify_hashes: bool) -> dict:
    """Validate the publication manifest before DCP mutates any live state."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"checkpoint is not a real directory: {path}")
    if not (path / COMPLETE).is_file():
        raise ValueError("incomplete/legacy checkpoint: no completion marker; legacy weights require allow_legacy=True")
    marker = json.loads((path / COMPLETE).read_text())
    if marker.get("schema") != SCHEMA:
        raise ValueError(f"unsupported checkpoint schema: {marker.get('schema')!r}")
    files = marker["files"]
    if not isinstance(files, dict) or not {"meta.json", ".metadata"} <= files.keys():
        raise ValueError("completion manifest is missing DCP metadata")
    for name, expected in files.items():
        if Path(name).name != name or name in (".", "..", COMPLETE):
            raise ValueError(f"unsafe checkpoint manifest member: {name!r}")
        member = path / name
        if member.is_symlink() or not member.is_file() or member.stat().st_size != expected["size"]:
            raise ValueError(f"checkpoint member missing or truncated: {name}")
        if (verify_hashes or name == "meta.json") and sha256_file(member) != expected["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {name}")
    meta = json.loads((path / "meta.json").read_text())
    step = meta["next_step"]
    if (meta.get("schema") != SCHEMA or type(step) is not int or step < 0
            or marker.get("next_step") != step or path.name != _step_name(step)):
        raise ValueError("checkpoint step/schema mismatch")
    if type(meta["world_size"]) is not int or meta["world_size"] < 1:
        raise ValueError("invalid checkpoint world size")
    for rank in range(meta["world_size"]):
        if f"rank_{rank:05d}.json" not in files:
            raise ValueError(f"checkpoint is missing rank {rank} state")
    return meta


class CheckpointManager:
    """Collective same-topology full restart, plus explicit weights-only import."""

    def __init__(self, root: str, run_id: str, keep_last: int = 5,
                 topology: dict | None = None):
        if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("run_id must be a single safe directory name")
        if type(keep_last) is not int or keep_last < 1:
            raise ValueError("keep_last must be a positive integer; best is retained in addition")
        self.root = os.path.join(root, run_id, "ckpts")
        self.keep_last = keep_last
        _, world = _rank_world()
        self.topology = topology or {"dp": world, "tp": 1, "pp": 1}
        self.last_runtime_state: dict = {}
        self.last_extra: dict = {}
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def _path(self, step: int) -> str:
        return str(Path(self.root) / _step_name(step))

    def save(self, model, optimizer, loader, step: int, extra: dict | None = None,
             *, runtime_state: dict | None = None, is_best: bool = False) -> str:
        """Publish state after exactly ``step`` updates. Never overwrite a step."""
        rank, world = _rank_world()
        destination = Path(self._path(step))
        opts = _as_list(optimizer)

        def prepare() -> tuple[dict, list[dict], dict]:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"refusing to overwrite checkpoint {destination}")
            model_state = get_model_state_dict(model)
            optim_states = [_optimizer_state(model, opt) for opt in opts]
            meta = {"schema": SCHEMA, "next_step": step, "world_size": world,
                    "topology": self.topology, "model": _model_schema(model_state),
                    "optimizers": [_optimizer_schema(o, sd) for o, sd in zip(opts, optim_states)],
                    "extra": extra or {}}
            # Reject non-JSON metadata on every rank before entering DCP.
            json.dumps([meta, runtime_state, loader.state_dict()], allow_nan=False)
            return model_state, optim_states, meta

        model_state, optim_states, meta = _collective_call(prepare, "prepare")
        names = [f".staging-{destination.name}-{uuid.uuid4().hex}" if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(names, src=0)
        staging = Path(self.root) / names[0]
        _collective_call(lambda: staging.mkdir() if rank == 0 else None, "create staging")
        state = {"model": model_state, **{f"optim_{i}": sd for i, sd in enumerate(optim_states)}}
        _collective_call(lambda: dcp.save(state, checkpoint_id=staging), "DCP write")
        rank_state = {"schema": SCHEMA, "rank": rank, "next_step": step,
                      "loader": loader.state_dict(), "rng": capture_rng_state(),
                      "runtime": runtime_state or {}}
        _collective_call(lambda: atomic_json(staging / f"rank_{rank:05d}.json", rank_state), "rank state")

        def publish() -> None:
            if rank != 0:
                return
            atomic_json(staging / "meta.json", meta)
            members = list(staging.iterdir())
            if any(not f.is_file() or f.is_symlink() for f in members):
                raise ValueError("unexpected checkpoint staging member")
            manifest = {f.name: {"size": f.stat().st_size, "sha256": sha256_file(f)} for f in members}
            atomic_json(staging / COMPLETE, {"schema": SCHEMA, "next_step": step, "files": manifest})
            fsync_dir(staging)
            # Same-filesystem directory rename is the only publication point.
            if destination.exists():
                raise FileExistsError(f"checkpoint appeared during save: {destination}")
            os.rename(staging, destination)
            fsync_dir(Path(self.root))
            if is_best:
                self.mark_best(step)
            self._gc()

        _collective_call(publish, "publication")
        return str(destination)

    def load(self, model, optimizer, loader, step: int | str = "latest", *,
             runtime_state: dict | None = None, expected_runtime: dict | None = None) -> int:
        """Restore a complete same-topology job; return the NEXT update index."""
        selected = self.discover_latest() if step == "latest" else int(step)
        if selected is None:
            _collective_call(self.reject_legacy_resume, "legacy restart validation")
            return 0
        path = Path(self._path(selected))
        rank, world = _rank_world()
        opts = _as_list(optimizer)

        def prepare() -> tuple[dict, dict, dict, dict]:
            meta = _read_checkpoint(path, verify_hashes=True)
            if meta["world_size"] != world or meta["topology"] != self.topology:
                raise ValueError("full restart across topology changes is unsupported; use a weights-only warm start")
            saved = meta["optimizers"]
            if len(saved) != len(opts):
                raise ValueError("optimizer count changed; refusing to discard or initialize optimizer history")
            rank_state = json.loads((path / f"rank_{rank:05d}.json").read_text())
            if (rank_state["schema"] != SCHEMA or rank_state["rank"] != rank
                    or rank_state["next_step"] != meta["next_step"]):
                raise ValueError("rank-state identity mismatch")
            if (rank_state["rng"]["cuda"] is not None) != torch.cuda.is_available():
                raise ValueError("RNG device changed; use weights-only warm start")
            if expected_runtime is not None:
                for key, value in expected_runtime.items():
                    if rank_state["runtime"].get(key) != value:
                        raise ValueError(f"training restart contract changed: {key}")
            validator = getattr(loader, "validate_state_dict", None)
            if validator is not None:
                validator(rank_state["loader"])
            model_state = get_model_state_dict(model)
            if _model_schema(model_state) != meta["model"]:
                raise ValueError("model names, shapes, or parameter dtype changed")
            state = {"model": model_state}
            for i, (opt, schema) in enumerate(zip(opts, saved)):
                destination = _optimizer_state(model, opt, materialize=True)
                current = _optimizer_schema(opt, destination)
                for field in ("class", "groups", "group_keys"):
                    if schema[field] != current[field]:
                        raise ValueError(f"optimizer {i} {field} changed")
                destination["state"] = {
                    name: {key: destination["state"][name][key] for key in fields}
                    for name, fields in schema["state_fields"].items()
                }
                state[f"optim_{i}"] = destination
            return state, meta, rank_state, rank_state["runtime"]

        state, meta, rank_state, runtime = _collective_call(prepare, "load validation")
        _collective_call(lambda: dcp.load(state, checkpoint_id=path), "DCP read")

        def restore() -> None:
            set_model_state_dict(model, state["model"])
            for i, opt in enumerate(opts):
                set_optimizer_state_dict(model, opt, state[f"optim_{i}"])
            loader.load_state_dict(rank_state["loader"])
            restore_rng_state(rank_state["rng"])

        _collective_call(restore, "restore")
        self.last_runtime_state = runtime
        self.last_extra = meta["extra"]
        if runtime_state is not None:
            runtime_state.clear()
            runtime_state.update(runtime)
        return meta["next_step"]

    @staticmethod
    def load_weights_only(model, ckpt_dir: str, *, allow_legacy: bool = False) -> None:
        """Warm-start weights; never pretend this restores optimizer/data/RNG."""
        path = Path(ckpt_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"warm-start checkpoint not found: {ckpt_dir!r}; load_ckpt must name a step directory")
        if (path / COMPLETE).exists():
            _collective_call(lambda: _read_checkpoint(path, verify_hashes=True), "warm-start validation")
        elif not allow_legacy:
            raise ValueError("unmarked legacy/incomplete checkpoint; set allow_legacy=True only for a trusted legacy weights import")
        elif not (path / ".metadata").is_file():
            raise ValueError("legacy checkpoint has no DCP metadata")
        else:
            warnings.warn("legacy weights-only import: completeness cannot be verified; optimizer, data and RNG are reset", stacklevel=2)
        state = {"model": get_model_state_dict(model)}
        _collective_call(lambda: dcp.load(state, checkpoint_id=path), "weights read")
        set_model_state_dict(model, state["model"])

    def _complete_steps(self) -> list[int]:
        steps = []
        for path in Path(self.root).iterdir():
            step = _canonical_step(path.name)
            if step is None or path.is_symlink() or not (path / COMPLETE).is_file():
                continue
            try:
                _read_checkpoint(path, verify_hashes=False)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.warn(f"ignoring invalid checkpoint {path.name}: {exc}", stacklevel=2)
                continue
            steps.append(step)
        return sorted(steps)

    def latest(self) -> int | None:
        """Select only published, structurally complete checkpoints."""
        steps = self._complete_steps()
        return steps[-1] if steps else None

    def discover_latest(self) -> int | None:
        """Collectively agree on visibility before any rank enters DCP load."""
        selected = _collective_call(self.latest, "latest discovery")
        _, world = _rank_world()
        if world > 1:
            choices = [None] * world
            dist.all_gather_object(choices, selected)
            if any(value != choices[0] for value in choices):
                raise RuntimeError(f"checkpoint visibility differs across ranks: {choices}")
        return selected

    def reject_legacy_resume(self) -> None:
        """Prevent a silent cold start when only unmarked legacy DCP state exists."""
        legacy = [p for p in Path(self.root).iterdir()
                  if _canonical_step(p.name) is not None and not p.is_symlink()
                  and (p / ".metadata").is_file() and not (p / COMPLETE).is_file()]
        if legacy:
            raise ValueError("unmarked legacy/incomplete checkpoints exist; full resume is unsafe. Use a new run_id and an explicit legacy weights-only warm start")

    def mark_best(self, step: int) -> None:
        """Pin a published checkpoint; call on rank zero before retention GC."""
        if _rank_world()[0] != 0:
            return
        _read_checkpoint(Path(self._path(step)), verify_hashes=False)
        atomic_text(Path(self.root) / "best.txt", _step_name(step) + "\n")

    def _gc(self) -> None:
        """Keep newest K plus pinned best; never touch staging/unknown/symlink paths."""
        steps = self._complete_steps()
        pointer = Path(self.root) / "best.txt"
        best = None
        if pointer.exists() or pointer.is_symlink():
            if pointer.is_symlink():
                raise ValueError("best checkpoint pointer must not be a symlink")
            best = _canonical_step(pointer.read_text().strip())
            if best not in steps:
                raise ValueError("best checkpoint pointer is invalid; refusing garbage collection")
        for step in steps[:-self.keep_last]:
            if step != best:
                shutil.rmtree(self._path(step))
        fsync_dir(Path(self.root))
