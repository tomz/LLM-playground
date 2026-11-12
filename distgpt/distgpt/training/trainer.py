"""Validated single-process, DDP, dense CUDA FSDP, and restricted TP paths.

PP and DP×TP remain fail-closed until composition/restart parity is established.
Their lower-level modules remain available for explicit experiments.
Checkpoints are saved only at completed optimizer-update boundaries. Evaluation
uses independent, content-identified shards and cannot consume training data/RNG.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ..data.streaming import StreamingLoader, assert_disjoint
from ..model.config import ModelConfig
from ..model.transformer import GPT
from ..parallel.mesh import ParallelContext, build_mesh, resolve_dimensions
from ..parallel.tensor import apply_tp
from ..parallel.fsdp import apply_fsdp
from ..utils.dist import init as dist_init, destroy as dist_destroy, is_master
from ..utils.logging import Logger
from ..utils.metrics import compute_param_norm, estimate_mfu, peak_tflops_for_device
from ..utils.state import atomic_json, capture_rng_state, restore_rng_state
from .checkpoint import CheckpointManager, _collective_call
from .gradients import clip_grad_norm
from .muon import build_muon_and_adamw
from .optim import build_optimizer, cosine_lr
from .stability import SpikeMonitor, RewindController

LOG = logging.getLogger(__name__)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _maybe_compile(model: torch.nn.Module, enabled: bool, pp_active: bool) -> torch.nn.Module:
    """Optional single-process compile; never silently change precision."""
    if not enabled:
        return model
    if pp_active:
        raise ValueError("torch.compile with pipeline parallelism is unsupported")
    return torch.compile(model, mode="default", fullgraph=False)


def validate_config(cfg: dict, world: int, device_type: str) -> tuple[int, int, int]:
    """Fail before group/model/data allocation for unsupported training modes."""
    parallel, training = cfg["parallel"], cfg["train"]
    dp, tp, pp = resolve_dimensions(world, parallel["dp"], parallel["tp"], parallel["pp"])
    if pp != 1:
        raise NotImplementedError("integrated PP training is disabled: macro-batch, gradient and restart parity are unvalidated; use the lower-level pipeline experiment")
    if dp > 1 and tp > 1:
        raise NotImplementedError("integrated DP×TP training is disabled pending mixed-topology parity")
    if parallel["zero"] not in ("none", "fsdp"):
        raise ValueError("parallel.zero must be 'none' or 'fsdp'")
    if parallel["zero"] == "fsdp" and dp > 1:
        if device_type != "cuda":
            raise NotImplementedError("FSDP trainer requires CUDA; use zero: none for CPU DDP")
        if cfg["optim"].get("optimizer", "adamw") != "adamw":
            raise NotImplementedError("FSDP trainer optimizer parity currently covers AdamW only")
        if parallel.get("activation_ckpt", "none") != "none":
            raise NotImplementedError("FSDP activation-checkpoint integration needs separate parity evidence")
        if type(parallel.get("reshard_after_forward", True)) is not bool:
            raise ValueError("reshard_after_forward must be a boolean")
    dtype = cfg["dtype"]
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype: {dtype!r}")
    if dtype == "float16":
        raise ValueError("float16 training requires a validated GradScaler/restart path; use float32 or CUDA bfloat16")
    if dtype == "bfloat16" and device_type != "cuda":
        raise ValueError("CPU trainer supports float32 only; bfloat16 AMP requires CUDA")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("the selected CUDA device does not support bfloat16")
    if training.get("fp8", "off") not in ("off", False, None):
        raise ValueError("FP8 trainer integration is unvalidated (ordinary Linear layers are not TE-converted); set train.fp8: 'off'")
    if training.get("compile", False) and world > 1:
        raise NotImplementedError("distributed torch.compile integration is gated on parity tests")
    if parallel.get("activation_ckpt", "none") not in ("none", "full", "selective"):
        raise ValueError("unknown activation checkpoint policy")
    model = ModelConfig(**cfg["model"])
    if dp > 1 and (model.moe_enabled or model.mla_enabled or model.mtp_tokens):
        raise NotImplementedError("DDP trainer currently validates dense GQA only; MoE/MLA/MTP remain single-process experiments")
    if tp > 1:
        if tp != 2 or device_type != "cpu" or dtype != "float32":
            raise NotImplementedError("trainer TP currently validates TP=2 on CPU/Gloo float32 only")
        if (model.moe_enabled or model.mla_enabled or model.mtp_tokens or model.qk_norm
                or model.tie_embeddings or model.fused_ce):
            raise NotImplementedError("TP trainer requires dense GQA, untied embeddings, qk_norm=false, mtp_tokens=0 and fused_ce=false")
        if cfg["optim"].get("optimizer", "adamw") != "adamw":
            raise NotImplementedError("TP trainer optimizer parity currently covers AdamW only")
        if any(value % tp for value in (model.n_head, model.n_kv_head, model.vocab_size, model.d_ffn)):
            raise ValueError("TP heads, KV heads, vocabulary and FFN dimensions must divide by tp")
        if parallel.get("sequence_parallel", False) and cfg["data"]["seq_len"] % tp:
            raise ValueError("sequence length must divide by tp for sequence parallelism")
    if model.fused_ce and device_type != "cuda":
        raise ValueError("fused_ce requires CUDA; the CPU reference uses dense cross entropy")
    for key in ("micro_batch", "grad_accum", "log_every"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    for key in ("eval_every", "ckpt_every"):
        if type(training[key]) is not int or training[key] < 0:
            raise ValueError(f"train.{key} must be nonnegative (0 explicitly disables it)")
    if training["eval_every"] and not (cfg["data"].get("val_dir") or cfg["data"].get("val_manifest")):
        raise ValueError("evaluation requires data.val_dir or data.val_manifest; set train.eval_every: 0 to disable it explicitly")
    if training.get("eval_batches", 1) < 1:
        raise ValueError("train.eval_batches must be positive")
    if cfg["data"]["seq_len"] > model.max_seq_len:
        raise ValueError("data.seq_len exceeds model.max_seq_len")
    total = cfg["optim"]["total_steps"]
    if type(total) is not int or total < 1:
        raise ValueError("optim.total_steps must be a positive integer")
    stop = training.get("stop_after", total)
    if type(stop) is not int or not 0 <= stop <= total:
        raise ValueError("train.stop_after must be a completed-update boundary in [0, total_steps]")
    clip = cfg["optim"]["grad_clip"]
    if not math.isfinite(clip) or clip < 0:
        raise ValueError("optim.grad_clip must be finite and nonnegative")
    if cfg["optim"].get("optimizer", "adamw") not in ("adamw", "muon"):
        raise ValueError(f"unknown optim.optimizer: {cfg['optim']['optimizer']!r}; expected 'adamw' or 'muon'")
    return dp, tp, pp


def evaluate(model: torch.nn.Module, loader: StreamingLoader,
             autocast: Callable, topology: ParallelContext, batches: int = 1) -> float:
    """A fixed held-out prefix, without altering cursor, mode, grads, or RNG."""
    cursor, rng, training = loader.state_dict(), capture_rng_state(), model.training
    total = 0.0
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(batches):
                x, y = loader.next_batch()
                with autocast():
                    _, loss = model(x, y)
                total += topology.mean_dp(loss.detach().float()).item()
        return total / batches
    finally:
        loader.load_state_dict(cursor)
        restore_rng_state(rng)
        model.train(training)


def _make_loaders(cfg: dict, topology: ParallelContext, device: str,
                  override: str | None) -> tuple[StreamingLoader, StreamingLoader | None]:
    data = cfg["data"]
    common = dict(seq_len=data["seq_len"], micro_batch=cfg["train"]["micro_batch"],
                  rank=topology.dp_rank, world_size=topology.dp, seed=cfg["seed"], device=device)
    loader = StreamingLoader(override or data.get("dir"), **common,
                              manifest_path=None if override else data.get("manifest"))
    validation = None
    if data.get("val_dir") or data.get("val_manifest"):
        validation = StreamingLoader(data.get("val_dir"), **common,
                                      manifest_path=data.get("val_manifest"))
        assert_disjoint(loader, validation)
    return loader, validation


def _record_manifests(run_dir: Path, loader: StreamingLoader,
                      validation: StreamingLoader | None) -> None:
    if not is_master():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, source in (("train", loader), ("validation", validation)):
        if source is None:
            continue
        path = run_dir / f"{name}_manifest.json"
        if path.exists() and json.loads(path.read_text()) != source.manifest:
            raise ValueError(f"immutable run manifest changed: {path}")
        atomic_json(path, source.manifest)


def train(cfg: dict, data_dir_override: str | None = None) -> None:
    """Run to total_steps (or stop_after) and publish an exact boundary restart."""
    world = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    validate_config(cfg, world, device_type)
    logger = None
    try:
        rank, _, world = dist_init()
        device = f"cuda:{torch.cuda.current_device()}" if device_type == "cuda" else "cpu"
        dtype = DTYPES[cfg["dtype"]]
        mesh, dp, tp, pp = build_mesh(world, **{k: cfg["parallel"][k] for k in ("dp", "tp", "pp")},
                                      device_type=device_type)
        topology = ParallelContext(rank, dp, tp, pp, mesh)
        # Same initialization on every model-parallel rank. Stochastic training
        # later uses a per-DP seed; TP peers keep identical streams.
        torch.manual_seed(cfg["seed"])
        mcfg = ModelConfig(**cfg["model"])
        model = GPT(mcfg, activation_ckpt=cfg["parallel"]["activation_ckpt"]).to(device=device)
        # FP32 parameters/optimizer moments, including CUDA bf16 autocast runs.
        if tp > 1:
            apply_tp(model, mesh["tp"], sequence_parallel=cfg["parallel"].get("sequence_parallel", False))
        fsdp_active = dp > 1 and cfg["parallel"]["zero"] == "fsdp"
        if fsdp_active:
            model = apply_fsdp(model, mesh["dp"], torch.float32,
                               reshard_after_forward=cfg["parallel"].get("reshard_after_forward", True))
        elif dp > 1:
            model = DistributedDataParallel(model, process_group=mesh["dp"].get_group(),
                                             device_ids=[torch.cuda.current_device()] if device_type == "cuda" else None,
                                             broadcast_buffers=False)
        model = _maybe_compile(model, cfg["train"].get("compile", False), pp_active=False)
        optim = cfg["optim"]
        if optim.get("optimizer", "adamw") == "muon":
            optims = build_muon_and_adamw(
                model, muon_lr=optim.get("muon_lr", optim["lr"] * 100), adamw_lr=optim["lr"],
                muon_momentum=optim.get("muon_momentum", 0.95),
                muon_weight_decay=optim.get("muon_weight_decay", 0.0),
                muon_update_scale=optim.get("muon_update_scale"), adamw_betas=optim["betas"],
                weight_decay=optim["weight_decay"], fused=device_type == "cuda")
        else:
            optims = [build_optimizer(model, optim["lr"], optim["betas"], optim["weight_decay"], fused=device_type == "cuda")]
        base_lrs = [[g["lr"] for g in o.param_groups] for o in optims]
        loader, validation = _collective_call(lambda: _make_loaders(cfg, topology, device, data_dir_override), "data setup")
        ckpt = CheckpointManager(cfg["out_dir"], cfg["run_id"],
                                  keep_last=cfg["train"].get("keep_last", 5), topology=topology.dimensions)
        spike, rewind = SpikeMonitor(), RewindController(ckpt)
        contract = {"model": asdict(mcfg), "optimizer": optim,
                    "dtype": cfg["dtype"], "device_type": device_type,
                    "grad_accum": cfg["train"]["grad_accum"],
                    "activation_ckpt": cfg["parallel"]["activation_ckpt"],
                    "zero": cfg["parallel"]["zero"],
                    "reshard_after_forward": cfg["parallel"].get("reshard_after_forward", True),
                    "sequence_parallel": cfg["parallel"].get("sequence_parallel", False),
                    "validation_manifest": validation.manifest_sha256 if validation else None}
        # Normalize tuple/list distinctions to the JSON representation on disk.
        contract = json.loads(json.dumps(contract))
        torch.manual_seed(cfg["seed"] + topology.dp_rank)
        random.seed(cfg["seed"] + topology.dp_rank)
        np.random.seed(cfg["seed"] + topology.dp_rank)
        next_step, best_val = 0, None
        last_saved = ckpt.discover_latest()
        if last_saved is not None:
            runtime = {}
            next_step = ckpt.load(model, optims, loader, step=last_saved, runtime_state=runtime,
                                  expected_runtime={"contract": contract})
            spike.load_state_dict(runtime["spike"])
            rewind.load_state_dict(runtime["rewind"])
            best_val, base_lrs = runtime["best_val"], runtime["base_lrs"]
            LOG.info("resuming at next update %d", next_step)
        elif cfg.get("load_ckpt"):
            ckpt.load_weights_only(model, cfg["load_ckpt"], allow_legacy=cfg.get("load_legacy_weights", False))
        else:
            _collective_call(ckpt.reject_legacy_resume, "legacy restart validation")
        run_dir = Path(cfg["out_dir"]) / cfg["run_id"]
        _collective_call(lambda: _record_manifests(run_dir, loader, validation), "run manifest")
        logger = Logger(str(Path(cfg["out_dir"]) / "log.jsonl") if cfg["log"]["jsonl"] else None,
                        cfg["log"].get("wandb_project"), cfg)
        logger.log(next_step, event="init", device=device, compute_dtype=cfg["dtype"], parameter_dtype="float32",
                   **topology.dimensions, next_step=next_step, train_manifest=loader.manifest_sha256,
                   validation_manifest=validation.manifest_sha256 if validation else None)

        @contextlib.contextmanager
        def autocast():
            if dtype == torch.bfloat16:
                with torch.amp.autocast(device_type, dtype=dtype):
                    yield
            else:
                yield

        def runtime_state() -> dict:
            return {"contract": contract, "best_val": best_val, "base_lrs": base_lrs,
                    "spike": spike.state_dict(), "rewind": rewind.state_dict()}

        total = optim["total_steps"]
        stop = cfg["train"].get("stop_after", total)
        accum = cfg["train"]["grad_accum"]
        tokens_per_step = cfg["train"]["micro_batch"] * accum * dp * cfg["data"]["seq_len"]
        peak = peak_tflops_for_device(dtype)
        started, measured_steps = time.perf_counter(), 0
        while next_step < stop:
            step = next_step
            scheduled = cosine_lr(step, optim["warmup_steps"], total, optim["lr"], optim["min_lr"])
            effective_lr = scheduled * rewind.lr_multiplier()
            scale = effective_lr / optim["lr"]
            for optimizer, lrs in zip(optims, base_lrs):
                for group, lr in zip(optimizer.param_groups, lrs):
                    group["lr"] = lr * scale
                optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for micro in range(accum):
                if fsdp_active:
                    model.set_requires_gradient_sync(micro == accum - 1)
                no_sync = model.no_sync() if isinstance(model, DistributedDataParallel) and micro < accum - 1 else contextlib.nullcontext()
                with no_sync:
                    x, y = loader.next_batch()
                    with autocast():
                        _, loss = model(x, y)
                        loss = loss / accum
                    loss.backward()
                    loss_sum += loss.detach().float()
            loss_value = topology.mean_dp(loss_sum).item()
            grad_norm = clip_grad_norm(model, optim["grad_clip"])
            if topology.any_rank(not math.isfinite(loss_value) or not math.isfinite(grad_norm), device):
                raise FloatingPointError("nonfinite loss/gradient on a rank; no optimizer update or checkpoint was published")
            for optimizer in optims:
                optimizer.step()
            next_step += 1
            measured_steps += 1
            if topology.any_rank(spike.observe(loss_value), device):
                restored = {}
                count = rewind.n_rewinds
                next_step = rewind.on_spike(model, optims, loader, next_step, runtime_state=restored)
                if rewind.n_rewinds != count:
                    spike.load_state_dict(restored["spike"])
                    best_val, base_lrs = restored["best_val"], restored["base_lrs"]
                    logger.log(next_step, event="rewind", **rewind.state_dict())
                    started, measured_steps = time.perf_counter(), 0
                    continue
            if step % cfg["train"]["log_every"] == 0:
                if device_type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) / measured_steps
                metrics = {"loss": loss_value, "lr": effective_lr, "ms": elapsed * 1000,
                           "tok_per_s": tokens_per_step / elapsed,
                           "tok_per_s_per_gpu": tokens_per_step / elapsed / world,
                           "grad_norm": grad_norm, "next_step": next_step}
                if (step // cfg["train"]["log_every"]) % 10 == 0:
                    metrics["param_norm"] = compute_param_norm(model)
                if peak is not None:
                    metrics["mfu"] = estimate_mfu(mcfg, tokens_per_step, elapsed, peak, world)
                logger.log(step, **metrics)
                started, measured_steps = time.perf_counter(), 0
            improved = False
            every_eval = cfg["train"]["eval_every"]
            if every_eval and next_step % every_eval == 0:
                value = evaluate(model, validation, autocast, topology, cfg["train"].get("eval_batches", 1))
                if topology.any_rank(not math.isfinite(value), device):
                    raise FloatingPointError("nonfinite held-out loss")
                logger.log(step, eval_loss=value, next_step=next_step,
                           validation_manifest=validation.manifest_sha256)
                improved = best_val is None or value < best_val
                if improved:
                    best_val = value
            every_ckpt = cfg["train"]["ckpt_every"]
            if improved or (every_ckpt and next_step % every_ckpt == 0):
                ckpt.save(model, optims, loader, next_step, runtime_state=runtime_state(), is_best=improved)
                last_saved = next_step
        if last_saved != next_step:
            ckpt.save(model, optims, loader, next_step, runtime_state=runtime_state())
        if dist.is_initialized():
            dist.barrier()
    finally:
        if logger is not None:
            logger.close()
        dist_destroy()
