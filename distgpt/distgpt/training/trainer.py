"""The training loop. Wires model + parallelism + data + optim + ckpt + logging.

Pipeline-parallelism note
-------------------------
When PP > 1 the schedule (`Schedule1F1B`) runs both forward AND backward over
its `n_microbatches` configured at construction time, so we set
`n_microbatches = grad_accum` once and execute it as a SINGLE step — there is
no outer `for _ in range(accum)` loop nor a manual `loss.backward()`. The
schedule reports per-microbatch losses on the last PP rank only; intermediate
ranks see no loss tensor.

Eval under PP
-------------
The old code returned ``nan`` for any pipelined eval, which silently disabled
``best.txt`` tracking. We now build a separate Schedule1F1B with
``n_microbatches=1`` and run it once at eval time; the last PP rank produces
the loss, gathers across DP, and broadcasts to all ranks so logging /
best-checkpoint logic works the same as in the non-PP case.

Compilation
-----------
``train.compile: true`` runs ``torch.compile(model, mode='reduce-overhead')``
on the wrapped model (after TP/PP/FSDP composition). Skipped when PP > 1
because ``Schedule1F1B``'s graph-capture doesn't currently round-trip with
``torch.compile`` cleanly.

Metrics
-------
Every log step records:
  * ``loss``: DP-averaged cross-entropy
  * ``lr``: cosine-with-warmup schedule value × rewind multiplier
  * ``ms``, ``tok_per_s``, ``tok_per_s_per_gpu``: wall-clock throughput
  * ``mfu``: model-FLOPs-utilization vs the device's peak (when known)
  * ``grad_norm`` (pre-clip): the single best early-warning signal for
    divergence — a diverging run usually announces itself in grad-norm
    100+ steps before the loss spikes.
  * ``param_norm``: aggregate L2 of model parameters (sanity check that
    weight-decay is doing what you expect).
"""
from __future__ import annotations
import contextlib, os, time
import torch
import torch.distributed as dist

from ..model.config import ModelConfig
from ..model.transformer import GPT
from ..parallel.mesh import build_mesh
from ..parallel.fsdp import apply_fsdp
from ..parallel.tensor import apply_tp
from ..parallel.pipeline import build_pipeline
from ..data.streaming import StreamingLoader
from ..utils.dist import init as dist_init, destroy as dist_destroy, is_master, all_reduce_mean
from ..utils.logging import Logger
from ..utils.metrics import (
    compute_grad_norm, compute_param_norm, estimate_mfu, peak_tflops_for_device,
)
from .optim import build_optimizer, cosine_lr
from .muon import build_muon_and_adamw
from .precision import resolve_fp8_recipe, autocast_fp8_context, log_fp8_choice
from .checkpoint import CheckpointManager
from .stability import SpikeMonitor, RewindController

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _maybe_compile(model, enabled: bool, pp_active: bool) -> object:
    """Wrap with ``torch.compile`` when requested and safe.

    ``pp_active=True`` skips compilation because PyTorch's pipelining schedule
    captures its own graph and the two layers of graph capture don't currently
    compose. We log the decision so users notice when their compile=true got
    quietly ignored under PP.

    We use ``mode="default"`` rather than ``"reduce-overhead"`` because the
    latter enables CUDA Graph trees, which interact badly with the lazy RoPE
    cache (``GPT._rope`` is overwritten across steps and CUDA Graph trees
    treat it as a live output between calls). ``"default"`` still gets the
    inductor kernel fusion speedup at the cost of a few extra CPU launches
    per step. If you need cudagraphs, either pre-allocate RoPE in
    ``GPT.__init__`` as a buffer or use ``"reduce-overhead"`` after that
    refactor.
    """
    if not enabled:
        return model
    if pp_active:
        if is_master():
            print("[compile] disabled (PP active; torch.compile + Schedule1F1B "
                  "don't compose yet)")
        return model
    try:
        compiled = torch.compile(model, mode="default", fullgraph=False)
        if is_master():
            print("[compile] torch.compile(mode='default') applied")
        return compiled
    except Exception as e:
        # If compile fails (older torch / unsupported op), fall back gracefully
        # rather than killing a multi-day run.
        if is_master():
            print(f"[compile] failed: {type(e).__name__}: {e}; running uncompiled")
        return model


def train(cfg: dict, data_dir_override: str | None = None) -> None:
    rank, local_rank, world = dist_init()
    colocate = os.environ.get("DISTGPT_COLOCATE_RANKS") == "1"
    device = ("cuda:0" if colocate else f"cuda:{local_rank}") if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda")
    dtype = DTYPES[cfg["dtype"]]
    if is_master():
        os.makedirs(cfg["out_dir"], exist_ok=True)
        print(f"[init] rank={rank}/{world} device={device} dtype={dtype}")

    torch.manual_seed(cfg["seed"])

    pcfg = cfg["parallel"]
    mesh, dp, tp, pp = build_mesh(world, pcfg["dp"], pcfg["tp"], pcfg["pp"], "cuda" if is_cuda else "cpu")
    if is_master():
        print(f"[mesh] dp={dp} tp={tp} pp={pp}")

    mcfg = ModelConfig(**cfg["model"])
    if is_master():
        print(f"[model] {mcfg.param_count()/1e9:.2f}B params")
    model = GPT(mcfg, activation_ckpt=pcfg["activation_ckpt"]).to(device=device, dtype=dtype)

    # Apply parallelism: TP first, then PP (which trims to local stage), then
    # FSDP on the data dim.
    pp_schedule = None
    is_last_pp_rank = True
    if mesh is not None:
        model = apply_tp(model, mesh["tp"],
                          sequence_parallel=pcfg.get("sequence_parallel", False))
        model, pp_schedule = build_pipeline(model, mesh["pp"], n_microbatches=cfg["train"]["grad_accum"])
        if pp_schedule is not None:
            pp_mesh = mesh["pp"]
            is_last_pp_rank = (pp_mesh.get_local_rank() == pp_mesh.size() - 1)
        if pcfg["zero"] == "fsdp":
            model = apply_fsdp(model, mesh["dp"], dtype,
                                reshard_after_forward=pcfg.get("reshard_after_forward", True))

    model = _maybe_compile(model, cfg["train"].get("compile", False),
                            pp_active=pp_schedule is not None)

    optim_kind = cfg["optim"].get("optimizer", "adamw")
    if optim_kind == "muon":
        # Dual Muon (hidden 2D weights) + AdamW (embeddings / heads / 1-D)
        # on one shared cosine schedule. ``muon_lr`` defaults to 20× the
        # ``adamw_lr`` because Muon's update is scaled to RMS≈1 already
        # (the modded-nanogpt convention; tune per-run if you change ndim).
        muon_lr = cfg["optim"].get("muon_lr", cfg["optim"]["lr"] * 100)
        optims = build_muon_and_adamw(
            model,
            muon_lr=muon_lr,
            adamw_lr=cfg["optim"]["lr"],
            muon_momentum=cfg["optim"].get("muon_momentum", 0.95),
            muon_weight_decay=cfg["optim"].get("muon_weight_decay", 0.0),
            muon_update_scale=cfg["optim"].get("muon_update_scale"),
            adamw_betas=cfg["optim"]["betas"],
            weight_decay=cfg["optim"]["weight_decay"],
            fused=is_cuda,
        )
        if is_master():
            n_muon = sum(1 for o in optims if o.__class__.__name__ == "Muon")
            n_adam = sum(1 for o in optims if o.__class__.__name__ == "AdamW")
            print(f"[optim] muon={n_muon} adamw={n_adam} (dual schedule)")
    elif optim_kind == "adamw":
        optims = [build_optimizer(
            model, cfg["optim"]["lr"], cfg["optim"]["betas"],
            cfg["optim"]["weight_decay"], fused=is_cuda,
        )]
    else:
        raise ValueError(
            f"unknown optim.optimizer={optim_kind!r}; expected 'adamw' or 'muon'"
        )

    data_dir = data_dir_override or cfg["data"]["dir"]
    loader = StreamingLoader(
        data_dir, cfg["data"]["seq_len"], cfg["train"]["micro_batch"],
        rank=rank, world_size=world, seed=cfg["seed"], device=device,
    )
    ckpt = CheckpointManager(cfg["out_dir"], cfg["run_id"])
    spike = SpikeMonitor()
    rewind = RewindController(ckpt)
    logger = Logger(
        os.path.join(cfg["out_dir"], "log.jsonl") if cfg["log"]["jsonl"] else None,
        cfg["log"].get("wandb_project"), cfg,
    )

    # Snapshot the per-group base LRs so we can re-apply the schedule on top
    # of them without losing per-optimizer scaling (Muon's LR is much higher
    # than AdamW's; the cosine should preserve the ratio, not collapse them).
    _base_lrs = [
        [g["lr"] for g in o.param_groups] for o in optims
    ]

    # Resume / warm-start logic. Priority order:
    #   1. Native resume: if `out_dir/run_id/ckpts/` has a step dir, pick up
    #      where we left off (model + optim + loader cursor + step counter).
    #   2. Warm start: `load_ckpt: path` in YAML loads weights only from an
    #      external checkpoint and starts at step 0 with a fresh optim and
    #      loader. Used by cooldown / longctx / domain-adapt recipes.
    #   3. Cold start: brand-new model, step 0.
    if ckpt.latest() is not None:
        start = ckpt.load(model, optims, loader, step="latest")
        if is_master():
            print(f"[resume] @ step {start}")
    elif cfg.get("load_ckpt"):
        warm_path = cfg["load_ckpt"]
        if is_master():
            print(f"[warm-start] loading weights from {warm_path}")
        ckpt.load_weights_only(model, warm_path)
        start = 0
    else:
        start = 0
    best_val = float("inf")

    accum = cfg["train"]["grad_accum"]
    total = cfg["optim"]["total_steps"]
    log_every = cfg["train"]["log_every"]
    eval_every = cfg["train"]["eval_every"]
    ckpt_every = cfg["train"]["ckpt_every"]
    tokens_per_step = cfg["train"]["micro_batch"] * accum * dp * cfg["data"]["seq_len"]
    bf16_autocast = torch.amp.autocast("cuda", dtype=dtype) if is_cuda else contextlib.nullcontext()
    # FP8 (Transformer Engine) wraps the bf16 autocast when train.fp8 != "off".
    # The recipe resolver downgrades to no-FP8 on unsupported HW/dtype with a
    # warning, so a misconfigured run keeps training in bf16 instead of erroring.
    fp8_recipe = resolve_fp8_recipe(cfg.get("train", {}).get("fp8", "off"), device, dtype)
    if is_master():
        log_fp8_choice(fp8_recipe, dtype)

    @contextlib.contextmanager
    def autocast():
        """Combined fp8 (optional) + bf16 autocast. Re-entered every step
        so TE's amax-history buffers update correctly per-iteration."""
        with autocast_fp8_context(fp8_recipe), bf16_autocast:
            yield

    # Peak device throughput for MFU. None on CPU / unrecognized GPU — MFU
    # is then omitted from the log rather than reported as a wrong number.
    peak_tflops = peak_tflops_for_device(dtype)
    if is_master() and peak_tflops is not None:
        print(f"[mfu] peak {peak_tflops:.1f} TFLOP/s @ {dtype}")

    t0 = time.time()
    step = start
    while step < total:
        base_lr = cosine_lr(step, cfg["optim"]["warmup_steps"], total,
                            cfg["optim"]["lr"], cfg["optim"]["min_lr"])
        eff_lr = base_lr * rewind.lr_multiplier()
        # Apply the cosine schedule by scaling each param group's *original*
        # LR by the same factor — that way Muon (whose base is 100× AdamW's)
        # keeps its relative LR through warmup/decay.
        scale = eff_lr / max(cfg["optim"]["lr"], 1e-12)
        for o_idx, o in enumerate(optims):
            for g_idx, g in enumerate(o.param_groups):
                g["lr"] = _base_lrs[o_idx][g_idx] * scale

        for o in optims:
            o.zero_grad(set_to_none=True)
        loss_acc = 0.0
        if pp_schedule is None:
            # No pipelining: regular gradient accumulation. Under FSDP2 we
            # gate gradient sync so the reduce-scatter fires only on the LAST
            # micro-step instead of all `accum` of them — on a PCIe fabric
            # (no NVLink) that single change is the difference between
            # comm-bound (~6% MFU) and compute-bound. `set_requires_grad_sync`
            # is a no-op on non-FSDP models, so this is safe single-GPU too.
            sync_gate = getattr(model, "set_requires_gradient_sync", None)
            for micro in range(accum):
                if sync_gate is not None:
                    sync_gate(micro == accum - 1)
                x, y = loader.next_batch()
                with autocast():
                    _, loss = model(x, y)
                    loss = loss / accum
                loss.backward()
                loss_acc += loss.detach().float().item()
            step_loss = loss_acc  # already averaged across micro-steps
        else:
            # Pipelined: schedule.step() runs forward + backward over its own
            # n_microbatches (= accum). Pull a single concatenated batch.
            x, y = loader.next_batch()
            losses: list[torch.Tensor] = []
            with autocast():
                if is_last_pp_rank:
                    pp_schedule.step(x, target=y, losses=losses)
                else:
                    pp_schedule.step(x)
            if losses:
                step_loss = sum(l.detach().float().item() for l in losses) / len(losses)
            else:
                step_loss = 0.0  # non-tail PP ranks don't see a loss tensor

        # Pre-clip grad norm: the single best early-warning signal for
        # divergence. clip_grad_norm_ returns the (pre-clip) total norm so
        # we get it for free; expose it in the log.
        if cfg["optim"]["grad_clip"]:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            grad_norm = float(gnorm) if isinstance(gnorm, torch.Tensor) else gnorm
        else:
            grad_norm = compute_grad_norm(model)
        for o in optims:
            o.step()

        # Aggregate loss across DP for accurate logging. PP intermediates have
        # 0.0 — we average only across the DP dim, not the PP dim, so they
        # don't drag the value down.
        loss_t = torch.tensor(step_loss, device=device)
        loss_t = all_reduce_mean(loss_t)
        loss_val = loss_t.item()

        if is_last_pp_rank and spike.observe(loss_val):
            if is_master():
                print(f"[spike] step={step} loss={loss_val:.3f} — rewinding")
            step = rewind.on_spike(model, optims, loader, step)
            continue

        if step % log_every == 0:
            dt = (time.time() - t0) / max(1, log_every); t0 = time.time()
            tps = tokens_per_step / dt
            tps_per_gpu = tps / max(1, world)
            metrics = dict(loss=loss_val, lr=eff_lr, ms=dt * 1000,
                            tok_per_s=tps, tok_per_s_per_gpu=tps_per_gpu,
                            grad_norm=grad_norm)
            # Param norm is a few hundred ms per call on large models; sample
            # every 10× log_every to keep it cheap.
            if (step // log_every) % 10 == 0:
                metrics["param_norm"] = compute_param_norm(model)
            if peak_tflops is not None:
                mfu = estimate_mfu(mcfg, tokens_per_step, dt, peak_tflops,
                                    world_size=world)
                if mfu is not None:
                    metrics["mfu"] = mfu
            if is_master():
                mfu_str = f" | mfu {metrics.get('mfu', 0)*100:.1f}%" if "mfu" in metrics else ""
                print(f"step {step:7d} | loss {loss_val:.4f} | lr {eff_lr:.2e} "
                       f"| gnorm {grad_norm:.3f} | {dt*1000:.0f} ms "
                       f"| {tps/1e6:.2f}M tok/s{mfu_str}")
            logger.log(step, **metrics)

        if step > 0 and step % ckpt_every == 0:
            path = ckpt.save(model, optims, loader, step)
            if is_master():
                print(f"[ckpt] saved → {path}")

        if step > 0 and step % eval_every == 0:
            ev = _eval_one_batch(model, loader, autocast, pp_schedule,
                                  is_last_pp_rank, accum, mesh, device, world)
            if is_master():
                print(f"           eval | loss {ev:.4f}")
            logger.log(step, eval_loss=ev)
            if ev == ev and ev < best_val:  # NaN-safe
                best_val = ev
                bp = ckpt.save(model, optims, loader, step)
                # Mark as "best" via a sibling sentinel file (DCP dirs are
                # already step-named; we just leave a pointer).
                if is_master():
                    with open(os.path.join(ckpt.root, "best.txt"), "w") as f:
                        f.write(f"step_{step:09d}\n")
                    print(f"           best val {best_val:.4f} → {bp}")
            model.train()

        step += 1

    if is_master():
        print("[done]")
    # `ckpt.save` is a collective (dcp.save + dist.barrier) — every rank
    # must participate, not just rank-0. The old `if is_master(): ckpt.save`
    # gated the entire collective on rank-0 and deadlocked on rank-1+ as
    # soon as either side hit a barrier. Bug was invisible because no
    # multi-rank test ever exercised the final-save path.
    ckpt.save(model, optims, loader, total)
    if is_master():
        if is_cuda:
            alloc = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
            resv = torch.cuda.max_memory_reserved(0) // (1024 * 1024)
            print(f"[vram] peak_alloc={alloc} MiB  peak_reserved={resv} MiB")
    logger.close()
    # Synchronize one last time so every rank exits dist together; without
    # this rank-0's slower final FS writes (DCP fsync, JSONL flush) can
    # outlive rank-1's destroy() and rank-1 already-destroyed group makes
    # rank-0's destroy() take a slow path.
    if dist.is_initialized():
        dist.barrier()
    dist_destroy()


def _eval_one_batch(model, loader, autocast, pp_schedule, is_last_pp_rank,
                     accum, mesh, device, world) -> float:
    """Single-batch held-out loss that works under both non-PP and PP.

    Under PP we build a side eval schedule with ``n_microbatches=1`` (a real
    PP eval pass) so the last rank produces the loss; we then DP-average
    and broadcast back so every rank logs the same value. Under non-PP this
    is just ``model(x, y)`` like before.
    """
    from torch.distributed.pipelining import Schedule1F1B

    model.eval()
    try:
        with torch.no_grad():
            xs, ys = loader.next_batch()
            with autocast():
                if pp_schedule is None:
                    _, ev_loss = model(xs, ys)
                    ev = all_reduce_mean(ev_loss.detach().float()).item()
                    return ev
                # PP path: build a 1-microbatch eval schedule on the same
                # stage. We reuse the existing stage attached to pp_schedule.
                stage = pp_schedule._stage  # private but stable across 2.4-2.7
                eval_schedule = Schedule1F1B(stage, n_microbatches=1)
                losses: list[torch.Tensor] = []
                if is_last_pp_rank:
                    eval_schedule.step(xs, target=ys, losses=losses)
                    ev_loss = (sum(l.float() for l in losses) / max(1, len(losses)))
                    ev_t = ev_loss.detach()
                else:
                    eval_schedule.step(xs)
                    ev_t = torch.tensor(0.0, device=device)
                # DP-average over DP ranks only; then broadcast from the
                # last PP rank so every rank in the world logs the same
                # value (rank-0 might not be the last PP rank).
                ev_t = all_reduce_mean(ev_t)
                if mesh is not None and "pp" in mesh.mesh_dim_names:
                    pp_mesh = mesh["pp"]
                    src = pp_mesh.size() - 1  # last PP rank within pp dim
                    # Use the pp-dim subgroup for the broadcast so we only
                    # talk to peers along the pp axis.
                    if pp_mesh.size() > 1:
                        dist.broadcast(ev_t, src=src, group=pp_mesh.get_group())
                return float(ev_t.item())
    finally:
        model.train()
