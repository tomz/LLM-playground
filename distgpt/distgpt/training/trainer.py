"""The training loop. Wires model + parallelism + data + optim + ckpt + logging."""
from __future__ import annotations
import contextlib, os, time
import torch

from ..model.config import ModelConfig
from ..model.transformer import GPT
from ..parallel.mesh import build_mesh
from ..parallel.fsdp import apply_fsdp
from ..parallel.tensor import apply_tp
from ..parallel.pipeline import build_pipeline
from ..data.streaming import StreamingLoader
from ..utils.dist import init as dist_init, destroy as dist_destroy, is_master, all_reduce_mean
from ..utils.logging import Logger
from .optim import build_optimizer, cosine_lr
from .checkpoint import CheckpointManager
from .stability import SpikeMonitor, RewindController

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def train(cfg: dict, data_dir_override: str | None = None) -> None:
    rank, local_rank, world = dist_init()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
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

    # Apply parallelism: TP first, then PP, then FSDP on the data dim.
    if mesh is not None:
        model = apply_tp(model, mesh["tp"])
        model, pp_schedule = build_pipeline(model, mesh["pp"], n_microbatches=cfg["train"]["grad_accum"])
        if pcfg["zero"] == "fsdp":
            model = apply_fsdp(model, mesh["dp"], dtype)
    else:
        pp_schedule = None

    optim = build_optimizer(model, cfg["optim"]["lr"], cfg["optim"]["betas"],
                            cfg["optim"]["weight_decay"], fused=is_cuda)

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

    start = ckpt.load(model, optim, loader, step="latest") if ckpt.latest() is not None else 0
    if is_master() and start > 0:
        print(f"[resume] @ step {start}")

    accum = cfg["train"]["grad_accum"]
    total = cfg["optim"]["total_steps"]
    log_every = cfg["train"]["log_every"]
    eval_every = cfg["train"]["eval_every"]
    ckpt_every = cfg["train"]["ckpt_every"]
    tokens_per_step = cfg["train"]["micro_batch"] * accum * dp * cfg["data"]["seq_len"]
    autocast = torch.amp.autocast("cuda", dtype=dtype) if is_cuda else contextlib.nullcontext()

    t0 = time.time()
    for step in range(start, total):
        base_lr = cosine_lr(step, cfg["optim"]["warmup_steps"], total,
                            cfg["optim"]["lr"], cfg["optim"]["min_lr"])
        eff_lr = base_lr * rewind.lr_multiplier()
        for g in optim.param_groups:
            g["lr"] = eff_lr

        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            x, y = loader.next_batch()
            with autocast:
                if pp_schedule is None:
                    _, loss = model(x, y)
                    loss = loss / accum
                else:
                    # pipeline schedule consumes a single (x,y) and runs n_microbatches internally;
                    # we treat the outer accum as further accumulation by re-invoking.
                    losses = []
                    pp_schedule.step(x, target=y, losses=losses)
                    loss = sum(losses) / max(1, len(losses)) / accum
            loss.backward()
            loss_acc += loss.detach().float().item()

        if cfg["optim"]["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
        optim.step()

        # Aggregate loss across DP for accurate logging.
        loss_t = torch.tensor(loss_acc * accum, device=device)
        loss_t = all_reduce_mean(loss_t)
        loss_val = loss_t.item()

        if spike.observe(loss_val):
            if is_master():
                print(f"[spike] step={step} loss={loss_val:.3f} — rewinding")
            step = rewind.on_spike(model, optim, loader, step) - 1
            continue

        if step % log_every == 0:
            dt = (time.time() - t0) / max(1, log_every); t0 = time.time()
            tps = tokens_per_step / dt
            if is_master():
                print(f"step {step:7d} | loss {loss_val:.4f} | lr {eff_lr:.2e} | {dt*1000:.0f} ms | {tps/1e6:.2f}M tok/s")
            logger.log(step, loss=loss_val, lr=eff_lr, ms=dt * 1000, tok_per_s=tps)

        if step > 0 and step % ckpt_every == 0:
            path = ckpt.save(model, optim, loader, step)
            if is_master():
                print(f"[ckpt] saved → {path}")

        if step > 0 and step % eval_every == 0:
            # quick held-out loss estimate using current loader (for brevity).
            model.eval()
            with torch.no_grad():
                xs, ys = loader.next_batch()
                with autocast:
                    _, ev_loss = model(xs, ys)
            ev = all_reduce_mean(ev_loss.detach().float()).item()
            if is_master():
                print(f"           eval | loss {ev:.4f}")
            logger.log(step, eval_loss=ev)
            model.train()

    if is_master():
        ckpt.save(model, optim, loader, total)
        print("[done]")
    logger.close()
    dist_destroy()
