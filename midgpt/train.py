"""Single-node training. DDP if launched via torchrun, else single-process.

  python train.py --config configs/gpt2_124m.yaml
  torchrun --standalone --nproc_per_node 8 train.py --config configs/gpt2_350m.yaml
"""
import argparse, contextlib, os, time
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from data import ShardDataset
from model import GPT, GPTConfig
from utils import JsonlLogger, cosine_lr, save_ckpt


def setup_ddp() -> tuple[bool, int, int, int]:
    """Initialize the process group iff torchrun gave us WORLD_SIZE > 1.

    Backend selection: CUDA → NCCL (the default). Set ``MIDGPT_BACKEND=gloo``
    to force a CPU-friendly backend — useful in CI / multi-rank CPU smoke
    tests where NCCL is unavailable, and matches how ``distgpt`` exposes the
    same knob (``DISTGPT_BACKEND``).
    """
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return False, 0, 0, 1
    backend = os.environ.get("MIDGPT_BACKEND",
                             "nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world


@torch.no_grad()
def evaluate(model, datasets, eval_iters, batch_size, ctx, *,
             eval_seed: int, world: int = 1, device: str = "cpu"):
    """Compute mean loss over ``eval_iters`` random batches per split.

    Reproducibility: we always seed a *fresh* generator from ``eval_seed`` so
    val loss is the same number on every call regardless of training history
    (with the old code, val ppl drifted between runs / resumes because the
    training generator's state was the seed source).

    DDP-correct: when ``world > 1`` every rank runs the eval (each on its own
    rank-offset shuffle of the val set) and the losses are mean-all-reduced
    across ranks, so the reported val ppl reflects the full val set rather
    than just rank-0's slice. With ``world == 1`` it's a plain mean over
    ``eval_iters`` batches.
    """
    import torch.distributed as dist
    model.eval()
    out = {}
    for split, ds in datasets.items():
        eval_gen = torch.Generator()
        # Per-rank seed offset so each rank scans different val windows; the
        # cross-rank average then covers ``world * eval_iters`` distinct batches.
        rank = dist.get_rank() if (world > 1 and dist.is_initialized()) else 0
        eval_gen.manual_seed(eval_seed + 7919 * rank)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = ds.get_batch(batch_size, eval_gen)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        local_mean = losses.mean()
        if world > 1 and dist.is_initialized():
            t = local_mean.to(device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            local_mean = (t / world).cpu()
        out[split] = local_mean.item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_ddp, rank, local_rank, world = setup_ddp()
    is_master = (rank == 0)
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    is_cuda = device.startswith("cuda")
    is_mps = device == "mps"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
    if is_cuda:
        ctx = torch.amp.autocast("cuda", dtype=dtype)
    elif is_mps:
        # MPS supports fp16 autocast natively; bf16 is also accepted.
        ctx = torch.amp.autocast("mps", dtype=dtype)
    else:
        ctx = contextlib.nullcontext()

    if is_master:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        print(f"world_size={world}  device={device}  dtype={dtype}")

    torch.manual_seed(cfg["seed"] + rank)
    gen = torch.Generator()
    gen.manual_seed(cfg["seed"] + rank * 1000003)

    shard_dir = os.path.join("data", cfg["dataset"])
    train_ds = ShardDataset(shard_dir, cfg["model"]["block_size"], device,
                            split="train")
    val_ds = ShardDataset(shard_dir, cfg["model"]["block_size"], device,
                          split="val")

    mcfg = GPTConfig(**cfg["model"])
    # Liger fused-linear-CE (optional, GPU+Triton only). Returns loss-only, so we
    # disable it automatically for eval paths that need logits.
    fused_ce = bool(cfg.get("fused_ce", False))
    model = GPT(mcfg, grad_checkpoint=cfg["grad_checkpoint"], fused_ce=fused_ce).to(device)
    if is_master:
        print(f"model: {model.num_params()/1e6:.2f}M non-emb params"
              + ("  [liger fused-CE]" if fused_ce else ""))

    if cfg.get("compile"):
        model = torch.compile(model)

    # Optimizer choice: "adamw" (default) or "muon". Muon orthogonalizes the
    # update for 2D hidden weights via Newton-Schulz and routes embeddings /
    # pos_emb / lm_head / 1-D params to a small AdamW — ~1.35x sample-efficiency
    # on the modded-nanogpt FineWeb speedrun. See muon.py.
    base_lr = cfg["optim"]["lr"]
    if cfg["optim"].get("optimizer", "adamw") == "muon":
        from muon import Muon, split_muon_params
        muon_params, adamw_params = split_muon_params(model)
        muon_lr = cfg["optim"].get("muon_lr", 0.02)
        muon = Muon(muon_params, lr=muon_lr, momentum=cfg["optim"].get("muon_momentum", 0.95))
        aux = torch.optim.AdamW(adamw_params, lr=base_lr, betas=tuple(cfg["optim"]["betas"]),
                                weight_decay=cfg["optim"]["weight_decay"], fused=is_cuda)
        optimizers = [muon, aux]
        if is_master:
            print(f"optimizer: Muon ({len(muon_params)} 2D mats, lr={muon_lr}) "
                  f"+ AdamW ({len(adamw_params)} other, lr={base_lr})")
    else:
        optimizers = [model.configure_optimizer(
            cfg["optim"]["weight_decay"], base_lr,
            tuple(cfg["optim"]["betas"]), fused=is_cuda,
        )]
    # Stamp each param group with its base LR so the cosine schedule can scale
    # all optimizers (Muon + AdamW) by a single multiplier while preserving
    # their different absolute learning rates.
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and is_cuda))

    # Wrap in DDP AFTER optimizer construction (params are the same tensors).
    # device_ids=[local_rank] is required for CUDA but must be None for CPU/gloo
    # (DDP would otherwise try to look up a CUDA device that doesn't exist on
    # CPU CI runners and crash before the first forward).
    if use_ddp:
        ddp_kwargs = dict(gradient_as_bucket_view=True)
        if is_cuda:
            ddp_kwargs["device_ids"] = [local_rank]
        model = DDP(model, **ddp_kwargs)
    inner = model.module if use_ddp else model

    start_iter = 0
    best_val = float("inf")
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.pt")
    best_path = os.path.join(cfg["out_dir"], "ckpt_best.pt")
    if args.resume and os.path.exists(ckpt_path):
        # Load tensors straight to CPU so saved RNG ByteTensors stay ByteTensors
        # (the previous map_location=device crashed on CUDA resume: torch.load
        # promoted the CPU ByteTensor RNG state to a CUDA tensor which then
        # failed `torch.set_rng_state`'s ByteTensor check). Per-rank model/optim
        # state moves to the right device when load_state_dict copies into the
        # already-on-device parameters.
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        inner.load_state_dict(sd["model"])
        # Backward-compatible: old checkpoints stored a single "optim"; new ones
        # store a list of optimizer state dicts under "optims" (Muon + AdamW).
        if "optims" in sd:
            for opt, osd in zip(optimizers, sd["optims"]):
                opt.load_state_dict(osd)
        elif "optim" in sd:
            optimizers[0].load_state_dict(sd["optim"])
        if "scaler" in sd and sd["scaler"] is not None:
            scaler.load_state_dict(sd["scaler"])
        if "rng_state" in sd:
            # torch.get_rng_state() returns a CPU ByteTensor; restore as-is.
            torch.set_rng_state(sd["rng_state"].to("cpu", dtype=torch.uint8))
        if is_cuda and sd.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(sd["cuda_rng_state"].to("cpu", dtype=torch.uint8))
        if "gen_state" in sd and sd["gen_state"] is not None:
            gen.set_state(sd["gen_state"].to("cpu", dtype=torch.uint8))
        start_iter = sd["iter"] + 1
        best_val = sd.get("best_val", best_val)
        if is_master:
            print(f"resumed @ iter {start_iter} (best_val={best_val:.4f})")

    logger = JsonlLogger(os.path.join(cfg["out_dir"], "log.jsonl")) if (is_master and cfg["log"]["jsonl"]) else JsonlLogger(None)
    wb = None
    if is_master and cfg["log"].get("wandb_project"):
        import wandb
        wb = wandb.init(project=cfg["log"]["wandb_project"], config=cfg)

    micro_bs = cfg["train"]["micro_batch"]
    accum = cfg["train"]["grad_accum"]
    log_int = cfg["train"]["log_interval"]
    eval_int = cfg["train"]["eval_interval"]
    ckpt_int = cfg["train"]["ckpt_interval"]
    block = cfg["model"]["block_size"]
    tokens_per_step = micro_bs * accum * world * block

    def _save(path: str, it: int):
        save_ckpt(path, dict(
            model=inner.state_dict(),
            optims=[opt.state_dict() for opt in optimizers],
            scaler=scaler.state_dict() if scaler.is_enabled() else None,
            iter=it, cfg=cfg, best_val=best_val,
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state() if is_cuda else None,
            gen_state=gen.get_state(),
        ))

    t0 = time.time()
    for it in range(start_iter, cfg["optim"]["max_iters"]):
        lr = cosine_lr(it, cfg["optim"]["warmup_iters"], cfg["optim"]["lr_decay_iters"],
                       cfg["optim"]["lr"], cfg["optim"]["min_lr"])
        # Scale every optimizer's groups by the same cosine multiplier relative to
        # each group's own base LR (so Muon's higher LR and AdamW's lower LR decay
        # on one shared schedule).
        mult = lr / base_lr if base_lr else 1.0
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(accum):
            # only sync grads on last micro-step
            sync_ctx = model.no_sync() if (use_ddp and micro < accum - 1) else contextlib.nullcontext()
            with sync_ctx:
                x, y = train_ds.get_batch(micro_bs, gen)
                with ctx:
                    _, loss = model(x, y)
                    loss = loss / accum
                scaler.scale(loss).backward()
                loss_acc += loss.detach().float().item()
        if cfg["optim"]["grad_clip"]:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(inner.parameters(), cfg["optim"]["grad_clip"])
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()

        # Each micro-batch loss was already divided by accum, so loss_acc is
        # already the per-step average. Optionally all-reduce across DP for an
        # unbiased value before logging.
        step_loss = loss_acc
        if use_ddp:
            t = torch.tensor(step_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            step_loss = (t / world).item()

        if is_master and it % log_int == 0:
            dt = (time.time() - t0) / max(1, log_int); t0 = time.time()
            tps = tokens_per_step / dt
            row = dict(iter=it, loss=step_loss, lr=lr, ms=dt * 1000, tok_per_s=tps)
            print(f"iter {it:6d} | loss {row['loss']:.4f} | lr {lr:.2e} | {row['ms']:.0f} ms | {tps/1e3:.1f}k tok/s")
            logger.log(**row)
            if wb: wb.log(row, step=it)

        if it > 0 and it % eval_int == 0:
            # Eval runs on EVERY rank (mean is all-reduced across DP) so the
            # reported val loss covers the full val set rather than rank-0's
            # slice. Old code ran eval inside `if is_master:` on a rank-sharded
            # val set → val ppl was measured on only 1/world of the val set.
            ev = evaluate(inner, {"val": val_ds}, cfg["train"]["eval_iters"],
                          micro_bs, ctx, eval_seed=cfg["seed"] + 12345,
                          world=world, device=device)
            if is_master:
                print(f"           eval | val {ev['val']:.4f} | ppl {torch.tensor(ev['val']).exp().item():.2f}")
                logger.log(iter=it, **{f"eval_{k}": v for k, v in ev.items()})
                if wb: wb.log({f"eval/{k}": v for k, v in ev.items()}, step=it)
                if ev["val"] < best_val:
                    best_val = ev["val"]
                    _save(best_path, it)
                    print(f"           best val {best_val:.4f} → saved {best_path}")

        if is_master and it > 0 and it % ckpt_int == 0:
            _save(ckpt_path, it)
            print(f"           saved ckpt -> {ckpt_path}")

    if is_master:
        _save(ckpt_path, cfg["optim"]["max_iters"] - 1)
        logger.close()
        if wb: wb.finish()
        print(f"done -> {ckpt_path}  best val {best_val:.4f} -> {best_path}")
        if is_cuda:
            alloc = torch.cuda.max_memory_allocated(local_rank) // (1024 * 1024)
            resv = torch.cuda.max_memory_reserved(local_rank) // (1024 * 1024)
            print(f"[vram] peak_alloc={alloc} MiB  peak_reserved={resv} MiB")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
