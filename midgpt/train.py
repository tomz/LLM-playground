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
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world


@torch.no_grad()
def evaluate(model, datasets, eval_iters, batch_size, ctx, gen):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = ds.get_batch(batch_size, gen)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
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
                            rank=rank, world_size=world, split="train")
    val_ds = ShardDataset(shard_dir, cfg["model"]["block_size"], device,
                          rank=rank, world_size=world, split="val")

    mcfg = GPTConfig(**cfg["model"])
    model = GPT(mcfg, grad_checkpoint=cfg["grad_checkpoint"]).to(device)
    if is_master:
        print(f"model: {model.num_params()/1e6:.2f}M non-emb params")

    if cfg.get("compile"):
        model = torch.compile(model)

    optim = model.configure_optimizer(
        cfg["optim"]["weight_decay"], cfg["optim"]["lr"],
        tuple(cfg["optim"]["betas"]), fused=is_cuda,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and is_cuda))

    # Wrap in DDP AFTER optimizer construction (params are the same tensors)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    inner = model.module if use_ddp else model

    start_iter = 0
    best_val = float("inf")
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.pt")
    best_path = os.path.join(cfg["out_dir"], "ckpt_best.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        inner.load_state_dict(sd["model"])
        optim.load_state_dict(sd["optim"])
        if "scaler" in sd and sd["scaler"] is not None:
            scaler.load_state_dict(sd["scaler"])
        if "rng_state" in sd:
            torch.set_rng_state(sd["rng_state"])
        if is_cuda and sd.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(sd["cuda_rng_state"])
        if "gen_state" in sd and sd["gen_state"] is not None:
            gen.set_state(sd["gen_state"])
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
            model=inner.state_dict(), optim=optim.state_dict(),
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
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
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
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(inner.parameters(), cfg["optim"]["grad_clip"])
        scaler.step(optim); scaler.update()

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

        if is_master and it > 0 and it % eval_int == 0:
            ev = evaluate(inner, {"val": val_ds}, cfg["train"]["eval_iters"], micro_bs, ctx, gen)
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

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
