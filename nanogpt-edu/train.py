"""Training loop. Single-GPU or CPU. Cosine LR, AMP, grad clip, ckpt."""
import argparse, importlib.util, math, os, time
import torch
from data import ShardDataset, load_meta
from model import GPT, GPTConfig


def load_config(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("cfg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def pick_device(want: str) -> str:
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(it: int, warmup: int, decay: int, lr: float, min_lr: float) -> float:
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > decay:
        return min_lr
    progress = (it - warmup) / max(1, decay - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, datasets, cfg, ctx) -> dict:
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            x, y = ds.get_batch(cfg["batch_size"])
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

    cfg = load_config(args.config)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    is_cuda = device.startswith("cuda")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg["dtype"]]
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if is_cuda else torch.amp.autocast(
        device_type="cpu", dtype=torch.bfloat16, enabled=(cfg["dtype"] == "bfloat16")
    )

    meta = load_meta(cfg["data_dir"])
    cfg["vocab_size"] = meta["vocab_size"]
    train_ds = ShardDataset(cfg["data_dir"], "train", cfg["block_size"], device)
    val_ds = ShardDataset(cfg["data_dir"], "val", cfg["block_size"], device)

    mcfg = GPTConfig(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
        dropout=cfg["dropout"], rope_base=cfg["rope_base"],
    )
    model = GPT(mcfg).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    if cfg.get("compile") and hasattr(torch, "compile"):
        model = torch.compile(model)

    # AdamW with no WD on biases/norms/embeddings
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.dim() < 2 or n.endswith("weight") and "norm" in n.lower()) else decay).append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=cfg["betas"], fused=is_cuda,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    start_iter = 0
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model"]); optim.load_state_dict(sd["optim"])
        start_iter = sd["iter"] + 1
        print(f"resumed from iter {start_iter}")

    t0 = time.time()
    for it in range(start_iter, cfg["max_iters"]):
        lr = cosine_lr(it, cfg["warmup_iters"], cfg["lr_decay_iters"], cfg["lr"], cfg["min_lr"])
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        for _ in range(cfg["grad_accum"]):
            x, y = train_ds.get_batch(cfg["batch_size"])
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg["grad_accum"]
            scaler.scale(loss).backward()
        if cfg["grad_clip"]:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optim); scaler.update()

        if it % cfg["log_interval"] == 0:
            dt = time.time() - t0; t0 = time.time()
            print(f"iter {it:6d} | loss {loss.item()*cfg['grad_accum']:.4f} | lr {lr:.2e} | {dt*1000/cfg['log_interval']:.1f} ms/it")
        if it > 0 and it % cfg["eval_interval"] == 0:
            ev = evaluate(model, {"train": train_ds, "val": val_ds}, cfg, ctx)
            print(f"           eval | train {ev['train']:.4f} | val {ev['val']:.4f}")
        if it > 0 and it % cfg["ckpt_interval"] == 0:
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                        "iter": it, "cfg": cfg, "meta": meta}, ckpt_path)
            print(f"           saved ckpt -> {ckpt_path}")

    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "iter": cfg["max_iters"] - 1, "cfg": cfg, "meta": meta}, ckpt_path)
    print(f"done. final ckpt -> {ckpt_path}")


if __name__ == "__main__":
    main()
