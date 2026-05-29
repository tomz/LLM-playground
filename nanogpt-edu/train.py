"""Training loop. Single-GPU / MPS / CPU. Cosine LR, AMP, grad clip, ckpt.

Saves both `ckpt.pt` (latest, for resume) and `ckpt_best.pt` (best val).
Logs JSONL to <out_dir>/train.jsonl alongside stdout.
"""
import argparse, contextlib, importlib.util, json, math, os, time
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


def make_autocast(device: str, dtype: torch.dtype):
    """Build an autocast context appropriate for the device.

    - cuda: native autocast for bf16/fp16/fp32.
    - mps:  native autocast (PyTorch 2.4+ supports bf16/fp16 on Apple silicon).
    - cpu:  bf16 autocast if requested, else nullcontext.
    """
    if device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    if device == "mps":
        return torch.amp.autocast(device_type="mps", dtype=dtype)
    if dtype == torch.bfloat16:
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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


class JsonlLogger:
    def __init__(self, path: str | None):
        self.fh = open(path, "a") if path else None
        self.t0 = time.time()

    def log(self, **kw):
        if not self.fh:
            return
        kw["wall"] = round(time.time() - self.t0, 3)
        self.fh.write(json.dumps(kw) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


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
    ctx = make_autocast(device, dtype)

    meta = load_meta(cfg["data_dir"])
    cfg["vocab_size"] = meta["vocab_size"]
    train_ds = ShardDataset(cfg["data_dir"], "train", cfg["block_size"], device)
    val_ds = ShardDataset(cfg["data_dir"], "val", cfg["block_size"], device)

    mcfg = GPTConfig(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"],
        dropout=cfg["dropout"], rope_base=cfg["rope_base"],
        qk_norm=cfg.get("qk_norm", False),
        zero_init_proj=cfg.get("zero_init_proj", False),
        tie_embeddings=cfg.get("tie_embeddings", True),
        mtp_tokens=cfg.get("mtp_tokens", 0),
        mtp_weight=cfg.get("mtp_weight", 0.3),
    )
    model = GPT(mcfg).to(device)
    print(f"model: {model.num_params(non_embedding=False)/1e6:.2f}M params "
          f"({model.num_params(non_embedding=True)/1e6:.2f}M non-emb)")
    if cfg.get("compile") and hasattr(torch, "compile"):
        model = torch.compile(model)

    # AdamW with no WD on biases / norms / 1-D params (embeddings included via tying).
    # Optimizer choice: "adamw" (default) or "muon". Muon orthogonalizes the
    # update for 2D hidden weights (Newton-Schulz) and routes embeddings /
    # lm_head / 1-D params to a small AdamW — ~1.35x sample-efficiency on the
    # FineWeb speedrun. See muon.py.
    optimizers = []
    if cfg.get("optimizer", "adamw") == "muon":
        from muon import Muon, split_muon_params
        muon_params, adamw_params = split_muon_params(model)
        muon_lr = cfg.get("muon_lr", 0.02)
        muon = Muon(muon_params, lr=muon_lr, momentum=cfg.get("muon_momentum", 0.95))
        # The non-Muon params still want AdamW at the configured `lr`.
        aux = torch.optim.AdamW(adamw_params, lr=cfg["lr"], betas=cfg["betas"],
                                weight_decay=cfg["weight_decay"], fused=is_cuda)
        optimizers = [muon, aux]
        print(f"optimizer: Muon ({len(muon_params)} 2D mats, lr={muon_lr}) "
              f"+ AdamW ({len(adamw_params)} other, lr={cfg['lr']})")
    else:
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            (no_decay if p.dim() < 2 else decay).append(p)
        optimizers = [torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg["weight_decay"]},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg["lr"], betas=cfg["betas"], fused=is_cuda,
        )]
    # Stamp each param group with its base LR so the cosine schedule can scale
    # all optimizers (Muon + AdamW) by a single multiplier while preserving
    # their different absolute learning rates.
    for opt in optimizers:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and is_cuda))

    start_iter = 0
    best_val = float("inf")
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.pt")
    best_path = os.path.join(cfg["out_dir"], "ckpt_best.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        # Backward-compatible: old checkpoints stored a single "optim"; new ones
        # store a list of optimizer state dicts under "optims".
        if "optims" in sd:
            for opt, osd in zip(optimizers, sd["optims"]):
                opt.load_state_dict(osd)
        elif "optim" in sd:
            optimizers[0].load_state_dict(sd["optim"])
        if "scaler" in sd and sd["scaler"] is not None:
            scaler.load_state_dict(sd["scaler"])
        if "rng_state" in sd:
            torch.set_rng_state(sd["rng_state"])
        if is_cuda and sd.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(sd["cuda_rng_state"])
        start_iter = sd["iter"] + 1
        best_val = sd.get("best_val", best_val)
        print(f"resumed from iter {start_iter} (best_val={best_val:.4f})")

    logger = JsonlLogger(os.path.join(cfg["out_dir"], "train.jsonl"))

    def save(path: str, it: int):
        tmp = path + ".tmp"
        torch.save({
            "model": model.state_dict(),
            "optims": [opt.state_dict() for opt in optimizers],
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "iter": it, "cfg": cfg, "meta": meta,
            "best_val": best_val,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if is_cuda else None,
        }, tmp)
        os.replace(tmp, path)

    grad_accum = cfg["grad_accum"]
    base_lr = cfg["lr"]
    t0 = time.time()
    for it in range(start_iter, cfg["max_iters"]):
        lr = cosine_lr(it, cfg["warmup_iters"], cfg["lr_decay_iters"], cfg["lr"], cfg["min_lr"])
        # Scale every optimizer's groups by the same cosine multiplier, relative
        # to each group's own base LR (so Muon's higher LR and AdamW's lower LR
        # both decay on the same schedule).
        mult = lr / base_lr if base_lr else 1.0
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(grad_accum):
            x, y = train_ds.get_batch(cfg["batch_size"])
            with ctx:
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            loss_sum += loss.detach().float().item()
        if cfg["grad_clip"]:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()

        if it % cfg["log_interval"] == 0:
            dt = (time.time() - t0) / max(1, cfg["log_interval"]); t0 = time.time()
            # Each micro-batch loss was already divided by grad_accum, so the sum
            # is the per-step average loss across all micro-batches.
            avg_loss = loss_sum
            row = dict(iter=it, loss=avg_loss, lr=lr, ms=dt * 1000)
            print(f"iter {it:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {row['ms']:.1f} ms/it")
            logger.log(**row)
        if it > 0 and it % cfg["eval_interval"] == 0:
            ev = evaluate(model, {"train": train_ds, "val": val_ds}, cfg, ctx)
            print(f"           eval | train {ev['train']:.4f} | val {ev['val']:.4f}")
            logger.log(iter=it, eval_train=ev["train"], eval_val=ev["val"])
            if ev["val"] < best_val:
                best_val = ev["val"]
                save(best_path, it)
                print(f"           best val {best_val:.4f} → saved {best_path}")
        if it > 0 and it % cfg["ckpt_interval"] == 0:
            save(ckpt_path, it)
            print(f"           saved ckpt -> {ckpt_path}")

    save(ckpt_path, cfg["max_iters"] - 1)
    logger.close()
    print(f"done. final ckpt -> {ckpt_path}  best val {best_val:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
