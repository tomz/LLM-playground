"""Train a dense and a top-2 MoE transformer on the example 01 shards,
compare loss curves, tokens/sec, and MoE expert utilisation.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

from platform.data.mix import DomainSpec, MixtureSampler
from platform.data.loader import StreamingLoader
from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.training.optim import OptimConfig, build_optimizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SHARDS = HERE.parent / "01_pretrain_shakespeare" / "out" / "shards" / "shakespeare"


def make_loader(seq_len: int, micro_batch: int, seed: int):
    shard_glob = str(SHARDS / "*.bin")
    if not list(SHARDS.glob("*.bin")):
        raise SystemExit(
            f"No shards at {SHARDS}\nRun examples/01_pretrain_shakespeare/run.sh first."
        )
    mixer = MixtureSampler(
        [DomainSpec(name="shakespeare", shard_glob=shard_glob,
                    weight=1.0, epochs_cap=1000.0)],
        global_seed=seed,
    )
    return StreamingLoader(mixer, seq_len=seq_len, micro_batch=micro_batch,
                           rank=0, world_size=1, seed=seed)


def train_one(name: str, cfg: ModelConfig, *, steps: int, seq_len: int,
              micro_batch: int, device: torch.device, log_every: int = 50) -> dict:
    print(f"\n=== training {name} ===")
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params={n_params/1e6:.2f}M  d_model={cfg.d_model} "
          f"d_ffn={cfg.d_ffn} experts={cfg.moe_num_experts or 1}")

    ocfg = OptimConfig(peak_lr=3e-4, warmup_steps=50, total_steps=steps,
                       grad_clip=1.0, weight_decay=0.1)
    opt, sched = build_optimizer(model, ocfg)
    loader = make_loader(seq_len, micro_batch, seed=42)

    losses: list[tuple[int, float]] = []
    aux_losses: list[tuple[int, float]] = []
    step_times: list[float] = []
    model.train()

    is_moe = cfg.moe_num_experts > 0
    moe_layer = model.layers[0].ffn if is_moe else None

    it = iter(loader)
    t_start = time.time()
    last_t = t_start
    for step in range(steps):
        x_np, y_np = next(it)
        x = torch.from_numpy(np.ascontiguousarray(x_np)).long().to(device)
        y = torch.from_numpy(np.ascontiguousarray(y_np)).long().to(device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        now = time.time()
        step_times.append(now - last_t)
        last_t = now

        if step % log_every == 0 or step == steps - 1:
            losses.append((step, float(loss.detach())))
            if is_moe:
                aux_losses.append((step, float(moe_layer.last_aux_loss.detach())))
            print(f"[{name}] step {step:4d}  loss={float(loss.detach()):6.3f}"
                  + (f"  aux={float(moe_layer.last_aux_loss.detach()):.4f}" if is_moe else ""))

    # tokens/sec over the last 500 steps
    tail = step_times[-500:] if len(step_times) > 500 else step_times
    tail_secs = sum(tail)
    tail_toks = len(tail) * micro_batch * seq_len
    tok_per_sec = tail_toks / max(tail_secs, 1e-6)

    result: dict = {
        "name": name,
        "params": n_params,
        "steps": steps,
        "wall_secs": time.time() - t_start,
        "tokens_per_sec_last500": tok_per_sec,
        "loss_curve": losses,
        "first10_mean": sum(v for _, v in losses[:2]) / 2,  # first 2 logged
        "last10_mean": sum(v for _, v in losses[-2:]) / 2,
    }
    if is_moe:
        # Final expert counts from one extra forward pass
        with torch.no_grad():
            x_np, y_np = next(iter(loader))
            x = torch.from_numpy(np.ascontiguousarray(x_np)).long().to(device)
            model(x)
        counts = moe_layer.last_expert_counts.cpu().tolist()
        total = sum(counts)
        utilisation = [c / total for c in counts]
        result["aux_loss_curve"] = aux_losses
        result["expert_counts"] = counts
        result["expert_utilisation"] = utilisation
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    common = dict(vocab_size=4096, n_layer=6, n_head=4, n_kv_head=2,
                  max_seq_len=256)
    dense_cfg = ModelConfig(d_model=192, d_ffn=512, **common)
    moe_cfg   = ModelConfig(d_model=128, d_ffn=384,
                            moe_num_experts=4, moe_top_k=2, **common)

    print(f"[init] dense param_count={dense_cfg.param_count()/1e6:.2f}M")
    print(f"[init]   moe param_count={moe_cfg.param_count()/1e6:.2f}M (top-{moe_cfg.moe_top_k} of {moe_cfg.moe_num_experts})")

    steps = 1000
    dense_res = train_one("dense", dense_cfg, steps=steps, seq_len=256,
                          micro_batch=8, device=device)
    torch.cuda.empty_cache()
    moe_res   = train_one("moe",   moe_cfg,   steps=steps, seq_len=256,
                          micro_batch=8, device=device)

    (OUT / "dense.json").write_text(json.dumps(dense_res, indent=2))
    (OUT / "moe.json").write_text(json.dumps(moe_res, indent=2))

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    wall = time.time() - t_start

    # ---- result.md ----
    lines = []
    lines.append("# 03 — MoE vs dense at matched active params: result\n")
    lines.append("Both models trained on TinyShakespeare shards from example 01,\n"
                 "identical optimizer (AdamW, lr 3e-4 cosine, warmup 50, 1000 steps),\n"
                 "micro-batch 8 × seq 256.\n")
    lines.append("## Headline\n")
    lines.append("| model | params | tokens/sec (last 500) | first-2 mean loss | last-2 mean loss |")
    lines.append("|---|--:|--:|--:|--:|")
    for r in (dense_res, moe_res):
        lines.append(f"| {r['name']} | {r['params']/1e6:.2f} M "
                     f"| {r['tokens_per_sec_last500']:,.0f} "
                     f"| {r['first10_mean']:.3f} | {r['last10_mean']:.3f} |")
    lines.append("")
    lines.append("## Loss curves (step, loss)\n")
    for r in (dense_res, moe_res):
        lines.append(f"### {r['name']}\n")
        lines.append("```")
        for s, v in r["loss_curve"]:
            lines.append(f"{s:5d}  {v:6.3f}")
        lines.append("```")
    lines.append("")
    lines.append("## MoE diagnostics\n")
    lines.append("### Aux-loss trajectory (z-loss + load-balance) — layer 0\n")
    lines.append("```")
    for s, v in moe_res["aux_loss_curve"]:
        lines.append(f"{s:5d}  {v:.4f}")
    lines.append("```")
    lines.append("")
    lines.append("### Final per-expert token counts (layer 0, one batch)\n")
    lines.append("| expert | count | utilisation |")
    lines.append("|---|--:|--:|")
    for i, (c, u) in enumerate(zip(moe_res["expert_counts"], moe_res["expert_utilisation"])):
        lines.append(f"| {i} | {c} | {u:.1%} |")
    lines.append("")
    lines.append(f"_Total wall time: {wall:.1f}s. Peak GPU memory: {peak_gb:.2f} GiB._\n")
    (HERE / "result.md").write_text("\n".join(lines))

    print(f"\n[done] wall={wall:.1f}s  peak_gpu={peak_gb:.2f} GiB")
    print(f"[done] dense last-loss={dense_res['last10_mean']:.3f}  "
          f"tok/s={dense_res['tokens_per_sec_last500']:,.0f}")
    print(f"[done]   moe last-loss={moe_res['last10_mean']:.3f}  "
          f"tok/s={moe_res['tokens_per_sec_last500']:,.0f}")
    print(f"[done] expert utilisation: {moe_res['expert_utilisation']}")


if __name__ == "__main__":
    main()
