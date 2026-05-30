"""Pretrain a ~12M-param transformer on TinyShakespeare using frontier-platform.

Pipeline (all real, all from `platform/*`):
  download TinyShakespeare
    -> train BPE (HF tokenizers, vocab 4096)
    -> tokenize+shard via platform.data.shard.tokenize_and_shard
    -> StreamingLoader + MixtureSampler
    -> Transformer (GQA, RoPE, RMSNorm, SwiGLU)
    -> Trainer (cosine LR, grad clip)
    -> CheckpointManager.save_async (rolling)
    -> TorchEngine generation from "ROMEO:"
"""
from __future__ import annotations
import asyncio
import time
from pathlib import Path

import numpy as np
import requests
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# --- platform imports (rely on `pip install -e frontier-platform`) ---
from platform.tokenizer.bpe import TokenizerConfig, Tokenizer, train as train_bpe
from platform.data.acquire import RawDoc
from platform.data.shard import tokenize_and_shard
from platform.data.mix import DomainSpec, MixtureSampler
from platform.data.loader import StreamingLoader
from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.training.optim import OptimConfig, build_optimizer
from platform.training.trainer import Trainer, TrainConfig
from platform.training.parallel import ParallelConfig
from platform.training.checkpoint import CheckpointManager
from platform.serving.engine import Engine, EngineConfig, GenRequest


SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# ============================================================================
# Data prep
# ============================================================================

def download_shakespeare() -> Path:
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / "input.txt"
    if not input_path.exists():
        print(f"[data] downloading {SHAKES_URL}")
        r = requests.get(SHAKES_URL, timeout=30)
        r.raise_for_status()
        input_path.write_bytes(r.content)
    print(f"[data] input.txt: {input_path.stat().st_size / 1024:.1f} KiB")
    text = input_path.read_text()
    n = len(text)
    split = int(n * 0.9)
    (data_dir / "train.txt").write_text(text[:split])
    (data_dir / "val.txt").write_text(text[split:])
    print(f"[data] train={split} chars, val={n - split} chars")
    return data_dir


def train_tokenizer(data_dir: Path) -> Path:
    tok_dir = OUT / "tok"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok_path = tok_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"[tok] reusing {tok_path}")
        return tok_path
    print("[tok] training BPE (vocab=4096) via platform.tokenizer.bpe.train")
    cfg = TokenizerConfig(vocab_size=4096, byte_level=True, split_digits=True)
    train_bpe(str(data_dir / "train.txt"), cfg, str(tok_path))
    print(f"[tok] wrote {tok_path}")
    return tok_path


def make_shards(data_dir: Path, tokenizer: Tokenizer) -> Path:
    shard_root = OUT / "shards"
    domain_dir = shard_root / "shakespeare"
    if domain_dir.exists() and any(domain_dir.glob("*.bin")):
        print(f"[shard] reusing {domain_dir}")
        return shard_root
    print("[shard] tokenize_and_shard -> uint32 .bin")
    train_text = (data_dir / "train.txt").read_text()
    docs = [RawDoc(source="tinyshakes", uri="train.txt", mime="text/plain",
                   payload=train_text.encode("utf-8"), meta={})]
    uris = tokenize_and_shard(docs, tokenizer, str(shard_root),
                              domain="shakespeare", shard_tokens=2_000_000)
    for u in uris:
        sz = Path(u).stat().st_size
        print(f"[shard] {u}  ({sz / 1024:.1f} KiB)")
    return shard_root


# ============================================================================
# GPU loader adapter
# ============================================================================

class _GpuLoader:
    """Wraps a StreamingLoader so each batch is moved to `device` as int64."""

    def __init__(self, base: StreamingLoader, device: torch.device):
        self._base = base
        self.device = device

    def __iter__(self):
        for x, y in self._base:
            x_t = torch.from_numpy(np.ascontiguousarray(x)).long().to(self.device, non_blocking=True)
            y_t = torch.from_numpy(np.ascontiguousarray(y)).long().to(self.device, non_blocking=True)
            yield x_t, y_t

    def state_dict(self):
        return self._base.state_dict()

    def load_state_dict(self, sd):
        return self._base.load_state_dict(sd)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this example.")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    dev_name = torch.cuda.get_device_name(0)
    print(f"[cuda] device={dev_name}")

    # ---- data ----
    data_dir = download_shakespeare()
    tok_path = train_tokenizer(data_dir)
    tokenizer = Tokenizer(str(tok_path))
    print(f"[tok] vocab_size={tokenizer.vocab_size}  bos={tokenizer.bos_id} eos={tokenizer.eos_id}")
    shard_root = make_shards(data_dir, tokenizer)

    # ---- model ----
    cfg = ModelConfig(
        vocab_size=4096, n_layer=8, n_head=8, n_kv_head=4,
        d_model=384, d_ffn=1024, max_seq_len=256,
    )
    n_params = cfg.param_count()
    print(f"[model] approx params: {n_params / 1e6:.2f} M")
    model = Transformer(cfg).to(device)
    actual = sum(p.numel() for p in model.parameters())
    print(f"[model] actual params: {actual / 1e6:.2f} M")

    # ---- loader ----
    seq_len, micro_batch = 256, 16
    domain = DomainSpec(name="shakespeare",
                        shard_glob=str(shard_root / "shakespeare" / "*.bin"),
                        weight=1.0, epochs_cap=1000.0)
    mixer = MixtureSampler([domain], global_seed=42)
    base_loader = StreamingLoader(mixer, seq_len=seq_len, micro_batch=micro_batch,
                                  rank=0, world_size=1, seed=42)
    loader = _GpuLoader(base_loader, device)

    # ---- training ----
    total_steps = 3000
    ocfg = OptimConfig(peak_lr=3e-4, warmup_steps=100, total_steps=total_steps,
                       grad_clip=1.0, weight_decay=0.1)
    opt, sched = build_optimizer(model, ocfg)

    run_id = "shakespeare-12M"
    ckpt_mgr = CheckpointManager(root_uri=str(OUT / "ckpts"), run_id=run_id,
                                 keep_last=2, milestone_every=10**9)

    tcfg = TrainConfig(
        run_id=run_id, seq_len=seq_len, micro_batch=micro_batch,
        global_batch_tokens=seq_len * micro_batch,
        total_tokens=10**12, log_every=50, eval_every=0,
        ckpt_every=1000, optim=ocfg, parallel=ParallelConfig(),
    )

    trainer = Trainer(model=model, dataloader=loader, ckpt_mgr=ckpt_mgr,
                      evaluator=None, cfg=tcfg, optimizer=opt, scheduler=sched)

    print(f"[train] starting fit() for {total_steps} steps  (silent until done)")
    t_train = time.time()
    final_metrics = trainer.fit()
    train_secs = time.time() - t_train
    losses = trainer.loss_history
    print(f"[train] done in {train_secs:.1f}s  steps={len(losses)}  final_loss={final_metrics['loss']:.3f}")
    # Print a coarse loss curve.
    chunk = max(1, len(losses) // 10)
    for i in range(0, len(losses), chunk):
        seg = losses[i:i+chunk]
        print(f"[train] step {i:5d}..{i+len(seg)-1:5d}  mean_loss={sum(seg)/len(seg):6.3f}")
    toks_per_sec = (len(losses) * micro_batch * seq_len) / train_secs
    print(f"[train] tokens/sec: {toks_per_sec:,.0f}")
    first_100 = sum(losses[:100]) / max(1, len(losses[:100]))
    last_100  = sum(losses[-100:]) / max(1, len(losses[-100:]))
    print(f"[train] loss first-100 mean={first_100:.3f}  last-100 mean={last_100:.3f}")

    # Save a self-contained final ckpt for downstream examples.
    final_path = OUT / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "model_cfg": cfg,
        "tokenizer_path": str(tok_path),
        "loss_history": losses,
        "first_100_mean": first_100,
        "last_100_mean": last_100,
    }, final_path)
    print(f"[ckpt] wrote {final_path}")

    # ---- generation ----
    print("[gen] running TorchEngine for 500 tokens from 'ROMEO:'")
    engine = Engine(EngineConfig(backend="torch", dtype="fp32"),
                    model=model, tokenizer=tokenizer)
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode("ROMEO:")
    req = GenRequest(prompt_ids=prompt_ids, max_new_tokens=500,
                     temperature=0.8, top_p=0.9)

    async def _gen():
        out_ids: list[int] = []
        async for ev in engine.generate(req):
            if ev.get("done"):
                break
            out_ids.append(ev["token_id"])
        return out_ids

    gen_ids = asyncio.run(_gen())
    sample = "ROMEO:" + tokenizer.decode(gen_ids)
    (OUT / "sample.txt").write_text(sample)
    print("[gen] first 300 chars:")
    print("    " + sample[:300].replace("\n", "\n    "))

    # ---- result.md ----
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    wall = time.time() - t_start
    write_result_md(
        n_params=actual, total_steps=total_steps,
        first_100=first_100, last_100=last_100,
        train_secs=train_secs, wall=wall, peak_gb=peak_gb,
        sample=sample, dev_name=dev_name,
    )
    print(f"\n[done] wall={wall:.1f}s  peak_gpu={peak_gb:.2f} GiB")


def write_result_md(*, n_params, total_steps, first_100, last_100,
                    train_secs, wall, peak_gb, sample, dev_name):
    md = f"""# 01 \u2014 TinyShakespeare pretraining: result

Recorded from one real run on a **{dev_name}** using
frontier-platform components end-to-end.

## Summary

| metric                          | value |
|--------------------------------|------:|
| parameters                     | {n_params/1e6:.2f} M |
| training steps                 | {total_steps} |
| tokens seen                    | {total_steps * 256 * 16 / 1e6:.2f} M |
| first-100-step mean loss       | {first_100:.3f} |
| last-100-step mean loss        | {last_100:.3f} |
| reduction                      | {first_100 - last_100:+.3f} |
| training wall time             | {train_secs:.1f} s |
| total wall time (incl. download/BPE/gen) | {wall:.1f} s |
| peak GPU memory                | {peak_gb:.2f} GiB |

## Generated sample (first 1000 chars)

The sample starts from the prompt `ROMEO:` with `temperature=0.8, top_p=0.9`.

```
{sample[:1000]}
```

Full 500-token sample in `out/sample.txt`.
"""
    (HERE / "result.md").write_text(md)


if __name__ == "__main__":
    main()
