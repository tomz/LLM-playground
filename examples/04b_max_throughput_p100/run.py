"""Max-throughput benchmark for the Tesla P100 (16 GB, sm_60).

Counterpart to example 04 but adapted for the P100:
- fp16 (not bf16 — sm_60 has no bf16 support; Pascal has no tensor cores either,
  so fp16 runs on the same FP units, just doubled throughput vs fp32)
- larger model + larger batch to use the 16 GB VRAM
- GradScaler for fp16 stability
- nvidia-smi indexed at physical device 0 (P100 is GPU0 on this box)
"""
from __future__ import annotations
import asyncio
import gc
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
EX01 = HERE.parent / "01_pretrain_shakespeare" / "out"

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
from platform.serving.engine import Engine, EngineConfig, GenRequest

SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
THEORETICAL_TFLOPS_FP16 = 18.7  # Tesla P100 fp16 dense (NVIDIA spec: 18.7 TFLOPS fp16)
# Note: P100 has no tensor cores; fp16 throughput comes from packed fp16x2 on FP32 units.


def prepare_data() -> tuple[Path, Path]:
    """Return (tokenizer_path, shard_root). Build locally with vocab=8192."""
    tok_path = OUT / "tok" / "tokenizer.json"
    shard_root = OUT / "shards"
    domain_dir = shard_root / "shakespeare"
    if tok_path.exists() and domain_dir.exists() and any(domain_dir.glob("*.bin")):
        print(f"[data] reusing {tok_path} and shards")
        return tok_path, shard_root

    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw = data_dir / "input.txt"
    if not raw.exists():
        print("[data] downloading TinyShakespeare")
        r = requests.get(SHAKES_URL, timeout=30); r.raise_for_status()
        raw.write_bytes(r.content)
    text = raw.read_text()
    # Repeat x30 for ~30 MB of training text -> ~8M BPE tokens
    big = data_dir / "train_x30.txt"
    if not big.exists():
        big.write_text(text * 30)
    print(f"[data] train corpus: {big.stat().st_size / 1024**2:.1f} MiB")

    if not tok_path.exists():
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        print("[tok] training BPE vocab=8192")
        train_bpe(str(big), TokenizerConfig(vocab_size=8192, byte_level=True,
                                            split_digits=True), str(tok_path))

    tokenizer = Tokenizer(str(tok_path))
    if not domain_dir.exists() or not any(domain_dir.glob("*.bin")):
        print("[shard] tokenizing into uint32 shards")
        docs = [RawDoc(source="tinyshakes", uri="train_x30.txt", mime="text/plain",
                       payload=big.read_bytes(), meta={})]
        uris = tokenize_and_shard(docs, tokenizer, str(shard_root),
                                  domain="shakespeare", shard_tokens=20_000_000)
        for u in uris:
            print(f"[shard] {u}  ({Path(u).stat().st_size / 1024**2:.1f} MiB)")
    return tok_path, shard_root


class _GpuLoader:
    def __init__(self, base: StreamingLoader, device: torch.device):
        self._base = base; self.device = device
    def __iter__(self):
        for x, y in self._base:
            x_t = torch.from_numpy(np.ascontiguousarray(x)).long().to(self.device, non_blocking=True)
            y_t = torch.from_numpy(np.ascontiguousarray(y)).long().to(self.device, non_blocking=True)
            yield x_t, y_t
    def state_dict(self): return self._base.state_dict()
    def load_state_dict(self, sd): return self._base.load_state_dict(sd)


def autotune_batch(cfg: ModelConfig, seq_len: int, device, candidates) -> int:
    print(f"[autotune] probing batches {candidates}")
    for B in candidates:
        m: object = None
        o: object = None
        try:
            torch.cuda.empty_cache(); gc.collect()
            torch.cuda.reset_peak_memory_stats()
            m = Transformer(cfg).to(device)
            o = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(0.9, 0.95))
            m.train()
            for _ in range(3):
                x = torch.randint(0, cfg.vocab_size, (B, seq_len), device=device, dtype=torch.long)
                y = torch.randint(0, cfg.vocab_size, (B, seq_len), device=device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = m(x, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                o.step(); o.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[autotune] B={B} OK, peak={peak:.2f} GiB")
            del m, o; torch.cuda.empty_cache(); gc.collect()
            return B
        except torch.cuda.OutOfMemoryError:
            print(f"[autotune] B={B} OOM, trying smaller")
            del m, o
            torch.cuda.empty_cache(); gc.collect()
            continue
    raise SystemExit("[autotune] even B=1 OOMs; model too large for this GPU")


def start_nvsmi(log_path: Path) -> subprocess.Popen:
    f = open(log_path, "w")
    p = subprocess.Popen(
        ["nvidia-smi", "-i", "0",
         "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
         "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=f, stderr=subprocess.DEVNULL,
    )
    p._logfile = f  # type: ignore
    return p


def parse_nvsmi(log_path: Path, t0: float, warmup_s: float = 5.0):
    """Return (mean, p50, p95) of util.gpu over the training window."""
    utils = []
    if not log_path.exists():
        return 0.0, 0.0, 0.0
    for i, line in enumerate(log_path.read_text().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            u = float(parts[1])
        except ValueError:
            continue
        utils.append(u)
    if not utils:
        return 0.0, 0.0, 0.0
    # drop first `warmup_s` * 2 samples (sampler at 2 Hz)
    drop = int(warmup_s * 2)
    utils = utils[drop:] if len(utils) > drop + 10 else utils
    a = np.array(utils, dtype=np.float64)
    return float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95))


def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda:0")
    print(f"[cuda] {torch.cuda.get_device_name(0)}  total={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GiB")

    tok_path, shard_root = prepare_data()
    tokenizer = Tokenizer(str(tok_path))
    print(f"[tok] vocab={tokenizer.vocab_size}")

    # P100 has 16 GB and is bandwidth-king (732 GB/s). Push wider+deeper.
    cfg = ModelConfig(
        vocab_size=max(8192, tokenizer.vocab_size),
        n_layer=32, n_head=16, n_kv_head=4,
        d_model=1024, d_ffn=4096, max_seq_len=1024,
        activation_ckpt="selective",
    )
    actual_params_approx = cfg.param_count()
    print(f"[model] approx params: {actual_params_approx/1e6:.2f} M  (act_ckpt=selective, fp16+GradScaler)")

    seq_len = 1024
    micro_batch = autotune_batch(cfg, seq_len, device, [32, 24, 16, 12, 8, 6, 4, 2, 1])
    print(f"[autotune] chosen micro_batch = {micro_batch}")

    # Real model + loader for the benchmark
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    model = Transformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] actual params: {n_params/1e6:.2f} M")

    # Monkey-patch ParallelEngine to do fp16 autocast + GradScaler (Pascal-friendly).
    # Weights stay fp32 (AdamW state too); only matmuls + activations run in fp16.
    from platform.training import parallel as _parallel
    scaler = torch.cuda.amp.GradScaler()
    def _fb_amp(self, batch):
        input_ids, targets = batch
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, loss = self.model(input_ids, targets=targets)
        scaler.scale(loss).backward()
        return {"loss": float(loss.detach()), "tokens": int(input_ids.numel())}
    def _step(self):
        if self.cfg.grad_clip > 0:
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
    _parallel.ParallelEngine.forward_backward = _fb_amp
    _parallel.ParallelEngine.step = _step

    domain = DomainSpec(name="shakespeare",
                        shard_glob=str(shard_root / "shakespeare" / "*.bin"),
                        weight=1.0, epochs_cap=1000.0)
    mixer = MixtureSampler([domain], global_seed=42)
    base_loader = StreamingLoader(mixer, seq_len=seq_len, micro_batch=micro_batch,
                                  rank=0, world_size=1, seed=42)
    loader = _GpuLoader(base_loader, device)

    total_steps = 1500
    ocfg = OptimConfig(peak_lr=3e-4, warmup_steps=100, total_steps=total_steps,
                       weight_decay=0.1, grad_clip=1.0, betas=(0.9, 0.95))
    opt, sched = build_optimizer(model, ocfg)
    tcfg = TrainConfig(
        run_id="bench", seq_len=seq_len, micro_batch=micro_batch,
        global_batch_tokens=seq_len * micro_batch,
        total_tokens=10**15, log_every=100, eval_every=0, ckpt_every=0,
        optim=ocfg, parallel=ParallelConfig(),
    )
    trainer = Trainer(model=model, dataloader=loader, ckpt_mgr=None,
                      evaluator=None, cfg=tcfg, optimizer=opt, scheduler=sched)

    nvsmi_log = OUT / "nvsmi.log"
    print(f"[nvsmi] sampling -> {nvsmi_log}")
    nvsmi = start_nvsmi(nvsmi_log)
    train_secs = 0.0
    try:
        time.sleep(1.0)  # let nvsmi settle
        print(f"[train] {total_steps} steps  seq={seq_len}  micro_batch={micro_batch}")
        t_train = time.time()
        trainer.fit()
        torch.cuda.synchronize()
        train_secs = time.time() - t_train
    finally:
        try:
            nvsmi.send_signal(signal.SIGTERM)
            nvsmi.wait(timeout=5)
        except Exception:
            try: nvsmi.kill()
            except: pass
        try: nvsmi._logfile.close()  # type: ignore
        except: pass

    losses = trainer.loss_history
    tokens = len(losses) * micro_batch * seq_len
    tps = tokens / train_secs
    # MFU: forward+backward ~ 6 * N * tokens FLOPs (Kaplan/Chinchilla approximation)
    achieved_tflops = 6.0 * n_params * tps / 1e12
    mfu = 100.0 * achieved_tflops / THEORETICAL_TFLOPS_FP16
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    mean_u, p50_u, p95_u = parse_nvsmi(nvsmi_log, t_start)
    first50 = sum(losses[:50]) / max(1, len(losses[:50]))
    last50 = sum(losses[-50:]) / max(1, len(losses[-50:]))

    print(f"\n[result] steps={len(losses)}  train_wall={train_secs:.1f}s")
    print(f"[result] tokens/sec = {tps:,.0f}")
    print(f"[result] achieved TFLOPS = {achieved_tflops:.2f}  (peak fp16 = {THEORETICAL_TFLOPS_FP16})")
    print(f"[result] MFU = {mfu:.1f}%")
    print(f"[result] peak GPU memory = {peak_gb:.2f} GiB / 8.00 GiB")
    print(f"[result] GPU util:  mean={mean_u:.1f}%  P50={p50_u:.1f}%  P95={p95_u:.1f}%")
    print(f"[result] loss  first-50={first50:.3f}  last-50={last50:.3f}")
    if mean_u < 85.0:
        print(f"[warn] mean GPU util {mean_u:.1f}% < 85% target. Consider longer seq, more layers, or larger d_model.")

    # ---- generation ----
    print("[gen] 200 tokens from 'ROMEO:'")
    model.eval()
    engine = Engine(EngineConfig(backend="torch", dtype="fp32"),
                    model=model, tokenizer=tokenizer)
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode("ROMEO:")
    req = GenRequest(prompt_ids=prompt_ids, max_new_tokens=200,
                     temperature=0.8, top_p=0.9)

    async def _gen():
        out = []
        async for ev in engine.generate(req):
            if ev.get("done"): break
            out.append(ev["token_id"])
        return out

    gen_ids = asyncio.run(_gen())
    sample = "ROMEO:" + tokenizer.decode(gen_ids)
    (OUT / "sample.txt").write_text(sample)
    print("[gen] first 240 chars:")
    print("    " + sample[:240].replace("\n", "\n    "))

    wall = time.time() - t_start
    write_result_md(
        n_params=n_params, micro_batch=micro_batch, seq_len=seq_len,
        total_steps=len(losses), tps=tps, achieved_tflops=achieved_tflops,
        mfu=mfu, peak_gb=peak_gb, mean_u=mean_u, p50_u=p50_u, p95_u=p95_u,
        first50=first50, last50=last50, train_secs=train_secs, wall=wall,
        sample=sample,
    )
    print(f"\n[done] wall={wall:.1f}s  peak={peak_gb:.2f} GiB  mean_util={mean_u:.1f}%  MFU={mfu:.1f}%")


def write_result_md(**k):
    md = f"""# 04b \u2014 Max-throughput benchmark on Tesla P100: result

Recorded from one real run on a **Tesla P100-PCIE (16 GB, sm_60)**.
Selective activation checkpointing, batch auto-tuned to fit VRAM.
Compute in **fp16** (autocast + GradScaler); master weights + AdamW state in fp32.
Pascal has no bf16 and no tensor cores; throughput comes from packed-fp16 on the FP32 ALUs.

## Configuration

| field | value |
|---|--:|
| parameters | {k['n_params']/1e6:.2f} M |
| seq_len | {k['seq_len']} |
| micro_batch (auto) | {k['micro_batch']} |
| tokens / step | {k['seq_len'] * k['micro_batch']:,} |
| training steps | {k['total_steps']} |
| optimizer | AdamW, peak_lr=3e-4, warmup=100, cosine |
| precision | fp16 autocast + GradScaler (fp32 master) |
| activation checkpointing | selective (per Block) |

## Throughput

| metric | value |
|---|--:|
| training wall time | {k['train_secs']:.1f} s |
| total wall time | {k['wall']:.1f} s |
| tokens / second | {k['tps']:,.0f} |
| achieved TFLOPS (6\u00b7N\u00b7tps) | {k['achieved_tflops']:.2f} |
| theoretical peak (fp16, P100) | 18.7 TFLOPS |
| **MFU** | **{k['mfu']:.1f}%** |

## GPU saturation (`nvidia-smi -lms 500`)

| stat | utilization.gpu |
|---|--:|
| mean | **{k['mean_u']:.1f}%** |
| P50 | {k['p50_u']:.1f}% |
| P95 | {k['p95_u']:.1f}% |
| peak memory | {k['peak_gb']:.2f} GiB / 16.00 GiB |

## Loss

| window | mean loss |
|---|--:|
| first 50 steps | {k['first50']:.3f} |
| last 50 steps | {k['last50']:.3f} |
| reduction | {k['first50']-k['last50']:+.3f} |

## Generated sample (200 tokens from `ROMEO:`)

```
{k['sample']}
```

Full sample in `out/sample.txt`. Raw nvidia-smi log in `out/nvsmi.log`.
"""
    (HERE / "result.md").write_text(md)


if __name__ == "__main__":
    main()
