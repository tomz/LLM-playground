# 5060 Ti example — 416 M Llama-arch GPT trained from scratch with the distgpt stack

A real, reproducible pretraining run of distgpt's full training pipeline on a
**single RTX 5060 Ti (16 GB, Blackwell, sm_120, native bf16)**. Trains a
416 M-parameter Llama-style GPT (RoPE + RMSNorm + SwiGLU + GQA 4:1) from random
init on a 1 B-token slice of `HuggingFaceFW/fineweb-edu`, consumes ~98 M
tokens, and lands at **val loss 4.105 (ppl 60.7)** on held-out FineWeb-Edu in
**2 h 22 min** of training.

distgpt is a multi-node framework (FSDP2 + TP + PP + DCP). The headline run
below is single-GPU, which can't fire the cross-GPU collectives, but it *does*
exercise every other code path: the Llama-arch model, the 3D mesh build
(dp=tp=pp=1), the streaming dataloader with mid-epoch resume, the DCP sharded
checkpointer, SpikeMonitor + RewindController, and the AdamW + cosine +
per-group-WD optim. For the **real cross-GPU FSDP2 collectives** — and the
two fixes it took to get *positive* scaling out of a no-NVLink PCIe pair — see
[Going multi-GPU](#going-multi-gpu-genuine-2-gpu-fsdp2-over-pcie) below,
plus [`scripts/smoke_2gpu.sh`](../scripts/smoke_2gpu.sh) and the reference
topologies in the [README](../README.md).

Companion to [`midgpt/examples/5060ti_350m_fineweb.md`](../../midgpt/examples/5060ti_350m_fineweb.md)
— same GPU, same FineWeb-Edu slice, similar parameter count, but a Llama-arch
model trained through the full distributed-training framework instead of
midgpt's single-file GPT-2.

## TL;DR

```bash
cd distgpt
.venv/bin/pip install -r requirements.txt        # one-time

# 1. Tokenize a 1 B-token slice of FineWeb-Edu (~9 min, ~2 GB on disk)
#    (this run reused the shards prepared by midgpt — same tokenizer)
ln -s ../midgpt/data/fineweb-edu data/fineweb-edu

# 2. Train (2 h 22 min on a 5060 Ti)
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --standalone --nproc_per_node 1 \
    -m distgpt.cli train --config configs/gpt_416m_fweb_5060ti.yaml \
    2>&1 | tee out/gpt_416m_fweb_5060ti_train.log

# 3. Plot
.venv/bin/python scripts/plot_distgpt.py \
    out/gpt_416m_fweb_5060ti/log.jsonl \
    --title "distgpt 416M · FineWeb-Edu · RTX 5060 Ti" \
    --subtitle "Llama-arch (RoPE + RMSNorm + SwiGLU + GQA 4:1) · 3000 steps · bf16"
```

## Headline numbers

| Metric                     | Value |
|----------------------------|------:|
| GPU                        | RTX 5060 Ti 16 GB (sm_120, Blackwell) |
| Model                      | 416 M Llama-arch (24 L × 1024 d × 16 H, GQA 4:1, tied embeddings) |
| Tokenizer                  | tiktoken `gpt2` (50 257 → padded to 50 304) |
| Dataset                    | FineWeb-Edu `sample-10BT`, 1 B-token slice |
| Sequence length            | 1024 |
| Effective batch            | 4 × grad_accum 8 = 32 sequences = **32 768 tokens / step** |
| Steps                      | 3 000 |
| **Wall-clock**             | **2 h 22 min** (median 2 840 ms / step) |
| **Throughput**             | **11.5 k tokens / second**, sustained ~98 % GPU util |
| **Peak VRAM**              | **12.0 GB allocated / 12.1 GB reserved** |
| Tokens trained             | **98 M** (~0.24× Chinchilla for 416 M) |
| Train loss                 | 11.02 → **4.58** |
| **Best val loss**          | **4.105** (perplexity **60.7**) at step 2 800 |
| Checkpoint size            | 2.5 GB DCP shard (bf16 weights + AdamW state + loader state) |
| Stack exercised            | model · 3D mesh · streaming loader · DCP ckpt · spike monitor · AdamW + cosine |

## Training curve

![training curves](../out/gpt_416m_fweb_5060ti/loss.png)

Three panels: loss (train EMA + per-step + val), cosine LR schedule, and
step time.

- **Loss panel**: textbook descent for a fresh-init transformer. Fast drop
  in the first ~200 steps as the model learns the vocab + frequent bigrams
  (11.0 → 6.5), then a long slow climb-down as it actually starts modelling
  text (6.5 → 4.5). Val (orange) tracks train within 0.1 the entire run —
  no overfitting, the model has plenty of capacity left for more tokens.
  The star marks the best val at step 2 800: loss 4.105, ppl 60.7.
- **LR panel**: cosine decay from 3e-4 to 3e-5 over all 3 000 steps, with a
  150-step linear warmup. Min-LR is reached at step ~2 850.
- **Step-time panel**: dead flat at 2 840 ms. The only spikes are the
  steps right after checkpoint saves (CUDA cache churn from the DCP writer).
  No drift, no thermal throttling on a sustained run.

That val curve still has slope at step 2 800 — running to 6 000 steps
(another ~2.4 h, ~200 M tokens total) would land near ppl 45 based on
the trend, and 12 000 steps (~4.7 h more, ~400 M tokens, getting near
0.96× Chinchilla) would put it in the 30s.

## What's actually exercised on one GPU

A single rank doesn't run any collectives, but every *other* part of the
distgpt stack is on the critical path of this run:

| Subsystem                             | Exercised? | Notes |
|---------------------------------------|:----------:|-------|
| `model/transformer.py` (Llama-arch)   | ✓          | 24-layer GPT, RoPE, RMSNorm, SwiGLU, GQA 4:1 fwd+bwd every step |
| `model/parallel_layers.py` (TP linears) | ✗         | Falls back to plain `nn.Linear` when tp=1 |
| `parallel/mesh.py` (3D DeviceMesh)    | ✓          | Mesh built with shape (dp=1, tp=1, pp=1) |
| `parallel/fsdp.py` (FSDP2 wrap)       | partial    | Policy runs; `fully_shard` is no-op when mesh.size()==1 |
| `parallel/pipeline.py` (1F1B)         | ✗          | Single stage = ordinary forward/backward |
| `data/streaming.py` (resumable loader) | ✓         | Reads uint16 `.bin` shards, advances `LoaderState` every step |
| `data/mixture.py` (weighted sampler)  | partial    | Single source, no mixing |
| `training/optim.py` (AdamW + cosine)  | ✓          | Per-group WD (no decay on norms / biases), warmup + cosine |
| `training/checkpoint.py` (DCP)        | ✓          | Sharded save every 500 steps, `best.txt` tracking, full reload tested |
| `training/stability.py` (SpikeMonitor + RewindController) | ✓ | Active throughout; fixed mid-run (see below) |
| `training/trainer.py`                 | ✓          | The loop itself |
| `eval/harness.py` (ppl + downstream)  | partial    | Held-out ppl runs every 200 steps; downstream tasks not invoked |
| `utils/dist.py` + `utils/logging.py`  | ✓          | World-size=1 path; rank-zero JSONL + console |

For the bits marked ✗ / partial, see `scripts/smoke_2gpu.sh` (real 2-GPU
FSDP/TP exercise) and the multi-node Slurm scripts under `scripts/`.

## Going multi-GPU: genuine 2-GPU FSDP2 over PCIe

The single-GPU run above leaves `parallel/fsdp.py` marked *partial* —
`fully_shard` is a no-op at `mesh.size()==1`. With a **second RTX 5060 Ti**
in the box we can finally light up the real FSDP2 path: cross-GPU
all-gather (forward) and reduce-scatter (backward gradient sync), a
genuinely sharded parameter + optimizer state, and a DCP checkpoint that
is sharded across two ranks and reloads on either.

The catch: these two consumer cards have **no NVLink**. `nvidia-smi
topo -m` reports `PHB` (PCIe host bridge) between them and P2P is `CNS`
(not supported), so we must run with `NCCL_P2P_DISABLE=1` — **every FSDP
collective routes through host memory over PCIe.** That fabric, not the
GPUs, is the bottleneck, and getting positive scaling out of it took two
specific fixes. This section is the calibration story.

### The naive result: 2 GPUs *slower* than 1

First attempt — `configs/gpt_416m_fweb_2gpu.yaml` with `dp=2`,
`micro_batch=4`, `grad_accum=8` (effective batch doubles to 65 536
tok/step because dp=2), everything else stock — was a shock:

| Config (416 M, same seed) | Step time | Aggregate tok/s | Scaling | Per-GPU MFU | Peak VRAM/GPU |
|---|---:|---:|---:|---:|---:|
| 1-GPU baseline            | 2.84 s / 32 k tok | 11.5 k | 1.00× | 16.2 % | 12.0 GB |
| 2-GPU **naive** FSDP      | 8.20 s / 65 k tok | **8.0 k** | **0.68×** ❌ | 5.6 % | 10.7 GB |

Adding a second GPU made the run **32 % slower in aggregate throughput**.
VRAM sharding worked (10.7 GB/GPU vs 12.0 GB — optimizer state is split),
but per-GPU compute efficiency collapsed from 16.4 % to 5.6 %. The model
was spending more time shuffling parameters over PCIe than doing matmuls.

Two things were generating far more collective traffic than necessary:

1. **`reshard_after_forward=True`** (the FSDP2 default). Parameters are
   freed after the forward all-gather, then **re-gathered a second time**
   for the backward pass. On NVLink that overlap is free; over PCIe it
   doubles the all-gather volume.
2. **Ungated gradient accumulation.** The trainer's accumulation loop
   called `loss.backward()` on all 8 micro-steps, and FSDP2 fires a
   **reduce-scatter on every backward** by default. That's 8
   reduce-scatters per optimizer step when only the *last* one matters —
   the other 7 are pure waste.

### Fix 1 — keep params resident (`reshard_after_forward=false`)

`distgpt/parallel/fsdp.py` now takes the flag and threads it into both
`fully_shard` calls:

```python
def apply_fsdp(model, dp_mesh, dtype, reshard_after_forward: bool = True):
    ...
    for blk in model.layers:
        fully_shard(blk, mesh=dp_mesh, mp_policy=mp,
                    reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp,
                reshard_after_forward=reshard_after_forward)
```

Setting it `false` keeps the unsharded params resident after the forward,
so the backward needs **no re-gather**. The cost is one full unsharded
param copy per GPU — for a 416 M bf16 model that's ~0.8 GB, trivially
affordable on 16 GB. **Optimizer state stays sharded either way**, so the
core FSDP memory win is preserved; we're only trading a little param VRAM
for half the collective volume.

### Fix 2 — gate gradient sync to the last micro-step

`distgpt/training/trainer.py`'s accumulation loop now toggles FSDP2's
`set_requires_gradient_sync` so the reduce-scatter fires **once per
optimizer step** instead of 8×:

```python
sync_gate = getattr(model, "set_requires_gradient_sync", None)
for micro in range(accum):
    if sync_gate is not None:
        sync_gate(micro == accum - 1)   # reduce-scatter only on the last
    x, y = loader.next_batch()
    with autocast():
        _, loss = model(x, y)
        loss = loss / accum
    loss.backward()
```

`set_requires_gradient_sync` is a documented no-op on non-FSDP modules,
so this same code path is safe (and free) on the single-GPU run above.

### The optimized result: positive scaling

Same model, same seed — **loss matches the naive run to four decimals
(8.5145), proving the math is unchanged** — but the collective volume is
roughly halved:

| Config (416 M, same seed) | Step time | Aggregate tok/s | Scaling | Per-GPU MFU | Peak VRAM/GPU |
|---|---:|---:|---:|---:|---:|
| 1-GPU baseline            | 2.84 s / 32 k tok | 11.5 k | 1.00× | 16.2 % | 12.0 GB |
| 2-GPU naive FSDP          | 8.20 s / 65 k tok | 8.0 k | 0.68× ❌ | 5.6 % | 10.7 GB |
| 2-GPU **optimized**       | 4.44 s / 65 k tok | **14.8 k** | **1.26×** ✅ | 10.4 % | 12.8 GB |

Step time **halved** (8.20 s → 4.44 s) and aggregate throughput went from
*below* the single GPU to **1.26× above it** — a flip from negative to
positive scaling from two config/code changes, no new hardware.

Read the numbers honestly, though: **1.26× on 2 GPUs is 63 % scaling
efficiency**, and per-GPU MFU is still only 10.4 % — *lower* than the
single GPU's 16.4 %. That gap is the PCIe tax: each GPU gives up ~6 points
of MFU to the cost of talking to its peer over host memory. You come out
ahead in wall-clock (more aggregate tokens/s) but you are paying a real
per-device efficiency penalty for the privilege. **On a fabric with
NVLink this tax largely vanishes** — the same code on an A100/H100 NVLink
pair would land much closer to 1.8–1.9× — which is precisely the lesson
distgpt exists to teach: the model code is identical, the *interconnect*
is what separates a consumer pair from a real training node.

### The knobs compete: bigger micro-batch OOMs

The obvious next lever for a comm-bound run is a **larger micro-batch**
(more compute per all-gather amortizes the fixed collective cost). We
tried `micro_batch=8, grad_accum=4` (same effective batch, 2× bigger
GEMMs) — and it **OOMed at step 0**, needing 1.54 GB more than the
576 MB free:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.54 GiB.
GPU 0 has a total capacity of 15.48 GiB of which 576.50 MiB is free.
```

This is the direct consequence of Fix 1: `reshard_after_forward=false`
spends ~0.8 GB/GPU keeping params resident, which is **exactly the
headroom a doubled micro-batch needs**. On this PCIe pair the two
memory knobs *compete for the same VRAM*, and the comm-light setting wins
the tradeoff. The proven optimum is therefore **`micro_batch=4`,
`grad_accum=8`, `reshard_after_forward=false`**: 14.8 k tok/s, 10.4 %
MFU, 12.8 GB peak. (With NVLink you'd leave reshard on, reclaim the
0.8 GB, and run the bigger micro-batch — but then you wouldn't need
either fix.)

### Reproduce

```bash
cd distgpt
export CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/torchrun --standalone --nproc_per_node 2 \
    -m distgpt.cli train --config configs/gpt_416m_fweb_2gpu.yaml \
    --data data/fineweb-edu
```

…or just run [`scripts/run_5060ti_2gpu.sh`](../scripts/run_5060ti_2gpu.sh),
which sets the env above, links the FineWeb-Edu shards, and launches the
two-rank `torchrun` for you.

The full 4 500-step run is ~5.5 h of wall-clock for ~295 M tokens
(0.71× Chinchilla) — a genuinely 2-GPU-trained, sharded-DCP checkpoint.

## Two things this run actually broke (and how we fixed them)

This wasn't a clean first-try run. Two real bugs surfaced; both fixes are
in this commit.

### 1. SpikeMonitor rewind-loop on a noisy small-batch run

With micro_batch=4 and grad_accum=8 (effective batch 32 sequences =
~33 k tokens) the gradient is noisy. By step ~1 500 train loss had
plateaued around 4.88 with per-step jitter of ~0.3. The old
`SpikeMonitor(sigma=5.0)` computed running std over a 200-step window,
got std ≈ 0.04 once converged, and flagged a routine 0.3 jump as a
~7σ spike. RewindController dutifully reloaded the last checkpoint and
halved the LR scale to 0.5×. The model trained back to the same plateau
in ~100 steps, fired again, halved to 0.25×, again to 0.125×, ... after
11 rewinds `eff_lr` was ~1e-10 and the cosine schedule was a fiction.
The run wasted ~6 hours retraining the same 100 steps in a loop.

Fix (in `distgpt/training/stability.py`):

- **`min_abs_jump=2.0` floor**: spike fires only if loss jumps `>2.0`
  *and* `>5σ`. Routine jitter on a converged loss can never trigger it.
- **`max_rewinds=5` cap**: even on a pathologically spiky run, after 5
  rewinds we stop reducing LR and let the cosine schedule finish.
- **`lr_floor=1e-3`** (was 1e-6): the LR multiplier never collapses
  more than 3 orders below the cosine value.

The clean second half of the run (steps 1 500 → 3 000) has zero spike
events, as expected.

### 2. Resumed log was a mess of duplicate steps

After the rewind-loop bug, the JSONL log had steps 1 500 → 1 800
repeated 14 times (once per rewind iteration, each at a different LR).
The plotter would happily draw all of them — the loss panel looked like
a barcode.

Fix (in `scripts/clean_log.py`): a small script that keeps the *first*
occurrence of each (`step`) tuple — that's the clean cosine-schedule
entry, before any rewind — and drops the rest. Backs up the raw log to
`log.jsonl.raw` for forensics.

```bash
.venv/bin/python scripts/clean_log.py out/gpt_416m_fweb_5060ti/log.jsonl
# wrote 314 unique-step entries (dropped 259 duplicates)
```

Both files (`log.jsonl` and `log.jsonl.raw`) are kept in
`out/gpt_416m_fweb_5060ti/` so you can see what happened.

## Why not 2 ranks on one 5060 Ti?

The section above uses **two physical** 5060 Ti cards. A reasonable
question is whether you can fake multi-GPU FSDP with **two ranks on a
single** card (real all-gathers, half the hardware). It doesn't work on
current NCCL:

- **NCCL 2.28+** hard-rejects two ranks bound to the same physical GPU
  with `Duplicate GPU detected` — even under NVIDIA MPS, which shares
  the CUDA context but doesn't fool NCCL's device-discovery probe.
- **Forcing `backend=gloo`** works (the wiring under `DISTGPT_BACKEND=gloo`
  + `DISTGPT_COLOCATE_RANKS=1` is in this commit and the launcher is
  `scripts/run_5060ti_2rank.sh`), but FSDP all-gather over gloo on
  loopback is ~50× slower than NCCL: the 3 000-step run extrapolated
  to ~24 hours instead of 1 h 12 min. Not worth the chart.

So colocating two ranks is a dead end — if you want real FSDP
collectives you need two *separate* visible devices, which is exactly the
two-physical-GPU calibration documented above.

## Files

- Config: [`configs/gpt_416m_fweb_5060ti.yaml`](../configs/gpt_416m_fweb_5060ti.yaml)
- 2-GPU config: [`configs/gpt_416m_fweb_2gpu.yaml`](../configs/gpt_416m_fweb_2gpu.yaml) (`dp=2`, `reshard_after_forward=false`)
- Training log (cleaned): [`out/gpt_416m_fweb_5060ti/log.jsonl`](../out/gpt_416m_fweb_5060ti/log.jsonl)
- Raw training log (with rewind duplicates): [`out/gpt_416m_fweb_5060ti/log.jsonl.raw`](../out/gpt_416m_fweb_5060ti/log.jsonl.raw)
- Console log: [`out/gpt_416m_fweb_5060ti_train.log`](../out/gpt_416m_fweb_5060ti_train.log)
- Loss chart: [`out/gpt_416m_fweb_5060ti/loss.png`](../out/gpt_416m_fweb_5060ti/loss.png)
- Best DCP checkpoint: `out/gpt_416m_fweb_5060ti/gpt_416m_fweb_5060ti/ckpts/step_000002800/` (2.5 GB)
- Plot script: [`scripts/plot_distgpt.py`](../scripts/plot_distgpt.py)
- Log dedup script: [`scripts/clean_log.py`](../scripts/clean_log.py)
- 2-GPU launcher (two physical cards): [`scripts/run_5060ti_2gpu.sh`](../scripts/run_5060ti_2gpu.sh)
- 2-rank-on-one-GPU launcher (gloo): [`scripts/run_5060ti_2rank.sh`](../scripts/run_5060ti_2rank.sh)
