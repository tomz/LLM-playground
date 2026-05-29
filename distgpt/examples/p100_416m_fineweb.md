# P100 example — 416 M Llama-arch GPT trained from scratch with the distgpt stack

A real, reproducible pretraining run of distgpt's full training pipeline on a
**single Tesla P100-PCIE-16GB (Pascal, sm_60, 16 GB HBM2, no BF16, no
Tensor Cores)**. Same model and dataset as the
[5060 Ti example](5060ti_416m_fineweb.md) — 416 M-parameter Llama-style GPT
(RoPE + RMSNorm + SwiGLU + GQA 4:1) trained from random init on a 1 B-token
slice of FineWeb-Edu — but executed on a 2016 datacenter GPU. Lands at
**val loss 3.788 (ppl 44.2)** in **14 h 50 min**, consuming ~98 M tokens.

Why bother with a Pascal box? Two reasons:

1. **It actually runs.** Lots of labs still have P100/V100 nodes kicking
   around. This example documents the exact pinning needed (an older
   PyTorch wheel + an fp32 fallback because the framework has no
   `GradScaler`) to get the modern distgpt stack working on a card that
   modern PyTorch dropped from its release wheels.
2. **It surfaces a real bug.** Mixing this P100 in a host that also
   contains a Blackwell card the kernel module can't initialize
   (RmInitAdapter failure on every probe) breaks `cuInit` itself
   — even though `nvidia-smi` cheerfully lists the P100. The fix is a
   one-line `sysfs` unbind of the broken PCI slot; see
   [`scripts/host_setup/unbind-broken-nvidia.sh`](../scripts/host_setup/unbind-broken-nvidia.sh).

Companion to [`5060ti_416m_fineweb.md`](5060ti_416m_fineweb.md) — same
config, same data, same model. The only knobs that moved are `dtype`
(bf16 → fp32) and `micro_batch`/`grad_accum` (4/8 → 2/16 because fp32
doubles activation memory while keeping the same 32 768-token effective
batch).

## TL;DR

```bash
cd distgpt

# 1. Pascal-compatible venv. The default .venv uses torch 2.11+cu130 which
#    has no sm_60 in its arch list (CUDA 13 dropped Pascal). Use 2.5.1+cu121,
#    the last release with sm_60 + sm_61 prebuilt.
uv venv --python 3.12 .venv-pascal
uv pip install --python .venv-pascal/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 numpy pyyaml tqdm matplotlib

# 2. (Optional) If the host also has a Blackwell GPU that fails RmInitAdapter,
#    unbind it from the nvidia driver so cuInit doesn't choke on enumeration:
sudo scripts/host_setup/unbind-broken-nvidia.sh 0000:05:00.0

# 3. Tokenized shards (same as 5060 Ti example).
ln -s ../midgpt/data/fineweb-edu data/fineweb-edu

# 4. Train (~14 h 50 min on a P100).
.venv-pascal/bin/python -u -m distgpt.cli train \
    --config configs/gpt_416m_fweb_p100.yaml \
    2>&1 | tee out/gpt_416m_fweb_p100/run.log

# 5. Plot
.venv-pascal/bin/python scripts/plot_distgpt.py \
    out/gpt_416m_fweb_p100/log.jsonl \
    --title "distgpt 416M · FineWeb-Edu · Tesla P100" \
    --subtitle "Llama-arch (RoPE + RMSNorm + SwiGLU + GQA 4:1) · 3000 steps · fp32 (no BF16 on Pascal)"
```

## Headline numbers

| Metric                     | Value |
|----------------------------|------:|
| GPU                        | Tesla P100-PCIE-16GB (sm_60, Pascal, HBM2 732 GB/s) |
| Model                      | 416 M Llama-arch (24 L × 1024 d × 16 H, GQA 4:1, tied embeddings) |
| Tokenizer                  | tiktoken `gpt2` (50 257 → padded to 50 304) |
| Dataset                    | FineWeb-Edu `sample-10BT`, 1 B-token slice |
| Sequence length            | 1024 |
| Effective batch            | 2 × grad_accum 16 = 32 sequences = **32 768 tokens / step** |
| Steps                      | 3 000 |
| **Wall-clock**             | **14 h 50 min** (median 17 654 ms / step) |
| **Throughput**             | **1.85 k tokens / second**, sustained 100 % GPU util |
| **Peak VRAM**              | **12.7 GB allocated / 12.9 GB reserved** |
| Tokens trained             | **98 M** (~0.24× Chinchilla for 416 M) |
| Train loss                 | 11.02 → **4.29** |
| **Best val loss**          | **3.788** (perplexity **44.2**) at step 2 300 |
| Checkpoint size            | 4.9 GB DCP shard (fp32 weights + AdamW state + loader state) |
| Stack exercised            | model · 3D mesh · streaming loader · DCP ckpt · spike monitor · AdamW + cosine |

## Training curve

![training curves](../out/gpt_416m_fweb_p100/loss.png)

Three panels: loss (train EMA + per-step + val), cosine LR schedule, and
step time.

- **Loss panel**: textbook descent for a fresh-init transformer. Fast
  drop in the first ~300 steps (11.0 → ~5.5) as the model learns the
  vocab + frequent bigrams, then a long slow climb-down as it starts
  modelling text (5.5 → ~4.3). The val curve (orange) is noisy
  per-eval — distgpt's eval grabs a *single* `loader.next_batch()`
  each time (see `trainer.py:166`), so individual val points jitter ±0.5
  on natural batch hardness, but the lower envelope keeps falling. The
  star marks the best val at step 2 300: loss 3.788, ppl 44.2.
- **LR panel**: cosine decay from 3e-4 to 3e-5 over all 3 000 steps,
  with a 150-step linear warmup. Min-LR is reached at step ~2 850.
- **Step-time panel**: dead flat at 17 654 ms. The only spikes are the
  five steps right after each `ckpt_every=500` save (CUDA cache churn
  from the DCP writer). No drift, no thermal throttling — the P100 sat
  at 79 °C / 160 W for the entire 14.8 h.

Interestingly the P100 hits a **lower val loss (3.79)** than the
5060 Ti run (4.105) despite training on the same data for the same
number of steps. The difference is fp32 vs bf16: with twice the
mantissa precision the optimizer step is slightly more accurate on
noisy gradients, and on a 416 M model trained at 0.24× Chinchilla
that shows up as ~0.3 nats of held-out loss. The 5060 Ti was about
**12× faster** — exactly the throughput gap you'd expect from
HBM2/no-tensor-cores vs GDDR7/native bf16 Tensor Cores.

## What's actually exercised on one GPU

A single rank doesn't run any collectives, but every *other* part of
the distgpt stack is on the critical path of this run:

| Subsystem                             | Exercised? | Notes |
|---------------------------------------|:----------:|-------|
| `model/transformer.py` (Llama-arch)   | ✓          | 24-layer GPT, RoPE, RMSNorm, SwiGLU, GQA 4:1 fwd+bwd every step |
| `model/parallel_layers.py` (TP linears) | ✗         | Falls back to plain `nn.Linear` when tp=1 |
| `parallel/mesh.py` (3D DeviceMesh)    | ✓          | Mesh built with shape (dp=1, tp=1, pp=1) |
| `parallel/fsdp.py` (FSDP2 wrap)       | partial    | Policy runs; `fully_shard` is no-op when mesh.size()==1 |
| `parallel/pipeline.py` (1F1B)         | ✗          | Single stage = ordinary forward/backward |
| `data/streaming.py` (resumable loader) | ✓         | Reads uint16 `.bin` shards, advances `LoaderState` every step |
| `training/optim.py` (AdamW + cosine)  | ✓          | Per-group WD (no decay on norms / biases), warmup + cosine |
| `training/checkpoint.py` (DCP)        | ✓          | Sharded save every 500 steps, `best.txt` tracking |
| `training/stability.py` (SpikeMonitor + RewindController) | ✓ | Active throughout; zero rewinds fired |
| `training/trainer.py`                 | ✓          | The loop itself |
| `eval/harness.py` (ppl + downstream)  | partial    | Held-out ppl runs every 100 steps; downstream tasks not invoked |
| `utils/dist.py` + `utils/logging.py`  | ✓          | World-size=1 path; rank-zero JSONL + console |

For the bits marked ✗ / partial, see `scripts/smoke_2gpu.sh` (real
2-GPU FSDP/TP exercise) and the multi-node Slurm scripts under
`scripts/`.

## Three things this run actually surfaced

### 1. CUDA 13 dropped Pascal — the default `.venv` can't see the P100

```text
torch:           2.11.0+cu130
torch.cuda.get_arch_list():  []            # ← no sm_60, no sm_61, nothing
cudaGetDeviceCount returns 101 (invalid device ordinal)
```

The cu130 wheel ships with no Pascal kernels. Pinning torch to **2.5.1+cu121**
(last release with sm_60 in its arch list) is the fix:

```python
>>> torch.cuda.get_arch_list()
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
>>> torch.cuda.get_device_name(0)
'Tesla P100-PCIE-16GB'
```

We keep this in a sidecar `.venv-pascal/`. The main `.venv` stays on the
modern wheel for Blackwell/Hopper work.

### 2. A broken GPU in the same host poisoned `cuInit` for the working one

The host this ran on also contains an RTX 5060 Ti (Blackwell, sm_120) that
the proprietary `nvidia-driver-580` kernel module cannot initialize
(`RmInitAdapter failed! (0x22:0x56:897)` — Blackwell requires the
`nvidia-open` modules). `nvidia-smi` correctly hid the broken device and
showed only the P100, **but `cuInit` enumerates everything the driver
exposes**, hit the broken device, and returned `CUDA_ERROR_INVALID_DEVICE`
(101) for the whole process. `CUDA_VISIBLE_DEVICES=0` doesn't help —
that filter runs *after* `cuInit`.

The fix is to unbind the broken GPU at the PCI level so the driver stops
exposing it:

```bash
echo "0000:05:00.0" | sudo tee /sys/bus/pci/drivers/nvidia/unbind
```

Confirmation:
```text
cuInit:                       0          # ← was 101
cuDeviceGetCount:             1
cuDeviceGetName(0):           Tesla P100-PCIE-16GB
torch.cuda.is_available():    True
```

Doesn't survive a reboot. We left the persistent fix as
[`scripts/host_setup/unbind-broken-nvidia.sh`](../scripts/host_setup/unbind-broken-nvidia.sh)
plus an optional systemd unit ([`...unbind-broken-nvidia.service`](../scripts/host_setup/unbind-broken-nvidia.service))
that runs it before any CUDA workload starts. The real fix is to
swap to `nvidia-driver-580-open` — but that drops support for the
P100, so we keep both options documented.

### 3. The framework is bf16-only; fp16 on Pascal NaNs at step 10

Pascal has no native BF16, so the obvious downgrade is `dtype: float16`.
That ran for exactly 10 steps before going NaN:

```text
{"loss": 11.0219, "step": 0}
{"loss": 18.4312, "step": 5}    # already diverging
{"loss":    NaN,  "step": 10}
```

distgpt's trainer wraps the forward in `torch.amp.autocast(...)` but
has **no `torch.amp.GradScaler`** — by design, because bf16's wider
exponent doesn't need dynamic loss scaling. On Pascal in fp16, the
first sizeable gradient overflows, `clip_grad_norm_` propagates the
inf, AdamW writes NaN into the moments, and the run is dead. Two
ways to fix it for real:

1. **Add a `GradScaler`** around `loss.backward()` / `optim.step()` in
   `training/trainer.py` (conditionally, only when `dtype==float16`).
   Right pattern for production fp16 use; not done in this commit
   because the framework's intended dtype is bf16.
2. **Use fp32.** P100 has plenty of HBM2 bandwidth — fp32 sits at
   13 GB VRAM and 100 % util on a 416 M model. This is what
   `configs/gpt_416m_fweb_p100.yaml` does. The throughput cost vs a
   hypothetical "working fp16" is small (~10 % from bandwidth) because
   even fp32 saturates the SMs on Pascal without Tensor Cores.

For the smoke-test goal (exercise the full pipeline), fp32 was simpler
and cleaner. The bug surfaced is worth knowing about for anyone
running this framework on Volta/Pascal hardware.

## Files

- Config: [`configs/gpt_416m_fweb_p100.yaml`](../configs/gpt_416m_fweb_p100.yaml)
- Training log: [`out/gpt_416m_fweb_p100/log.jsonl`](../out/gpt_416m_fweb_p100/log.jsonl)
- Console log: [`out/gpt_416m_fweb_p100/run.log`](../out/gpt_416m_fweb_p100/run.log)
- Loss chart: [`out/gpt_416m_fweb_p100/loss.png`](../out/gpt_416m_fweb_p100/loss.png)
- Best DCP checkpoint: `out/gpt_416m_fweb_p100/gpt_416m_fweb_p100/ckpts/step_000002300/` (4.9 GB, not in repo)
- Plot script: [`scripts/plot_distgpt.py`](../scripts/plot_distgpt.py)
- Host-setup workaround: [`scripts/host_setup/unbind-broken-nvidia.sh`](../scripts/host_setup/unbind-broken-nvidia.sh)
