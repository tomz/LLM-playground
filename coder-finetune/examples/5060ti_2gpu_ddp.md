# 5060 Ti example — genuine 2-GPU DDP LoRA (the real-NCCL proof)

A reproducible **two-process, two-GPU data-parallel** LoRA run on a pair of
**RTX 5060 Ti (16 GB, Blackwell, sm_120, native bf16)**. It trains a r=16
LoRA adapter on `Qwen/Qwen2.5-Coder-0.5B` over the hermetic built-in dataset
for 16 optimizer steps — long enough to prove every DDP code path fires on
real hardware, short enough (~4.5 s of training) to be a smoke test rather
than a training job.

**This doc is a correctness proof, not a throughput or quality run.** It is the
GPU escalation of [`tests/test_dist_launch.py`](../tests/test_dist_launch.py),
which proves the `cf_dist` topology contract on the **gloo (CPU) backend with
CUDA hidden** (`device_count()==0` while `world_size==2`). That test shows the
*wiring* is correct without touching a GPU; this run shows the *training step
itself* — real forward / backward / optimizer with a cross-GPU gradient
all-reduce — runs correctly under `accelerate launch --multi_gpu` on two
physical cards talking **real NCCL**.

For the *model-sharding* sibling over the same no-NVLink PCIe fabric (FSDP2
all-gather / reduce-scatter, where the interconnect tax actually shows up in
the numbers), see
[`distgpt/examples/5060ti_416m_fineweb.md`](../../distgpt/examples/5060ti_416m_fineweb.md).
This run is **DDP** — replicate the model, shard the batch — so each card holds
a full 0.5B replica; nothing is sharded across the two GPUs.

## TL;DR

```bash
cd coder-finetune
.venv/bin/pip install -r requirements.txt          # one-time
# fetch the base model once (CPU/network only):
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-Coder-0.5B')"

# two genuine processes, one per RTX 5060 Ti:
scripts/run_5060ti_2gpu_ddp.sh
```

`run_5060ti_2gpu_ddp.sh` exports the no-NVLink env (`NCCL_P2P_DISABLE=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) and `exec`s:

```bash
.venv/bin/accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
    train.py --config configs/lora_2gpu_5060ti.yaml
```

## Headline numbers

| Metric                     | Value |
|----------------------------|------:|
| GPUs                       | **2× RTX 5060 Ti 16 GB** (sm_120, PCIe `PHB`, **no NVLink**, `NCCL_P2P_DISABLE=1`) |
| Backend                    | **NCCL 2.29.7+cuda13.2** (real GPU collectives, not gloo/CPU) |
| Parallelism                | **DDP** — 2 full 0.5B replicas, gradient all-reduce per step |
| Base model                 | Qwen/Qwen2.5-Coder-0.5B (502.8 M params) |
| Method                     | LoRA r=16 / α=32 (**8.80 M** trainable, **1.75 %**) |
| Dataset                    | `builtin` (hermetic, no download — `cf_data.BUILTIN_PAIRS`) |
| Examples / seq len         | 64 / 1024 |
| Effective batch            | 1 micro × grad_accum 2 × **dp 2** = **4 sequences / step** |
| Steps                      | 16 (1 epoch) |
| **Train runtime**          | **4.48 s** (14.3 samples/s, 3.57 steps/s) |
| **Peak VRAM / replica**    | **1 573 MiB allocated / 1 672 MiB reserved** |
| Loss (logged)              | 3.382 → **0.738** (mean `train_loss` 1.404) |
| Mean token accuracy        | 0.615 → **0.894** (peak 0.917 @ step 14) |
| Adapter checkpoint         | 34 MB (`adapter_model.safetensors`), saved **once** (rank-0 guard) |

## The DDP evidence (verbatim from the log)

The whole point of this run is the log proving two genuine GPU processes
coordinated through NCCL. Five things to look for in
[`out/lora_2gpu_5060ti_ddp.log`](../out/lora_2gpu_5060ti_ddp.log):

**1. Real NCCL backend** (not the gloo/CPU stand-in the unit test uses):

```
NCCL version 2.29.7+cuda13.2
```

**2. Two genuine per-rank processes.** Model load, LoRA wrap, and the
trainable-param banner all print twice — once per card — and the
`[rank0]` / `[rank1]` tags are PyTorch's distributed reducer talking:

```
trainable params: 8,798,208 || all params: 502,830,976 || trainable%: 1.7497
trainable params: 8,798,208 || all params: 502,830,976 || trainable%: 1.7497
[rank1]:[W ... reducer.cpp:1528] ... find_unused_parameters=True was specified in DDP constructor ...
[rank0]:[W ... reducer.cpp:1528] ... find_unused_parameters=True was specified in DDP constructor ...
```

**3. The DDP constructor fired.** That `reducer.cpp` line is emitted only by
`torch.nn.parallel.DistributedDataParallel` — it is direct proof TRL/accelerate
wrapped the model in DDP and installed the gradient-all-reduce hooks across the
two cards. (The `find_unused_parameters=True` note is benign here — LoRA freezes
the base weights, so the reducer's extra graph traversal finds nothing unused;
it's a perf nit, not a correctness issue.)

**4. Loss actually moves** — the optimizer step works end-to-end under DDP,
not just the wiring. Eight logged points at `log_every=2` over the 16 steps:

```
loss=3.382  lr=2.00e-4    acc=0.615
loss=2.272  lr=1.733e-4   acc=0.652
loss=1.259  lr=1.467e-4   acc=0.797
loss=1.165  lr=1.20e-4    acc=0.847
loss=0.950  lr=9.333e-5   acc=0.876
loss=0.858  lr=6.667e-5   acc=0.876
loss=0.610  lr=4.00e-5    acc=0.917
loss=0.738  lr=1.333e-5   acc=0.894
```

A clean cosine descent (2e-4 → ~0 with 3 % warmup) — exactly the single-GPU
curve shape, because **DDP changes the batch wiring, not the math.** It's a
64-example memorize set, so the absolute loss is meaningless; what matters is
that gradients flowed, all-reduced, and the step landed.

**5. The rank-0 save guard held.** The tokenizer + adapter are written by the
main process only — one `saved ->` line, and `final/` contains exactly one file
set (no `WORLD_SIZE`× duplication / interleaving):

```
[train] saved -> out/lora_2gpu_5060ti/final
[vram] peak_alloc=1573 MiB  peak_reserved=1672 MiB
```

## Why 0.5B + the built-in set

This is deliberately the *smallest* thing that exercises the *whole* DDP path:

- **0.5B LoRA is ~1.6 GB/replica** (`peak_reserved=1672 MiB`), so both replicas
  fit with room to spare even when the cards are otherwise busy, and the model
  loads in seconds.
- **`dataset.source: builtin`** needs no download (`cf_data.BUILTIN_PAIRS`), so
  the run is hermetic — the only fetch is the base model. Re-runs are offline.
- **64 examples / eff-batch 4 = 16 steps** is enough for loss to move and every
  collective to fire, fast enough to be a smoke test (~4.5 s of training).

The adapter shape (r=16, α=32, all 7 projection modules) is identical to the
single-GPU 3050/5060 Ti recipes, so this run is directly comparable to its
single-process sibling — only the batch wiring differs.

## What this proves vs. doesn't

**Proves (on real hardware, real NCCL):**
- `accelerate launch --multi_gpu --num_processes 2` spawns two genuine
  per-card processes and TRL's `Trainer` owns the process group end-to-end.
- The full training step — forward, backward, **cross-GPU gradient
  all-reduce**, optimizer — runs correctly under DDP; loss descends.
- `cf_dist` reads the launcher-published topology correctly *with GPUs
  present*: the unit test pins `WORLD_SIZE==2` while `device_count()==0`
  (CUDA hidden); this run is the inverse — two real visible cards — and the
  rank-0 guards, single tokenizer save, and per-rank replica all behave.
- The no-NVLink PCIe pair (`PHB`, `P2P=CNS`) trains fine with
  `NCCL_P2P_DISABLE=1` routing the all-reduce through host shared memory.

**Doesn't (by design):**
- **Throughput / scaling numbers.** 16 steps on a memorize set says nothing
  about tokens/s or DDP scaling efficiency. For an honest PCIe-fabric scaling
  study (and the two fixes that flip a no-NVLink pair from negative to positive
  scaling), read the distgpt
  [2-GPU FSDP example](../../distgpt/examples/5060ti_416m_fineweb.md).
- **Model quality.** It's a 64-example smoke set; for a real LoRA with
  held-out generalization see [`5060ti_lora.md`](5060ti_lora.md).
- **Model sharding.** This is DDP — every GPU holds a *full* replica. To shard
  one model across GPUs (FSDP2/TP/PP) use `distgpt` / `frontier-platform`.
- **QLoRA device placement.** Plain LoRA lets `accelerate.prepare()` move
  weights per rank; the `device_map={"": local_rank}` path in `cf_dist` is the
  QLoRA-only branch, exercised by `configs/qlora.yaml`.

## Reproducing

```bash
cd coder-finetune
rm -rf out/lora_2gpu_5060ti
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-Coder-0.5B')"  # one-time
scripts/run_5060ti_2gpu_ddp.sh 2>&1 | tee out/lora_2gpu_5060ti_ddp.log
```

The 0.5B base weights are ~1 GB (one-time download, cached at
`~/.cache/huggingface/hub/`). After that the run is fully offline.

Drop-in variations to try:
- `scripts/run_5060ti_2gpu_ddp.sh configs/other.yaml` — any 2-GPU config.
- Bump `dataset.max_examples` / `train.epochs` to turn the smoke run into an
  actual short training job.
- `torchrun --standalone --nproc_per_node 2 -m cf_rl.grpo_train --config configs/grpo_3050.yaml`
  — the same DDP plumbing under GRPO (TRL reads `WORLD_SIZE` either way).

## Files

```
configs/lora_2gpu_5060ti.yaml       # the 2-GPU DDP recipe
scripts/run_5060ti_2gpu_ddp.sh      # launcher (sets no-NVLink NCCL env, accelerate launch)
out/lora_2gpu_5060ti_ddp.log        # run log — the DDP/NCCL evidence quoted above
out/lora_2gpu_5060ti/
└── final/                          # saved adapter (34 MB), written once by rank 0
    ├── README.md                   # PEFT auto-generated model card
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── chat_template.jinja
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── training_args.bin
```

## See also

- [`tests/test_dist_launch.py`](../tests/test_dist_launch.py) — the CPU/gloo
  topology proof this run escalates to real GPUs.
- [Multi-GPU (single-node DDP)](../README.md#multi-gpu-single-node-ddp) — the
  README section on what makes DDP correct here (`cf_dist.py`).
- [`distgpt/examples/5060ti_416m_fineweb.md`](../../distgpt/examples/5060ti_416m_fineweb.md)
  — the model-sharding (FSDP2) counterpart over the same PCIe fabric.
