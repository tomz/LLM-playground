# November repair: supported training and complete restart

The integrated trainer now has an explicit support matrix. Lower-level modules
remain available for experiments, but their existence does not imply arbitrary
composition is correct. See the [repository repair evidence](../../docs/2026-11-repair-pass.md)
and [test profiles](../../docs/testing-and-reproducibility.md).

| Integrated path | Status | Evidence / boundary |
|-----------------|--------|---------------------|
| Single process, FP32; CUDA BF16 autocast | Supported reference | FP32 master parameters; fixed-schedule CPU restart/eval-isolation tests |
| Dense DDP, `zero: none` | Supported reference | Real two-rank CPU/Gloo gradients/updates and shared control paths; larger/GPU jobs need their own evidence |
| Dense CUDA FSDP, `zero: fsdp`, DP-only | Supported restricted reference | Tiny two-GPU FP32/BF16 gradient, clipping, fresh-Adam and actual trainer restart tests |
| FSDP optimizer/checkpoint options | AdamW, `activation_ckpt: none`, no distributed compile | Both reshard modes have low-level parity; actual trainer restart tested with default resharding |
| Dense TP=2, optionally SP | CPU/Gloo FP32 reference | Same DP-owned inputs, nonuniform-data gradient/update parity; untied embeddings, no QK norm/MTP/fused CE |
| PP or combined DP×TP | Rejected by the trainer | Lower-level experiments remain; integrated batch/gradient/restart parity is not established |
| FP16/GradScaler or ordinary-Linear FP8 | Rejected by the trainer | Do not silently downgrade precision or claim an autocast wrapper performs FP8 training |
| Distributed compile, MoE/MLA/MTP in DP | Rejected pending separate integration evidence | Single-process/reference modules are not removed |

Two-GPU evidence is not a 70B, multi-node, throughput, or fault-tolerant service
result. The old large-topology tables are architecture targets, not a validation
matrix. Torch 2.12 is the tested DCP/DTensor family.

## Data and validation

Training shards are assigned by **DP coordinate**, not global rank. TP peers see
the same shard/sample identities. Input manifests hash the underlying files;
loader state records topology, manifest identity, epoch/order and cursor. Full
restart rejects changed data or topology instead of pretending it is exact.

Set `data.val_dir` or `data.val_manifest` for held-out evaluation. Training and
validation must have distinct paths and content. `train.eval_every: 0` explicitly
disables evaluation; there is no fallback that labels a consumed training batch
held-out. Evaluation restores its own cursor, Python/NumPy/Torch/CUDA RNG and
model mode, so changing evaluation cadence cannot change training updates.

## Checkpoints and completed-update boundaries

A checkpoint's step is the **number of completed optimizer updates** and the
next update index. `train.stop_after` provides a controlled interruption without
changing the fixed learning-rate schedule in `optim.total_steps`.

The manager stores normalized model and optimizer state, including fresh Adam
moments and multiple-optimizer state, plus per-rank RNG, loader and controller
state. All ranks participate in capture, validation and publication. Checkpoint
contents are written under staging; a hashed completion manifest and an atomic
same-filesystem rename publish them. Incomplete directories are not `latest`.
Best checkpoints are pinned separately from ordinary retention; unknown names,
symlinks and uncommitted staging paths are not casually garbage-collected.

`latest()` remains a local inspection function. `discover_latest()` collectively
checks visibility before startup resume or spike rewind. Rank disagreement fails
before ranks split into different DCP branches. Full restart is same-topology;
weights-only import is a different, explicit operation. Unmarked legacy state
cannot silently become a cold start. Use a new run ID and explicit
`load_ckpt`/`load_legacy_weights` for a reviewed weights-only migration.

These are shared-filesystem synchronous checkpoints, not asynchronous object-store
transactions. Hashes detect corruption; they do not authenticate an untrusted
checkpoint producer. Use trusted artifacts and a functioning distributed runtime.

## Small runnable recipes

From `distgpt/`, create fresh synthetic train/validation shards:

```bash
test ! -e data/repair
.venv/bin/python -m distgpt.data.prepare_dummy --out data/repair/train \
  --tokens 4096 --shard-tokens 256 --vocab 32 --seed 11
.venv/bin/python -m distgpt.data.prepare_dummy --out data/repair/validation \
  --tokens 1024 --shard-tokens 256 --vocab 32 --seed 13
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m distgpt.cli train --config configs/repair_cpu.yaml
```

The GPU recipe is also tiny. It requires two working CUDA devices and a matching
CUDA/NVML/NCCL runtime; it is not a performance benchmark:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node 2 \
  -m distgpt.cli train --config configs/repair_fsdp.yaml
```

On the audited host, the existing `nvidia-live.sh` runtime wrapper was needed to
select the matching NVML library. No system libraries, driver, permissions or
assistant settings were changed. See the root evidence note for the exact local
validation command. Do not copy host-specific runtime paths into portable code.

## Acceptance commands

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -ra --timeout=180 tests
# Only on idle, correctly configured GPUs; keep an outer job deadline.
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NCCL_P2P_DISABLE=1 \
  timeout --kill-after=10s 180 .venv/bin/python -m pytest -q \
  tests/test_fsdp_cuda.py tests/test_repair_acceptance.py::test_real_fsdp_trainer_interrupted_restart
```

The GPU fixtures are below one million parameters and use only a few updates.
The CPU suite is not substituted for those GPU results. Tests compare reference
gradients, logical clipping norms, fresh optimizer moments, next batches/updates,
publication failures, manifest identity and actual interrupted trainer state.
