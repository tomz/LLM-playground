# distgpt — Distributed multi-node training framework

Real code (not pseudocode) for training **1B–70B** parameter GPTs across **multiple nodes** with:

- **FSDP2** (PyTorch native) full sharding of params + grads + optimizer state
- **Tensor Parallelism** via `torch.distributed.tensor` DTensor (column/row parallel linears)
- **Pipeline Parallelism** via `torch.distributed.pipelining` (1F1B + interleaved schedules)
- **Sequence Parallelism** for memory savings on long context
- **Sharded distributed checkpointing** (DCP) — reshardable across topologies
- **Activation checkpointing** (selective or full)
- **Mixed precision** (BF16 + FP32 master)
- **Streaming dataloader** with deterministic per-rank, mid-epoch resume
- **Eval harness** (lm-eval-harness compatible)
- **W&B + JSONL logging**, loss-spike monitor, async checkpoint upload
- **Slurm + torchrun-elastic** launchers

This code is real and self-consistent. It will not actually finish a 70B run on your laptop — you need a cluster. But every code path is implemented, not stubbed.

## Layout

```
distgpt/
├── distgpt/
│   ├── model/
│   │   ├── config.py          # ModelConfig with param_count()
│   │   ├── transformer.py     # GPT (RoPE, RMSNorm, SwiGLU, GQA)
│   │   └── parallel_layers.py # ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
│   ├── parallel/
│   │   ├── mesh.py            # 3D device mesh (dp, tp, pp)
│   │   ├── fsdp.py            # FSDP2 wrapping policy
│   │   ├── tensor.py          # TP application via parallelize_module
│   │   └── pipeline.py        # PP stage building + schedule
│   ├── data/
│   │   ├── streaming.py       # iterable, resumable shard reader
│   │   └── mixture.py         # weighted multi-source sampler
│   ├── training/
│   │   ├── optim.py           # AdamW + cosine + per-group WD
│   │   ├── checkpoint.py      # DCP sharded save/load (reshardable)
│   │   ├── stability.py       # SpikeMonitor + RewindController
│   │   └── trainer.py         # the loop
│   ├── eval/
│   │   └── harness.py         # ppl + downstream tasks
│   ├── utils/
│   │   ├── logging.py         # JSONL + W&B + rank-zero filter
│   │   └── dist.py            # init/destroy, all_reduce helpers
│   └── cli.py                 # `distgpt train|eval|sample`
├── configs/                   # 1B / 7B / 70B YAML
├── scripts/                   # slurm + torchrun launchers
└── tests/                     # unit tests (single-process where possible)
```

## Reference topologies

| Model | GPUs       | DP | TP | PP | ZeRO | Activation | Throughput target  |
|-------|-----------:|---:|---:|---:|-----:|-----------|--------------------|
| 1B    | 8× H100    | 8  | 1  | 1  | FSDP-3 | none      | 250k tok/s         |
| 7B    | 64× H100   | 8  | 1  | 1  | FSDP-3 | selective | 1.5M tok/s         |
| 7B    | 64× H100   | 4  | 8  | 1  | ZeRO-1 | selective | 1.6M tok/s         |
| 70B   | 512× H100  | 8  | 8  | 8  | ZeRO-1 | selective | 6M tok/s           |

## Quickstart (single node, 8 GPUs, FSDP only)

```bash
pip install -r requirements.txt
# Tokenize some data (or symlink a directory of *.bin uint16 shards)
python -m distgpt.data.prepare_dummy --out data/tiny --tokens 50000000

torchrun --standalone --nproc_per_node 8 -m distgpt.cli train \
    --config configs/1b.yaml --data data/tiny
```

## Multi-node via Slurm

```bash
sbatch scripts/slurm_70b.sbatch
```

## What's *not* in scope

- Custom CUDA kernels (we lean on PyTorch SDPA + Transformer Engine if installed)
- Expert parallelism / MoE
- RLHF (see the `frontier-platform` blueprint)
- Inference engine (use vLLM / TRT-LLM)
