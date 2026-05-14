# LLM-playground

A monorepo of five self-contained PyTorch projects that walk the full
educational arc of building, training, fine-tuning, and serving GPT-class
language models — from a ~10M-parameter character-level toy you can train on
a laptop CPU, up to an architecture-only blueprint for a frontier-scale
(500B+) training platform.

Each subproject is **independent**: its own README, its own dependencies,
its own tests. Pick the one that matches the scale you care about.

## The five projects

| Project | Scale | What it teaches | Hardware |
|---|---|---|---|
| [`nanogpt-edu/`](./nanogpt-edu) | 10M–100M | A correct transformer + training loop in ~500 lines: RoPE, RMSNorm, SwiGLU, AMP, cosine LR. | 1 GPU or CPU |
| [`midgpt/`](./midgpt) | 124M–1.5B | GPT-2 scale with the real production toolbox: `tiktoken` BPE, gradient checkpointing, gradient accumulation, DDP, resumable runs, HellaSwag eval. | 1–8 GPUs, single node |
| [`distgpt/`](./distgpt) | 1B–70B | Real multi-node training: FSDP2 + Tensor Parallel + Pipeline Parallel on a 3D device mesh, sharded DCP checkpoints, loss-spike rewind, streaming dataloader. | Multi-node cluster |
| [`coder-finetune/`](./coder-finetune) | 0.5B–7B | Post-training on a single consumer GPU: full FT, LoRA, and QLoRA via HuggingFace `transformers` + `peft` + `trl`. HumanEval+ in a Docker sandbox. | 1 consumer GPU (≥6 GB) |
| [`frontier-platform/`](./frontier-platform) | 1B–500B+ | Architecture-only blueprint: data acquisition → filtering → dedup → tokenizer → pretrain → SFT → RLHF/DPO → eval → red-team → serving → observability. Interfaces + design docs; bodies are `NotImplementedError`. | Design doc; no GPUs required |

## The complexity ladder

The projects are designed to be read in order. Each one reuses the
vocabulary of the previous and adds **one** production concern:

```
nanogpt-edu  →  midgpt        →  distgpt          →  coder-finetune    →  frontier-platform
  minimal       real tokenizer    3D parallelism      post-training         the whole system
  correct       AMP / grad-ckpt   DCP checkpoints     LoRA / QLoRA          around training
  transformer   single-node DDP   spike rewind        HumanEval+
```

`coder-finetune` is the orthogonal track: instead of pretraining from
scratch, it takes pretrained weights and aligns them for code.
`frontier-platform` zooms back out to show the dozen production systems
that surround the training loop in a real frontier lab.

## Quickstart

Each subproject installs independently. There is no top-level build.

```bash
# Smallest — train a tiny GPT on TinyShakespeare in ~5 minutes
cd nanogpt-edu
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_shakespeare.py
.venv/bin/python train.py --config configs/tiny.py
.venv/bin/python sample.py --ckpt out/ckpt.pt --prompt "ROMEO:"
```

```bash
# GPT-2 scale on one node
cd midgpt
pip install -r requirements.txt
python prepare.py --dataset wikitext103
torchrun --standalone --nproc_per_node 8 train.py --config configs/gpt2_350m.yaml
```

```bash
# Fine-tune a code model on a consumer GPU
cd coder-finetune
pip install -r requirements.txt
python train.py --config configs/lora.yaml
python eval/run_humaneval.py --model out/lora --n-samples 1
```

```bash
# Multi-node FSDP2 + TP + PP
cd distgpt
pip install -e .
# launch via Slurm or torchrun-elastic — see distgpt/scripts/
```

```bash
# Read the blueprint
cd frontier-platform
pip install -e .
$EDITOR docs/00-overview.md
```

## Testing

Every subproject ships `pytest` smoke tests:

```bash
cd <subproject> && pytest
```

Tests run without installing the package — they use a `sys.path` shim so
you can iterate without a reinstall.

## Repository layout

```
LLM-playground/
├── nanogpt-edu/         # 10M–100M, single-file, educational
├── midgpt/              # 124M–1.5B, single-node DDP, tiktoken
├── distgpt/             # 1B–70B, multi-node FSDP2 + TP + PP
├── coder-finetune/      # 0.5B–7B, SFT / LoRA / QLoRA on HF
├── frontier-platform/   # 1B–500B+, architecture blueprint + design docs
├── JAAICODE.md          # AI-assistant project instructions
└── README.md            # this file
```

## Conventions

- Python ≥ 3.10, `from __future__ import annotations`, PEP-604 unions
  (`str | None`), built-in generics.
- `@dataclass` for configs and small value types.
- YAML configs in `configs/`; checkpoints and artefacts in `out/`.
- No cross-subproject imports — each project is deliberately standalone.

## Status

These are study projects. `nanogpt-edu`, `midgpt`, `distgpt`, and
`coder-finetune` are runnable code. `frontier-platform` is a design doc with
typed skeletons — every public function has a signature and a docstring,
but most bodies raise `NotImplementedError`. Running a real frontier model
takes thousands of GPUs and tens of millions of dollars; this repo is the
map, not the territory.

## License

See individual subprojects.
