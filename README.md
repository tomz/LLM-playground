# LLM-playground

A monorepo of five self-contained PyTorch projects that walk the full
educational arc of building, training, fine-tuning, and serving GPT-class
language models — from a ~10M-parameter character-level toy you can train on
a laptop CPU, up to an architecture-only blueprint for a frontier-scale
(500B+) training platform.

Each subproject is **independent**: its own README, its own dependencies,
its own tests. Pick the one that matches the scale you care about.

## Results gallery

Two projects in this repo come with **published training plots and headline
numbers** — one from real training on a single consumer GPU, one from a
discrete-event simulator that scales the same physics to a frontier
cluster.

### `nanogpt-edu/` — real training on an RTX 3050

Three runs of a real PyTorch training loop on a single RTX 3050 (8 GB,
bf16) against char-level Tiny Shakespeare (1 MB). The whole sweep,
including a 25 M parameter model overfitting hard for 2 h, was run on a
desktop GPU at home — no cluster, no API.

| Run         | Params  | Iters         | ms/it | Wall    | Final train | **Best val** | Final val | Overfit Δ |
|-------------|--------:|--------------:|------:|--------:|------------:|-------------:|----------:|----------:|
| `smoke`     |  0.86 M | 275 / 300     |  14.0 |   ~5 s  | 1.90        | **1.99**     | 1.99      |   ~0      |
| `tiny`      | 10.65 M | 4,990 / 5,000 | 203.4 | ~17 min | 0.07        | **1.53**     | 4.34      | **2.81**  |
| `tiny_clean`| 10.65 M | 1,500 / 1,500 | 203.0 |  ~5 min | 0.53        | **1.48**     | 1.85      |   0.36    |
| `small`     | 25.73 M | 9,600 / 15,000| 798.4 | ~2 h 15 | 0.04        | **1.87**     | 5.21      |   3.35    |

The classic "best val arrives in the first ~1000 iters and then val
climbs monotonically while train collapses to zero" overfit story
(`tiny`, `small`) — and the textbook **U-shaped** counter-example
(`tiny_clean`: same architecture, `dropout=0.1`, `max_iters=1500`),
which lands on a *better* best-val while overfitting **8× less**:

![nanogpt-edu cross-run comparison](nanogpt-edu/out/compare.png)

Side-by-side: `tiny` (no dropout) vs `tiny_clean` (dropout 0.1) on the
same 10.65 M model and same 1 MB dataset — the only intervention is
regularization + early stopping:

![tiny vs tiny_clean](nanogpt-edu/out/compare_overfit.png)

Per-run 3-panel plots (loss + LR + step time) and the parser that
generated them live at [`nanogpt-edu/out/`](./nanogpt-edu/out/) and
[`nanogpt-edu/tools/plot_nanogpt.py`](./nanogpt-edu/tools/plot_nanogpt.py).
Full discussion in
[`nanogpt-edu/README.md`](./nanogpt-edu/README.md#actual-training-results-1-rtx-3050-8-gb).

### `frontier-platform/` — simulated 1B → 400B program

Discrete-event simulator (pure Python, no torch) that runs the full
program end-to-end: Chinchilla-style scaling laws, MFU → throughput →
wall time, Poisson GPU failures, rolling $ accounting, eval-score
prediction, safety thresholds, serving cost models. Optional
`--real-gpu` flag probes local CUDA devices and recalibrates
`seconds_per_step` from a few real training steps so the simulated wall
clock and $ figures match the silicon you actually own.

| Run          | Cluster      | Wall    | Final loss | MMLU  | Arena ELO | **Total $**  | Throughput model |
|--------------|-------------:|--------:|-----------:|------:|----------:|-------------:|:-----------------|
| `1b`         |    64× H100  |   3.7 d |   2.21     | 50.6% |    1515   | $0.93 M      | 50% MFU × spec   |
| `7b`         |   512× H100  |   4.8 d |   2.02     | 62.7% |    1711   | $1.02 M      | 50% MFU × spec   |
| `70b`        | 4,096× H100  |  13.2 d |   1.88     | 76.8% |    1985   | $3.31 M      | 50% MFU × spec   |
| `400b`       |16,384× H100  |  54.0 d |   1.81     | 84.2% |    2142   | **$42.42 M** | 50% MFU × spec   |
| `7b_realgpu` |   512× H100  | **430.7 d** | 2.03   | 62.7% |    1711   | **$11.48 M** | RTX 3050 bf16 (measured) |

The 7B-vs-7B-realgpu comparison is the punchline: same simulated
cluster, but calibrating against an actually-measured 4.2 TFLOP/s per
RTX 3050 (vs H100's 989 TFLOP/s spec) blows wall-clock from 5 days to
14 months and cost from $1 M to $11.5 M — eval scores are identical
because scaling laws don't care how fast the GPUs are.

![frontier-platform size sweep](frontier-platform/out/sim/compare_sizes.png)

![spec-sheet vs real-GPU calibration](frontier-platform/out/sim/compare_7b_modeled_vs_real.png)

All five runs ship with per-run 3-panel plots (loss + cumulative $ +
cumulative failures), JSON summaries, and a reproducible CLI. See
[`frontier-platform/README.md`](./frontier-platform/README.md#simulator-results-scriptssimulatepy)
for the full story.

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

To run **everything** (pytest in each subproject + ruff at the root) in one
shot:

```bash
python3 tools/orchestrate.py            # tests + lint
python3 tools/orchestrate.py --tests    # tests only
python3 tools/orchestrate.py --lint     # lint only
python3 tools/orchestrate.py -p midgpt  # one project
```

CI mirrors this matrix in `.github/workflows/tests.yml`.

## Repository layout

```
LLM-playground/
├── nanogpt-edu/         # 10M–100M, single-file, educational
├── midgpt/              # 124M–1.5B, single-node DDP, tiktoken
├── distgpt/             # 1B–70B, multi-node FSDP2 + TP + PP
├── coder-finetune/      # 0.5B–7B, SFT / LoRA / QLoRA on HF
├── frontier-platform/   # 1B–500B+, architecture blueprint + design docs
├── tools/orchestrate.py # one-shot test+lint runner across all subprojects
├── pyproject.toml       # shared ruff config (no shared build)
├── .github/workflows/   # CI matrix: pytest each subproject + repo-wide ruff
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
