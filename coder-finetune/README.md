# coder-finetune

Fine-tune open-weights code models on a single consumer GPU. Three tiers:

| Config            | Base                     | Method | VRAM target | Time (RTX 3050)  |
|-------------------|--------------------------|--------|------------:|-----------------:|
| `configs/tiny.yaml`  | Qwen2.5-Coder-0.5B    | full FT (BF16) |  ~6 GB |  ~30 min on 5k  |
| `configs/lora.yaml`  | Qwen2.5-Coder-1.5B    | LoRA r=16     |  ~7 GB |  ~1.5 h on 20k  |
| `configs/qlora.yaml` | Qwen2.5-Coder-7B      | QLoRA NF4 r=32|  ~7 GB |  ~8 h on 50k    |

Uses HuggingFace `transformers` + `peft` + `trl`; no custom training loop.

## Layout

```
coder-finetune/
├── configs/        # YAML per recipe
├── data/           # dataset loaders (HF datasets + your own repo + synthetic)
├── train.py        # SFT / LoRA / QLoRA via TRL SFTTrainer
├── eval/           # HumanEval+ runner with Docker sandbox
├── infer/          # merge LoRA, export for vLLM
└── tests/
```

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Tiny full-finetune of 0.5B on a small built-in dataset
.venv/bin/python train.py --config configs/tiny.yaml

# 2. Eval HumanEval+ pass@1
.venv/bin/python eval/run_humaneval.py --model out/tiny --n-samples 1

# 3. Generate a sample
.venv/bin/python infer/generate.py --model out/tiny --prompt 'def fib(n):'
```

## What this is NOT

- Not a from-scratch trainer (use `nanogpt-edu` / `midgpt` / `distgpt`).
- Not multi-GPU (single 3050 / 4090 / A100; for 70B+ see `frontier-platform`).
- Not safe to run untrusted generated code outside the provided Docker sandbox.
