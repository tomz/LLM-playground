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
├── cf_data/        # dataset loaders (HF datasets + your own repo + synthetic)
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

## Worked example: 84-second LoRA on RTX 3050

A reproducible run with real numbers (1.76 GB peak VRAM, 80 steps, loss
2.85 → 0.45) lives at [`examples/3050_lora.md`](examples/3050_lora.md). It
uses the built-in 16-pair instruction set so no dataset download is needed
beyond the 0.5B base weights.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/lora_3050.yaml
.venv/bin/python infer/generate.py --model out/lora_3050/final --prompt-style raw \
    --prompt 'def gcd(a, b):
    """Return the greatest common divisor of a and b."""
'
```

## What this is NOT

- Not a from-scratch trainer (use `nanogpt-edu` / `midgpt` / `distgpt`).
- Not multi-GPU (single 3050 / 4090 / A100; for 70B+ see `frontier-platform`).
- Not safe to run untrusted generated code outside the provided Docker sandbox.
