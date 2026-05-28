# coder-finetune

Fine-tune open-weights code models on a single consumer GPU. Tiers:

| Config                  | Base                     | Method | VRAM peak | Wall-clock         |
|-------------------------|--------------------------|--------|----------:|-------------------:|
| `configs/tiny.yaml`     | Qwen2.5-Coder-0.5B       | full FT (BF16)  | ~6 GB     | ~30 min on 5k       |
| `configs/lora_3050.yaml`  | Qwen2.5-Coder-0.5B     | LoRA r=16       | 2.3 GB    | 1m 24s (smoke run)  |
| `configs/lora_3050_1p5b.yaml` | Qwen2.5-Coder-1.5B | LoRA r=16       | 7.5 GB    | 24 min on 2k        |
| **`configs/lora_5060ti.yaml`** | **Qwen2.5-Coder-3B** | **LoRA r=16, packed** | **15.1 GB** | **12 min on 2.5k**  |
| `configs/lora.yaml`     | Qwen2.5-Coder-1.5B       | LoRA r=16       | ~7 GB     | ~1.5 h on 20k       |
| `configs/qlora.yaml`    | Qwen2.5-Coder-7B         | QLoRA NF4 r=32  | ~7 GB     | ~8 h on 50k         |

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

## Worked examples

| GPU              | Recipe                                                      | What it shows |
|------------------|-------------------------------------------------------------|---------------|
| RTX 3050 8 GB    | [`examples/3050_lora.md`](examples/3050_lora.md)            | 84 s memorize-a-builtin-set smoke run (0.5B, no download) |
| RTX 3050 8 GB    | [`configs/lora_3050_1p5b.RESULTS.md`](configs/lora_3050_1p5b.RESULTS.md) | 24 min 1.5B + Magicoder, pushing the 8 GB limit |
| **RTX 5060 Ti 16 GB** | [`examples/5060ti_lora.md`](examples/5060ti_lora.md)   | **12 min 3B + Magicoder, packing on, grad_ckpt off** |

The 5060 Ti example is the one to read for current numbers — it shows
both *real generalization* (novel held-out prompts get correct DP / BFS
/ LRU / decorator code) and a clean throughput win from the 16 GB budget
(disable gradient checkpointing, enable packing) versus the 3050 recipes.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/lora_5060ti.yaml
.venv/bin/python infer/generate.py --model out/lora_5060ti/final \
    --prompt 'Write a Python function levenshtein(a, b) ...'
```

## What this is NOT

- Not a from-scratch trainer (use `nanogpt-edu` / `midgpt` / `distgpt`).
- Not multi-GPU (single 3050 / 4090 / A100; for 70B+ see `frontier-platform`).
- Not safe to run untrusted generated code outside the provided Docker sandbox.
