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
| **`configs/dpo_3050.yaml`** | **Qwen2.5-Coder-0.5B** | **DPO (LoRA), offline preference pairs** | **~4–5 GB** | **fast (no sampling)** |
| **`configs/grpo_3050.yaml`** | **Qwen2.5-Coder-0.5B** | **GRPO / RLVR (LoRA), unit-test reward** | **~5–6 GB** | **gen-heavy**       |

Uses HuggingFace `transformers` + `peft` + `trl`; no custom training loop.

## Three training tracks — the post-training ladder

`SFT  →  DPO/ORPO  →  GRPO/RLVR` — cheapest and most stable first.

1. **SFT (`train.py`)** — supervised fine-tune on demonstrations (full / LoRA /
   QLoRA via TRL `SFTTrainer`). Teaches format and style.
2. **DPO/ORPO (`cf_pref/dpo_train.py`)** — *offline preference optimization*.
   Takes a fixed dataset of `(prompt, chosen, rejected)` pairs and raises the
   policy's log-prob margin of `chosen` over `rejected` — no reward model, no
   sampling, no code execution. **DPO** (Rafailov et al.) measures the margin
   against a frozen reference; with a LoRA adapter the reference is just the base
   model with adapters disabled, so no second model copy is loaded. **ORPO**
   (Hong et al.) folds the same signal into SFT with an odds-ratio penalty and is
   reference-free. This is the cheap, stable rung you run *before* online RL.
3. **RLVR / GRPO (`cf_rl/grpo_train.py`)** — *RL against a verifiable reward*:
   sample G completions per prompt, **run each against hidden unit tests**,
   standardize the rewards within the group, take a clipped policy step (GRPO —
   DeepSeek-R1 / DeepSeekMath). The reward is deterministic (a verifier), so it
   can't be gamed the way a learned reward model can. Reuses the same LoRA/QLoRA
   plumbing as `train.py` and the HumanEval subprocess sandbox as the verifier.

```bash
# SFT first (optional), then DPO, then GRPO on top:
.venv/bin/python -m cf_pref.dpo_train --config configs/dpo_3050.yaml
.venv/bin/python -m cf_rl.grpo_train  --config configs/grpo_3050.yaml
.venv/bin/python eval/run_humaneval.py --model out/grpo_3050/final
```

The DPO preference set carries two candidate answers per prompt instead of one
gold answer (`cf_pref/pairs.py`: a dependency-free `builtin` set that pairs each
task's correct solution against a realistic near-miss bug, or any HF preference
set in `{prompt, chosen, rejected}` form). The built-in `chosen`/`rejected`
labels are cross-checked against the RLVR unit-test verifier in the test suite,
so the preference signal is provably correct.

> **TRL note:** TRL 1.x removed the standalone `ORPOTrainer`. `pref.objective:
> dpo` works on every supported TRL; `orpo` raises a clear, actionable error if
> your TRL doesn't ship it (pin `trl<0.12` for ORPO).

The GRPO prompt set carries unit tests instead of gold answers
(`cf_rl/prompts.py`: a dependency-free `builtin` set, or real `mbpp` tasks).
Reward functions live in `cf_rl/reward.py` (correctness verifier always on;
optional format / length shaping, mirroring frontier-platform's
`CompositeReward`). **GRPO executes model-generated code every step — that *is*
the reward — so run untrusted models inside Docker/gVisor.**


## Speed & quality knobs (opt-in, config-driven)

Recent SOTA add-ons that plug into the existing TRL/PEFT stack. All default
**off** for clean A/B comparisons; the `lora_5060ti.yaml` recipe turns them on.

| Knob | Where | Effect | Cost |
|------|-------|--------|------|
| **Liger Kernel** | `train.use_liger_kernel: true` | Fused Triton RMSNorm/RoPE/SwiGLU + FusedLinearCrossEntropy. ~20% faster, up to ~60% less memory. The fused linear-CE is a big deal over Qwen's ~150K vocab (never materializes full logits). *Exact*, not approximate. | needs `pip install liger-kernel` + Triton GPU |
| **DoRA** | `lora.use_dora: true` | Weight-decomposed LoRA — better quality at low rank (our r=16). | ~10–20% slower step |
| **rsLoRA** | `lora.use_rslora: true` | `alpha/sqrt(r)` scaling so higher ranks actually help. | free |
| **NEFTune** | `train.neftune_noise_alpha: 5` | Embedding-noise regularizer; better instruction-following. | free, train-only |
| **Unsloth** | `model.use_unsloth: true` | ~2× faster / ~70% less memory fast-path via custom kernels. Replaces the loader; great for the 7B QLoRA recipe. | heavier dep; `pip install unsloth` |

```bash
# 5060 Ti recipe now ships with Liger + DoRA + rsLoRA + NEFTune enabled:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/lora_5060ti.yaml
```

## Layout

```
coder-finetune/
├── configs/        # YAML per recipe (SFT + dpo_3050.yaml + grpo_3050.yaml)
├── cf_data/        # SFT dataset loaders (HF datasets + your own repo + synthetic)
├── cf_pref/        # DPO/ORPO: preference pairs + dpo_train.py
├── cf_rl/          # RLVR/GRPO: verifiable reward + prompt sets + grpo_train.py
├── train.py        # SFT / LoRA / QLoRA via TRL SFTTrainer
├── eval/           # HumanEval+ runner with Docker sandbox (also the RL verifier)
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
