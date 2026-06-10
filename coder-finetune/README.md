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
   reference-free. **SimPO** (`pref.objective: simpo`) is a third, reference-free
   pairwise objective routed through TRL's `DPOTrainer` (`loss_type: simpo`) — no
   reference model to sync. **KTO** consumes *unary* desirable/undesirable labels
   instead of pairs; its reference loss lives in `cf_pref/objectives.py` and a
   binary data adapter in `cf_pref/binary.py` derives a tiny KTO set from the
   built-in pairs. This is the cheap, stable rung you run *before* online RL.
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
`CompositeReward`). Two **DAPO** add-ons are config-gated: **overlong reward
shaping** (`grpo.overlong_shaping: true`) ramps a smooth penalty as a completion
approaches the hard token budget instead of a single cliff, and
**`dynamic_sampling_mask`** drops all-correct / all-wrong groups that carry zero
relative-advantage signal. The DAPO `epsilon_high` decoupled-clip knob is exposed
via `grpo.epsilon_high`. **GRPO executes model-generated code every step — that
*is* the reward — so run untrusted models inside Docker/gVisor.**


## Sandbox & safety

The HumanEval eval and the GRPO reward both `exec()` model-generated code
locally — the eval to score it, GRPO *every step* because the verifier is the
reward. The README has always said "not a security boundary, run untrusted
models in Docker"; the in-process executor (`eval/run_humaneval.py`) is a
**safety floor**, not a jail. Concretely, each program runs in its own
subprocess with (POSIX, best-effort):

| Bound | Default | What it stops |
|-------|---------|---------------|
| `RLIMIT_FSIZE` | `0` | writing files to disk |
| `RLIMIT_NOFILE` | `64` | exhausting file descriptors |
| `RLIMIT_CPU` | `2× timeout` | CPU-pegging busy loops (backstop under the wall-clock kill) |
| `RLIMIT_AS` | `1 GiB` *(spawn only)* | runaway memory allocation |
| wall-clock `timeout` | `5 s` | hangs / infinite loops (the primary kill) |
| stdout/stderr | silenced | a chatty `print()` flooding the trainer log |

What it does **not** do: it is not a syscall jail, not a network sandbox, and
not a container. For *untrusted* models, two escalations:

- **`mp_mode='spawn'`** (`run_one(..., mp_mode='spawn')`) — a fresh interpreter
  that doesn't inherit the parent's heap / fds / imported modules. ~100 ms vs
  ~10 ms per call; the default stays `fork` for speed since the rlimit floor
  applies in both modes. Spawn also gets the `RLIMIT_AS` memory ceiling (a fork
  child inherits the parent's multi-GiB address space, so bounding it would
  SIGKILL before `exec()` even runs).
- **Docker / gVisor / Firecracker** — the real boundary. Wrap the whole eval /
  trainer for anything you didn't train yourself.

Kill causes are surfaced in the result message (`"timeout"`,
`"killed-SIGKILL"`, `"killed-SIGXFSZ"`, …) so a debugging run can tell an
rlimit kill from a wall-clock timeout from an ordinary test failure.


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
| **vLLM rollouts (GRPO)** | `grpo.use_vllm: true` | Delegates GRPO's rollout generation to vLLM (paged attention + continuous batching). ~3–8× faster rollouts — the single biggest GRPO speedup, since the step is generation-heavy (G samples/prompt). | needs `pip install vllm`; one-time engine warm-up |

```bash
# 5060 Ti recipe now ships with Liger + DoRA + rsLoRA + NEFTune enabled:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/lora_5060ti.yaml
```

## Layout

```
coder-finetune/
├── configs/        # YAML per recipe (SFT + dpo_3050.yaml + grpo_3050.yaml)
├── cf_data/        # SFT dataset loaders (HF datasets + your own repo + synthetic)
├── cf_pref/        # DPO/ORPO/SimPO pairs + KTO objectives/binary adapter + dpo_train.py
├── cf_rl/          # RLVR/GRPO: verifiable reward + prompt sets + grpo_train.py
├── train.py        # SFT / LoRA / QLoRA via TRL SFTTrainer
├── eval/           # HumanEval+ runner with Docker sandbox (also the RL verifier)
├── infer/          # merge LoRA, export for vLLM
└── tests/          # 74 tests — bug regressions, sandbox, DPO/SimPO/KTO/GRPO pins
```

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Tiny full-finetune of 0.5B on a small built-in dataset
.venv/bin/python train.py --config configs/tiny.yaml

# 2. Eval HumanEval+ pass@1
.venv/bin/python eval/run_humaneval.py --model out/tiny --n-samples 1

# 2b. Reproducible pass@5, saving every completion + a machine-readable summary
.venv/bin/python eval/run_humaneval.py --model out/tiny \
    --n-samples 5 --temperature 0.8 --seed 0 \
    --save-completions out/tiny/completions.jsonl \
    --json-out out/tiny/eval.json

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

## Multi-GPU (single-node DDP)

The three entry points (`train.py`, `cf_pref/dpo_train.py`, `cf_rl/grpo_train.py`)
run unchanged under **data-parallel multi-GPU** — launch with `accelerate` or
`torchrun` and TRL's `Trainer` (built on HuggingFace `accelerate`) owns the
process group. This is **DDP, not model sharding**: each GPU holds a full model
replica and processes a different slice of the batch, so the effective batch
scales with GPU count. To shard one model across GPUs, use `distgpt`.

```bash
# 2-GPU LoRA SFT (each rank = one card; effective batch ×2):
accelerate launch --multi_gpu --num_processes 2 \
    train.py --config configs/lora_5060ti.yaml

# 2-GPU GRPO (torchrun also works; TRL reads WORLD_SIZE either way):
torchrun --standalone --nproc_per_node 2 \
    -m cf_rl.grpo_train --config configs/grpo_3050.yaml
```

What makes this correct under DDP — all in `cf_dist.py`, a read-only view of the
topology `accelerate`/`torchrun` publish (it never calls `init_process_group`):

- **World size from `WORLD_SIZE`, not `torch.cuda.device_count()`.** GRPO's
  group-divisibility validator must match TRL's `num_processes`. `device_count()`
  diverges in *both* directions — a single process on a 2-GPU box over-counts
  (it's really world size 1), and one-process-per-GPU under-counts (each process
  sees 1 card but the job is world size 2) — so it would both miss real
  mismatches and reject valid configs.
- **QLoRA placement: `device_map={"": local_rank}`.** bitsandbytes pins 4-bit
  weights to a device *at load time* and accelerate can't relocate them, so each
  rank must load its quantized replica onto its own GPU. Only the quantized path
  sets this; plain LoRA / full FT let `accelerate.prepare()` do the device move.
- **Rank-0-guarded prints + single tokenizer save.** Status lines fire once
  (not `WORLD_SIZE`×, interleaved), and the tokenizer is written by the main
  process only. On a single-process run `is_main` is True and `world_size` is 1,
  so the byte-for-byte single-GPU behavior documented above is unchanged.

## What this is NOT

- Not a from-scratch trainer (use `nanogpt-edu` / `midgpt` / `distgpt`).
- Not a *model-sharding* trainer: single-node **multi-GPU DDP** works (replicate
  the model, shard the batch — see [Multi-GPU](#multi-gpu-single-node-ddp)), but
  each GPU still holds a full model replica. To shard one model across GPUs
  (FSDP/TP/PP) or train 70B+, use `distgpt` / `frontier-platform`.
- Not safe to run untrusted generated code outside the provided Docker sandbox.

## Recent changes (Tier 8)

Hardening + frontier-toolbox pass, four hermetic commits (24 → 66 tests):

- **8.1 — bug fixes** (+13): `extract_code` now recovers code from prose-prefixed
  outputs (a correct answer used to score 0 on a `SyntaxError`); the format
  reward credits `async def`; eval `--n-samples` actually does pass@k and passes
  `eos_token_id`; `infer/generate.py` uses `torch_dtype=` for the pinned TRL
  range; and a GRPO divisibility validator fails fast instead of letting an
  opaque tensor-shape error surface minutes into a run.
- **8.2 — subprocess sandbox hardening** (+13): `resource.setrlimit` floor,
  two-layer stdout/stderr silencing, `mp_mode='spawn'` for untrusted models, and
  a `q.get(timeout=...)` queue-drain fix that removed a ~5% reward flake. See
  **Sandbox & safety** above. (A threaded parallel `run_many` was investigated
  and reverted — real 4× speedup but a 20%+ fork-from-threads `mp.Queue` flake;
  kept sequential with an order-preservation pin for any future safe rewrite.)
- **8.3 — frontier opt-ins + pedagogical pins** (+16): `grpo.use_vllm` knob,
  eval `--save-completions` (JSONL) + `--json-out` summary, and pins that assert
  *what the algorithms do* — GRPO group standardization
  (`[1,0,0,1] → [+1,-1,-1,+1]`, baseline-invariant), DPO margin monotonicity
  (one step raises the chosen-vs-rejected margin, on a hand-built tiny Llama so
  the suite needs no download), and an `extract_code` fence round-trip.
- **8.4 — docs** — this section, the Sandbox & safety section, and the knob/CLI
  tables above.
