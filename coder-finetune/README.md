# coder-finetune

Fine-tune code models on reviewed, versioned data. Start with the
[current repair profile](examples/repair_profile.md) and
[project-local installation/test contract](../docs/testing-and-reproducibility.md).
The tested HF profile requires explicit revisions, supervision and validation;
code evaluation is default-deny without an explicitly provisioned isolated executor.

The hardware/timing table below records earlier configurations and measurements.
The migrated SDPA recipes disable packing and optional fused kernels; historical
speed/VRAM figures are not measurements of the repaired profile. No pretrained
Qwen model was fine-tuned or benchmarked during this repair.

Historical tiers and sizing context:

| Config                  | Base                     | Method | VRAM peak | Wall-clock         |
|-------------------------|--------------------------|--------|----------:|-------------------:|
| `configs/tiny.yaml`     | Qwen2.5-Coder-0.5B       | full FT (BF16)  | ~6 GB     | ~30 min on 5k       |
| `configs/lora_3050.yaml`  | Qwen2.5-Coder-0.5B     | LoRA r=16       | 2.3 GB    | 1m 24s (smoke run)  |
| `configs/lora_3050_1p5b.yaml` | Qwen2.5-Coder-1.5B | LoRA r=16       | 7.5 GB    | 24 min on 2k        |
| **`configs/lora_5060ti.yaml`** | **Qwen2.5-Coder-3B** | **LoRA r=16, packed** | **15.1 GB** | **12 min on 2.5k**  |
| `configs/lora.yaml`     | Qwen2.5-Coder-1.5B       | LoRA r=16       | ~7 GB     | ~1.5 h on 20k       |
| **`configs/lora_hicap.yaml`** | **Qwen2.5-Coder-1.5B** | **LoRA r=256, all-linear (LoRA Without Regret)** | **~11 GB** | **~2 h on 20k**     |
| `configs/qlora.yaml`    | Qwen2.5-Coder-7B         | QLoRA NF4 r=32  | ~7 GB     | ~8 h on 50k         |
| **`configs/dpo_3050.yaml`** | **Qwen2.5-Coder-0.5B** | **DPO (LoRA), offline preference pairs** | **~4–5 GB** | **fast (no sampling)** |
| **`configs/grpo_3050.yaml`** | **Qwen2.5-Coder-0.5B** | **GRPO / RLVR (LoRA), unit-test reward** | **~5–6 GB** | **gen-heavy**       |

Uses HuggingFace `transformers` + `peft` + `trl`; no custom training loop.

## Three training tracks — the post-training ladder

`SFT  →  DPO  →  GRPO/RLVR` — cheapest and most stable first.

1. **SFT (`train.py`)** — supervised fine-tune on demonstrations (full / LoRA /
   QLoRA via TRL `SFTTrainer`). Teaches format and style.
2. **DPO (`cf_pref/dpo_train.py`)** — offline preference optimization on
   `(prompt, chosen, rejected)` pairs, without code execution. The tested adapter
   supports sigmoid DPO. Full fine-tuning uses a frozen copy of the supplied
   model, not an implicit reload of a possibly different base; PEFT uses its
   adapter-disabled base without a second full copy. ORPO and SimPO are rejected
   by this trainer profile rather than mapped to nonexistent TRL loss modes.
   Their standalone mathematical helpers, and the unary KTO reference helpers,
   remain educational experiments—not validated trainer integrations.
3. **RLVR / GRPO (`cf_rl/grpo_train.py`)** — *RL against a verifiable reward*:
   sample G completions per prompt, **run each against hidden unit tests**,
   standardize the rewards within the group, take a clipped policy step (GRPO —
   DeepSeek-R1 / DeepSeekMath). Verifier rewards can still be gamed. Isolation,
   task validity, and reward integrity need independent tests. Code rewards use
   an explicitly provisioned local Docker executor; unavailable execution aborts
   rather than becoming a fabricated zero reward.

```bash
# SFT first (optional), then DPO, then GRPO on top:
.venv/bin/python -m cf_pref.dpo_train --config configs/dpo_3050.yaml
.venv/bin/python -m cf_rl.grpo_train  --config configs/grpo_3050.yaml
.venv/bin/python eval/run_humaneval.py --model out/grpo_3050/final --python-image "${PYTHON_IMAGE:?Set a reviewed local immutable image ID}" --problems examples/python_eval_smoke.json --benchmark local-python
```

The DPO preference set carries two candidate answers per prompt instead of one
gold answer (`cf_pref/pairs.py`: a dependency-free `builtin` set that pairs each
task's correct solution against a realistic near-miss bug, or any HF preference
set in `{prompt, chosen, rejected}` form). The built-in `chosen`/`rejected`
labels are host-authored educational fixtures. Test coverage does not prove
that arbitrary preference labels, generated programs, or reward tasks are correct.

> **Compatibility:** the tested profile supports sigmoid DPO. ORPO/SimPO trainer
> integrations require a separately implemented and validated profile. Do not
> downgrade TRL underneath the current stack or silently discard objective fields.

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

Model-generated Python is default-deny without an explicit
`DockerPythonExecutor`. The evaluator and code-reward entrypoints preflight a
reviewed **local immutable image ID** and local daemon before loading models or
data. No image is pulled, and there is no fallback to host `fork`, `spawn`, or
`exec`. The old `mp_mode`/`start_method` switches are not supported.

The executor requests non-root execution, a read-only root filesystem, no network,
no added capabilities, no-new-privileges, bounded memory/CPU/PIDs/source/output,
a whole-job deadline and cleanup. It mounts only the temporary job inputs, not
the repository, home directory or Docker socket. Results distinguish assessed
success/failure from unavailable or failed infrastructure; unassessed work must
not be scored as an ordinary zero reward. `run_one` returns a typed result, not
the old Boolean/message tuple.

These are implementation and negative-test contracts, **not a security
certification**. Docker access and a trusted local image were unavailable during
this repair, so live containment tests remain explicit opt-in skips. Validate
filesystem/network denial, descendants, output caps, cleanup and test tampering
on the deployment backend before using untrusted outputs. Stronger VM/syscall
isolation may be appropriate. The separate Lean adapter also remains externally
gated. See [the repair profile](examples/repair_profile.md) for safe commands,
local fixtures and the required operator prerequisites.


## Speed & quality knobs (opt-in, config-driven)

Optional PEFT and noise settings remain explicit. Kernel and serving backends
need their own version, numerical and hardware validation; speedups are not
additive or guaranteed. The repaired SDPA baseline disables packing and Liger.

| Knob | Where | Current contract | Validation needed |
|------|-------|------------------|-------------------|
| **Liger Kernel** | `train.use_liger_kernel` | Optional fused-loss/kernel path; off in repaired recipes | Loss/gradient parity and workload memory/throughput |
| **DoRA** | `lora.use_dora` | Alternate adapter parameterization | Held-out quality, regressions and step cost |
| **rsLoRA** | `lora.use_rslora` | Rank-scaled adapters; no automatic quality guarantee | Matched-rank/budget comparison |
| **NEFTune** | `train.neftune_noise_alpha` | Training-time embedding noise | Held-out and retention checks |
| **Unsloth** | `model.use_unsloth` | Rejected by the tested loader profile | Separately reviewed loader/profile |
| **vLLM rollouts** | `grpo.use_vllm` | Optional backend, not live-validated here | Version, behavior-probability, cancellation and throughput evidence |

```bash
# Repaired recipe: packing/Liger off; other PEFT/noise settings remain explicit:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/lora_5060ti.yaml
```

### LoRA Without Regret — when LoRA *equals* full fine-tuning

The recipes above run **r=16**, which is right for a few-k-row smoke set. But
*LoRA Without Regret* (Schulman et al., TML 2025) showed the "regret" people
accept for using LoRA over full fine-tuning (FullFT) is usually a
**mis-configuration**, not a law — get four knobs right and LoRA *matches*
FullFT at **~⅔ the compute**:

1. **All linear layers, not attention-only.** The MLP/MoE matrices carry the
   capacity — attention-only LoRA underperforms even at matched param count.
   *(Every config here already targets `gate/up/down_proj`; a pin in
   `tests/test_lora_without_regret.py` stops anyone narrowing it back.)*
2. **Enough rank for the dataset** — **≈256 for post-training-scale SFT**. This
   is the one knob the r=16 recipes get "wrong" for a real instruction mixture:
   *"for datasets that exceed LoRA capacity, LoRA underperforms FullFT."*
3. **A higher, ~rank-independent LR** (the `1/r` scaling makes the optimal LoRA
   LR roughly rank-independent — so `lora_hicap.yaml` keeps the same `2e-4`, not
   a lowered one), and **effective batch < 32** (LoRA is less batch-tolerant).
4. **RL needs almost no rank** — policy-gradient extracts ~1 bit/episode, so
   `r=1–32` suffices for GRPO/DPO. The RL recipes stay at r=16 *on purpose*; do
   **not** copy the SFT high rank into `cf_rl`/`cf_pref`.

`configs/lora_hicap.yaml` is the worked recipe (r=256, all-linear, rsLoRA,
rank-independent LR, effective batch 16). Mirrors HF TRL's own
[`lora_without_regret`](https://huggingface.co/docs/trl/main/lora_without_regret).

```bash
# Post-training-scale SFT: r=256 all-linear on a real instruction mixture.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py --config configs/lora_hicap.yaml
```

## Layout

```
coder-finetune/
├── configs/        # YAML per recipe (SFT + dpo_3050.yaml + grpo_3050.yaml)
├── cf_data/        # SFT dataset loaders (HF datasets + your own repo + synthetic)
├── cf_pref/        # DPO/SimPO pairs + KTO objectives/binary adapter + dpo_train.py
├── cf_rl/          # RLVR/GRPO: verifiable reward + prompt sets + grpo_train.py
├── cf_dist.py      # read-only view of the DDP topology (WORLD_SIZE/RANK), launcher-set
├── train.py        # SFT / LoRA / QLoRA via TRL SFTTrainer
├── eval/           # HumanEval subprocess verifier; separate jailed Lean adapter and offline comparison
├── infer/          # merge LoRA, export for vLLM
└── tests/          # 106 tests — bug regressions, sandbox, DPO/SimPO/KTO/GRPO + DDP-env/launcher + LoRA-Without-Regret pins
```

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -c constraints-ci.txt -r requirements.txt

# 1. Tiny full-finetune of 0.5B on a small built-in dataset
.venv/bin/python train.py --config configs/tiny.yaml

# 2. Eval HumanEval+ pass@1
.venv/bin/python eval/run_humaneval.py --model out/tiny --n-samples 1 --python-image "${PYTHON_IMAGE:?Set a reviewed local immutable image ID}" --problems examples/python_eval_smoke.json --benchmark local-python

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
| **2× RTX 5060 Ti 16 GB** | [`examples/5060ti_2gpu_ddp.md`](examples/5060ti_2gpu_ddp.md) | **genuine 2-GPU DDP over real NCCL — the GPU escalation of the gloo launcher test** |
| **2× RTX 5060 Ti 16 GB** | [`examples/lora_without_regret_ab.md`](examples/lora_without_regret_ab.md) | **r=16 vs r=256 A/B (LoRA Without Regret) — ties at convergence, r=16 wins at fixed epoch budget; one arm per GPU** |

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

For a **worked 2-GPU run on real hardware** — two RTX 5060 Ti, real NCCL
collectives, with the DDP/rank evidence quoted from the log — see
[`examples/5060ti_2gpu_ddp.md`](examples/5060ti_2gpu_ddp.md). It's the GPU
escalation of the gloo/CPU launcher test (9.4): the unit test proves the
*wiring*, that run proves the *training step* fires across two physical cards.

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
- Not safe to run untrusted generated Python without an operator-provided
  container or stronger sandbox. The new Docker adapter is for Lean only.

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

## Recent changes (Tier 9)

Single-node **DDP (data-parallel)** for all three entry points (74 → 85 tests),
two commits. Each GPU holds a full model replica and processes a different batch
slice; effective batch scales with GPU count. This is *not* model sharding — see
[Multi-GPU](#multi-gpu-single-node-ddp) and **What this is NOT**.

- **9.1 — `cf_dist.py`, a read-only topology view** (+10): `dist_env()` reads the
  `RANK`/`LOCAL_RANK`/`WORLD_SIZE` that `accelerate`/`torchrun` publish (it never
  calls `init_process_group` — TRL's `Trainer` owns the process group), with a
  single-process fallback that tolerates unset/empty/garbage env values.
  `placement_device_map()` returns `{"": local_rank}` only on the QLoRA path
  (bitsandbytes pins 4-bit weights at load time) and `None` otherwise;
  `rank0_print()` de-dupes status lines to the main process.
- **9.2 — the core fix: world size from `WORLD_SIZE`, not `device_count()`.** GRPO's
  group-divisibility validator must match TRL's `num_processes`. `device_count()`
  diverges in *both* directions — a single process on a 2-GPU box over-counts, and
  one-process-per-GPU under-counts — so it would both miss real mismatches and
  reject valid configs. The validator, QLoRA placement, and rank-0-guarded saves
  are threaded identically through `train.py`, `cf_pref/dpo_train.py`, and
  `cf_rl/grpo_train.py`; **single-process behavior is byte-for-byte unchanged**.
- **9.3 — `tests/test_dist_env.py`** (+10): pins the single-process identity, env
  parsing, QLoRA placement, rank-0 gating, and the cross-module contract that a
  `bs=2 × accum=2 × G=8` config is *invalid at world=1 but valid at world=2* —
  proving the validator consults `WORLD_SIZE`, not the device count.
- **9.4 — `tests/test_dist_launch.py`, the real two-process launcher pin** (+1):
  9.3 sets `os.environ` in *one* process; this spawns **two genuine processes**
  via `python -m torch.distributed.run --nproc_per_node 2` and asserts each reads
  its own distinct topology from `cf_dist` (distinct rank/local_rank, distinct
  QLoRA pinning, one `is_main`), with a real gloo rendezvous + `all_gather`
  confirming they coordinate. Runs on the **gloo (CPU) backend with CUDA hidden**
  → zero GPU memory (safe alongside a live GPU job) and, because
  `device_count()==0` in the workers while `world_size==2`, it can only pass if
  the topology comes from `WORLD_SIZE` — the same fix 9.2 makes, now proven
  end-to-end through the real launcher instead of simulated env vars.
### Evaluation integrity and structured proofs

Two dependency-free evaluator modules implement the July–August harvest:

- `eval/task_contract.py` flags overly strict tests, underspecified prompts,
  low-coverage tests, and prompt/test contradictions with inspectable evidence.
  Reports always require human adjudication; the tool never silently removes a
  benchmark task.
- `eval/structured_proof.py` validates theorem/certificate/citation JSON, checks
  that the certificate targets the requested theorem, and delegates checking to
  an injected external verifier. Formal validity and novelty review are separate
  release conditions. The optional `DockerLeanChecker` now supplies a dedicated
  fail-closed Lean jail; the existing HumanEval subprocess runner is not a jail.

### October–November: bounded offline harvest

See [the runnable guide and measured results](examples/oct_nov_harvest.md) for
commands, trust boundaries, accepted proof syntax, and artifact definitions.
No dependencies were added and neither path downloads models or images.

- **Lean adapter:** `eval/lean_jail.py`, exposed by
  `python -m eval.structured_proof`, requires a trusted local immutable image ID.
  It uses a closed four-certificate language, a host-owned theorem/premise
  wrapper, resource-bounded Docker execution, and hashed evidence. It never
  pulls an image or falls back to host Lean. Real Lean integration is currently
  **unverified** here because Docker daemon access is denied and no local Lean
  image was supplied; the optional integration test skips without one.
- **Specialization smoke:** `eval/specialization.py` compares the same tiny CPU
  classifier baseline against full supervised training and rank-one readout
  LoRA on disjoint train/held-out/regression tasks. All attempts and failures,
  accuracy, calibration, costs, and local model checkpoints are recorded. This
  is educational, **not an LFM3 or language-model benchmark**.

From `coder-finetune/`, using an existing environment with PyTorch:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout 120 .venv/bin/python -m eval.specialization \
  --config configs/specialization_smoke.json \
  --output out/harvest-specialization-smoke

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest -q tests
```

The output directory must be new. The guide also shows reloading a saved local
model and checking `examples/lean_submission.json` against
`configs/lean_contract.json` using an operator-supplied local image.

- **9.5 — the capstone: a real two-GPU NCCL run** ([`examples/5060ti_2gpu_ddp.md`](examples/5060ti_2gpu_ddp.md)).
  9.4 proves the *wiring* on gloo/CPU with CUDA hidden; this escalates it to
  **two physical RTX 5060 Ti on real NCCL** (`2.29.7+cuda13.2`). A r=16 LoRA on
  Qwen2.5-Coder-0.5B trains for 16 steps under `accelerate launch --multi_gpu
  --num_processes 2`: the log shows the `[rank0]`/`[rank1]` DDP constructor
  firing, loss descending 3.382 → 0.738 through a cross-GPU gradient all-reduce,
  the rank-0 guard writing the adapter exactly once, and ~1.6 GB/replica peak —
  the *training step itself*, not just the topology, proven correct across two
  cards over a no-NVLink PCIe pair (`PHB`, `NCCL_P2P_DISABLE=1`).
