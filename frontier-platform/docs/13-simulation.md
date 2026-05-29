# 13 — Frontier Program Simulation

A discrete-event simulator that models the **entire** frontier-model training program — data ingestion, tokenizer training, pretraining, alignment, eval, safety, and serving — as a self-consistent virtual system. No GPUs required; no PyTorch imported. Hundreds of virtual GPU-days execute in <1s of wall-clock.

This document is the complete reference for the `platform.sim` package and the `scripts/simulate.py` CLI.

---

## 1. Goals

The simulator answers questions that are otherwise **financially or temporally impossible** to answer:

- "What does a 70B run cost in dollars, days, and GPU failures on 4096 H100s?"
- "If I double the preference pairs, by how much does the arena ELO move?"
- "At what model scale do my safety thresholds start failing?"
- "What's the inference $/Mtok at each serving tier under a given QPS load?"
- "How does cost composition shift between human labels and GPU compute as we scale?"

It does **not** train a real model. It produces realistic *aggregate* numbers from first-principles formulas calibrated to public reports (Chinchilla, GPT-4 technical report, Llama-3, etc.).

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        scripts/simulate.py                            │
│           (CLI — picks preset, builds ProgramSpec, prints report)     │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  platform/sim/orchestrator.py                         │
│    run_program(ProgramSpec) → drives every subsystem in order         │
└──────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────┬──────────┬────┴─────┬───────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼           ▼          ▼          ▼
   data_sim    tokenizer  pretrain  alignment   reasoning   eval_sim   serving
                  _sim       _sim       _sim      _rl_sim   +safety       _sim
                                                 (optional)

      All subsystems share three singletons:
        Clock           — virtual time, advanced in seconds
        CostBook        — rolling $ accounting (phase × resource)
        EventBus        — JSONL log of every event for post-hoc analysis

      Plus two long-lived state objects:
        Cluster         — virtual GPU fleet (failures, downtime, $/h)
        ProgramSpec     — immutable run configuration
```

Every subsystem has the same shape:

```python
def simulate_X(spec, ..., clock, cost, bus, seed) -> dict:
    ...
    clock.advance(seconds_consumed)
    cost.charge(phase, resource, dollars_spent)
    bus.emit("X.done", **summary)
    return summary
```

This makes the system trivially extensible: add a new phase, give it the same signature, register it in the orchestrator.

---

## 3. Module reference

### 3.1 `clock.py` — virtual time

```python
class Clock:
    t: float                          # seconds
    def advance(self, dt_seconds): ...
    @property days, hours: float
    def fmt() -> "day  D HH:MM"
```

All subsystems advance the same clock. There is no parallelism in the simulator — phases run strictly in order. (Doing one thing at a time is realistic: data and tokenizer must precede pretrain; eval and safety run on the trained checkpoint.)

### 3.2 `scaling.py` — Chinchilla loss + capability predictors

The core mathematical model. Drives every quantitative result.

#### Loss law (Hoffmann et al. 2022)

```
L(N, D) = E + A · N^-α + B · D^-β
        = 1.69 + 406.4 · N^-0.34 + 410.7 · D^-0.28
```

- `N` = parameters
- `D` = tokens
- `E` = irreducible entropy floor (≈ 1.69 nats)

```python
chinchilla_loss(n_params, n_tokens) -> float
compute_flops(n_params, n_tokens) -> 6 * N * D
step_loss_curve(N, D_total, step, total_steps) -> noisy interpolated loss
```

#### Eval predictors (sigmoid of log-FLOPs)

Each is a logistic curve over `log10(FLOPs)`, calibrated so well-known checkpoints land near their reported scores:

| Predictor          | Floor | Ceiling | Calibration anchor                              |
|--------------------|------:|--------:|-------------------------------------------------|
| `predict_mmlu`     | 0.25  | 0.90    | 1B/1T≈0.32, 7B/2T≈0.50, 70B/2T≈0.69, 400B/15T≈0.85 |
| `predict_humaneval`| 0.05  | 0.90    | scales with code fraction in mix                |
| `predict_gsm8k`    | 0.05  | 0.95    | scales with math fraction in mix                |
| `predict_arena_elo`|  1000 |  ~2400  | weighted blend × SFT × RLHF × reasoning quality multipliers |

These are intentionally **not** physical — they're regressions to public scores. Their virtue is producing the right *relative* movements when you twist the knobs (more tokens → ~+x% MMLU, etc.).

#### Frontier economics helpers (2025-class runs)

Three extra helpers model paradigms that postdate the original dense/Chinchilla
core. They let the simulator price runs on **hardware and recipes we don't
physically have**:

```python
moe_active_params(total_params, n_experts, top_k, shared_experts=1) -> float
    # active (per-token) params for a fine-grained sparse MoE. Treats the FFN
    # as ~2/3 of weights and scales only that share by (top_k+shared)/(experts
    # +shared). Dense models (n_experts<=1) pass through unchanged. The active
    # count — not the total — drives the 6·N·D training/inference FLOPs.

precision_speedup(precision) -> float
    # bf16=1.0, fp8=1.55, nvfp4=2.2 — effective throughput vs bf16 after
    # high-precision accumulation overhead (DeepSeek-V3 FP8, NVIDIA NVFP4).

reasoning_rl_quality(base_cap, rollouts, steps) -> float (>=1.0)
    # multiplier the eval phase applies on top of pretraining-only scores to
    # model an o1/R1-style RLVR phase. Saturates in both rollouts (~1.5M scale)
    # and optimizer steps (~3k scale), gated by base capability, max +30%.
    # Crucially NOT tied to the pretraining FLOP budget — R1-style gains came
    # from RL compute tiny next to pretraining.
```

A 1T-param MoE with 256 experts / top-8 routes through **356.7B active**
params; combined with `fp8` (1.55×) this is what makes a frontier-scale run
finish in simulated days rather than weeks.

### 3.3 `cluster.py` — GPU fleet & failure model

```python
GPU_SPECS = {
    "A100":  {"tflops": 312,  "hbm": 80,  "price": 1.20},
    "H100":  {"tflops": 989,  "hbm": 80,  "price": 2.00},
    "H200":  {"tflops": 989,  "hbm": 141, "price": 2.50},
    "B200":  {"tflops": 2250, "hbm": 192, "price": 4.50},
    # --- frontier-class hardware we don't physically own (simulated) ---
    "GB200": {"tflops": 2500, "hbm": 192, "price": 5.50},  # NVL72 per-GPU
    "B300":  {"tflops": 3300, "hbm": 288, "price": 7.00},  # Blackwell Ultra (projected)
}

GPU_MTBF_HOURS = 87_600    # ~10 years per GPU
NODE_RECOVERY_MIN = 8.0    # spare swap + restart time
```

The last two rows are the **"GPUs we don't have"**: they let you price a run on
a GB200 NVL72 or a projected B300 fleet without owning the silicon. `tflops`
is dense peak BF16; the low-precision speedups (`fp8`, `nvfp4`) stack on top via
`precision_speedup` rather than being baked into the per-GPU figure.

`Cluster.tick(dt, rng)` is called every pretraining log-window. It:

1. Computes node-MTBF = `GPU_MTBF / gpus_per_node`
2. Samples Poisson(`λ = rate × healthy_nodes × dt_hours`) for failure count
3. Spends `NODE_RECOVERY_MIN × failures` of stalled time
4. Replaces from a `hot_spare_frac` pool (default 3%); excess failures permanently quarantine the node

**Sanity check**: 4096 GPUs × 8 GPU/node × MTBF 10y → expected ≈ 1 failure / 21 hours. Matches OpenAI/Meta-cited figures for H100 superpods.

### 3.4 `economy.py` — accounting

```python
class CostBook:
    by_phase: dict[str, float]      # 'pretrain', 'sft.compute', 'rlhf.labels', ...
    by_resource: dict[str, float]   # 'gpu_H100', 'human_labels', 'cpu_nodes', ...
    def charge(phase, resource, dollars): ...
    def report() -> str
```

Two-axis accounting lets you see both:
- *what* the money went to (compute vs people vs storage),
- *which phase* consumed it (the eye-opener is when human-label cost dominates at small scale and inverts at frontier scale).

### 3.5 `events.py` — structured event log

Every simulated subsystem emits typed events. Written to `out/sim/<name>/events.jsonl` for post-hoc analysis (loss curves, cost burn-down, failure timeline, etc.).

Event kinds emitted by the simulator:

```
program.start  / program.done
data.start     / data.done
tokenizer.start/ tokenizer.done
pretrain.start / pretrain.log / pretrain.spike / pretrain.done
align.start    / align.sft.done / align.rlhf.done / align.done
reasoning_rl.start / reasoning_rl.done / reasoning_rl.skipped
eval.done
safety.done
serve.tier  (one per tier)
```

### 3.6 `data_sim.py` — corpus pipeline

Models a default 8-domain mix:

| Domain        | Raw TB | Yield | Tok/byte | Weight |
|---------------|-------:|------:|---------:|-------:|
| web_en        | 50,000 | 0.18  | 0.25     | 0.40   |
| web_multi     | 20,000 | 0.16  | 0.27     | 0.10   |
| code          |  3,500 | 0.55  | 0.30     | 0.15   |
| books         |    900 | 0.85  | 0.22     | 0.10   |
| papers        |    600 | 0.80  | 0.23     | 0.08   |
| math          |    120 | 0.70  | 0.30     | 0.07   |
| stackexchange |    150 | 0.75  | 0.27     | 0.05   |
| synthetic     |    300 | 0.95  | 0.30     | 0.05   |

`yield` × `tok/byte` gives effective tokens per raw byte. The simulator computes raw bytes needed, then bottlenecks on **CPU-node throughput** (default 50 MB/s/node × 200 nodes). Charges `cpu_nodes` (×$1.60/h) and `storage_egress` (×$0.01/GB).

### 3.7 `tokenizer_sim.py`

Linear scaling: 100 GB on 96 cores ≈ 12 h. Emits `tokenizer.start/done`.

### 3.8 `pretrain_sim.py`

The most important subsystem.

```python
@dataclass
class PretrainSpec:
    n_params: float
    total_tokens: float
    seq_len: int
    global_batch_tokens: int
    target_mfu: float = 0.50               # achieved/peak FLOPs
    spike_prob_per_1k_steps: float = 0.02  # loss-spike rate
    log_every: int = 100
```

The loop:

```
total_steps         = total_tokens / global_batch_tokens
flops_per_step      = 6 · N · global_batch_tokens
seconds_per_step    = flops_per_step / (cluster.peak_tflops · MFU · 1e12)

for chunk of log_every steps:
    cluster.tick(chunk_seconds, rng)         # may add downtime_seconds
    clock.advance(chunk_seconds + downtime_delta)
    cost.charge('pretrain', f'gpu_{type}', gpu_hours · $/h)
    if rng < spike_prob:  emit pretrain.spike, multiply loss × 1.6
    log loss = step_loss_curve(N, D, step, total_steps)  # noisy Chinchilla
```

The output `losses` list mirrors what a real W&B chart would look like — descending Chinchilla curve + occasional spikes + per-window cost.

### 3.9 `alignment_sim.py`

```python
@dataclass
class AlignmentSpec:
    sft_examples: int = 250_000
    sft_epochs: int = 3
    sft_seq_len: int = 4096
    pref_pairs: int = 200_000
    rlhf: str = 'dpo'                    # 'none' | 'dpo' | 'ppo'
    label_dollar_per_pair: float = 4.0   # human contractor rate
    label_dollar_per_sft: float = 0.50
```

Compute model:

| Phase | Cost                                  |
|-------|---------------------------------------|
| SFT   | `6 · N · examples · seq_len · epochs` FLOPs at 0.45 MFU |
| RM    | (PPO only) ≈ 0.4 × SFT seconds         |
| PPO   | ≈ 4 × SFT seconds (rollouts dominate)   |
| DPO   | ≈ 1.5 × SFT seconds (single forward+backward pair) |

Quality multipliers feed into the eval predictor:

```
sft_quality  = 0.85 + 0.13 · (1 - exp(-sft_examples / 100k))
rlhf_quality = 1.00 if 'none' else
               0.95 + 0.10 · (1 - exp(-pref_pairs / 100k))
```

This gives diminishing returns on label budget — exactly the empirical finding (going from 50k → 500k pairs helps less than 5k → 50k).

### 3.10 `eval_sim.py`

`simulate_eval(...)`:
- Charges 2 GPU-hours on 64 H100s (scaled to actual eval cluster size).
- Emits MMLU / HumanEval / GSM8K / Arena ELO from the predictors, modulated by `sft_quality` and `rlhf_quality`, with ≈1% Gaussian noise.

`simulate_safety(eval_scores, ...)` runs 6 categorical evals:

| Category    | Heuristic                          | Default threshold |
|-------------|------------------------------------|------------------:|
| cbrn        | 0.55 × capability                  | 0.30              |
| cyber       | 0.65 × capability                  | 0.40              |
| persuasion  | 0.70 × capability                  | 0.50              |
| autonomy    | 0.50 × capability                  | 0.35              |
| bias        | ~0.05 (model-independent)          | 0.10              |
| jailbreak   | 0.30 - 0.20 × capability           | 0.20              |

Capability = mean(MMLU, HumanEval, GSM8K). Verdict is `PASS` iff every category is below its threshold; otherwise `BLOCK` with the failed list. **Notice jailbreak gets *better* with capability** while CBRN/cyber/persuasion/autonomy get *worse* — this is the well-known dual-use shape of frontier models.

### 3.11 `serving_sim.py`

```python
@dataclass
class ServingTier:
    name: str
    n_params: float
    quant: str                       # 'bf16' | 'fp8' | 'int4'
    gpu: str = 'H100'
    gpus_per_replica: int = 1
    target_throughput_tok_s: float   # decode throughput per replica
    ttft_ms: int                     # time-to-first-token target
```

Given per-tier QPS, the simulator computes:
- Replicas needed = `qps × (prompt+completion tokens) / target_throughput`
- GPU count, daily $, and **$/Mtok**.

The 24h cost projection in the report is what a finance team would use to model API margins.

### 3.12 `reasoning_rl_sim.py` — RLVR / GRPO post-training (2025 paradigm)

The missing 2025 phase (DeepSeek-R1, o1): large-scale RL against **verifiable**
rewards. Optional — disabled by default, enabled with `--reasoning-rl`.

```python
@dataclass
class ReasoningRLSpec:
    enabled: bool = False
    prompts: int = 100_000           # verifiable math/code/STEM prompts
    group_size: int = 8              # G rollouts per prompt (GRPO group)
    steps: int = 1_000               # optimizer steps
    avg_response_tokens: int = 4_000 # long-CoT rollouts are token-heavy
    prompt_tokens: int = 512
    epochs: int = 1
    mfu: float = 0.35                # rollouts run at lower MFU than pretrain
    verifier_cpu_seconds_per_rollout: float = 0.05
    cpu_dollar_per_hour: float = 1.60
    coldstart_examples: int = 5_000
    label_dollar_per_coldstart: float = 6.0
```

Unlike SFT/DPO, the dominant cost is **generation**, not the gradient update:

```
rollouts        = prompts · group_size · epochs
total_tokens    = rollouts · (prompt_tokens + avg_response_tokens)
gen_flops       = 2 · N · total_tokens                 # forward-only rollouts
update_flops    = 6 · N · (steps · group_size · avg_response_tokens)
rl_flops        = gen_flops + update_flops
```

It charges three resources: `gpu_<type>` (rollouts+updates), `cpu_nodes` (the
sandboxed verifier fleet, overlapped so it adds $ but no wall-clock), and
`human_labels` (the small reasoning-SFT cold-start set). It returns a
`reasoning_quality` (≥1.0) multiplier that `simulate_eval` applies on top of the
pretraining-only scores — lifting GSM8K and arena ELO the way R1 post-training
does. Emits `reasoning_rl.start/done` (or `reasoning_rl.skipped`), including
`rl_vs_pretrain_compute` so you can see how tiny the RL FLOPs are next to
pretraining.

### 3.13 `orchestrator.py`

`ProgramSpec` glues all the above. `run_program(spec)` returns a dict of every subsystem's output and writes:
- `out/sim/<name>/events.jsonl` — every event in order
- `out/sim/<name>/summary.json`  — flat top-line numbers (consumed by the CLI)

---

## 4. CLI: `scripts/simulate.py`

### 4.1 Presets

```python
PRESETS = {
    "1b":   (1.2e9,  1.0e12, seq=4096, batch=1M,   default 64 GPUs),
    "7b":   (6.7e9,  2.0e12, seq=4096, batch=4M,   default 512 GPUs),
    "70b":  (7.0e10, 5.0e12, seq=4096, batch=8M,   default 4096 GPUs),
    "400b": (4.0e11, 1.5e13, seq=4096, batch=16M,  default 16384 GPUs),
    # Frontier-class targets we don't have the hardware to actually run:
    "1t":   (1.0e12, 2.0e13, seq=8192, batch=32M,  default 32768 GPUs),  # 1T-total MoE
    "2t":   (2.0e12, 3.0e13, seq=8192, batch=48M,  default 65536 GPUs),  # "GPT-5.x-class"
}
```

The `1t`/`2t` presets are the **"runs we can't afford"**: pair them with
`--moe-experts` (so only a fraction is active), `--precision fp8`, and a
`--gpu-type GB200`/`B300` fleet to price a credible 2025-class flagship.

### 4.2 Flags

```
--size {1b,7b,70b,400b,1t,2t}         default: 7b
--gpus N                              override default GPU count
--gpu-type {A100,H100,H200,B200,GB200,B300}  default: H100
--rlhf {none,dpo,ppo}                 default: dpo
--sft-examples N                      default: 250000
--pref-pairs N                        default: 200000

  Sparse MoE (frontier sparsity):
--moe-experts N                       default: 0 (dense); e.g. 256
--moe-top-k N                         default: 2; routed experts per token

  Low-precision training:
--precision {bf16,fp8,nvfp4}          default: bf16

  Reasoning RL (RLVR/GRPO, o1/R1-style; off by default):
--reasoning-rl                        enable the phase
--rl-prompts N                        default: 100000
--rl-group-size N                     default: 8
--rl-steps N                          default: 1000
--rl-response-tokens N                default: 4000

  Real-GPU calibration (optional; needs a visible CUDA device + torch):
--real-gpu                            probe local GPUs + time a few real steps
--real-gpu-index N                    which local CUDA device (default: 0)
--use-local-gpu-type                  price the sim with the measured local SKU
--real-steps N                        timed steps in the calibration run (default: 6)

--out-dir PATH                        default: out/sim/<size>
--seed N                              default: 0
```

`--moe-experts` makes the model **sparse**: total params stay at the preset size
but only the active per-token count drives FLOPs, so a 1T MoE trains for the cost
of its ~hundreds-of-billions active params. `--precision fp8` (or `nvfp4`)
multiplies achieved throughput. `--reasoning-rl` appends the RLVR phase after
alignment and lifts the reasoning/ELO scores. `--real-gpu` calibrates the
simulator's per-GPU TFLOP/s from a handful of real timed steps on whatever CUDA
device is present (it falls back to pure simulation if none is visible).

### 4.3 Output

The CLI prints a multi-section report:

1. Header — model (dense or MoE active/total), tokens, precision, cluster, wall-clock
2. **PRETRAIN** — steps, final loss, spikes, GPU failures
3. **ALIGN** — SFT and RLHF quality multipliers
4. **REASON-RL** — (only with `--reasoning-rl`) reasoning_quality, RL FLOPs and % of pretrain, wall-hours, $
5. **EVAL** — MMLU / HumanEval / GSM8K / Arena ELO
6. **SAFETY** — verdict + per-category scores
7. **SERVING** — per-tier replicas / GPUs / $/day / $/Mtok
8. **COST** — TOTAL, by phase, by resource

If `--real-gpu` is passed and a CUDA device is visible, a probe table and a
real-throughput calibration block are appended. Plus two files: `events.jsonl`
and `summary.json` (now also carrying `active_params`, `moe_experts`,
`precision`, and the `reasoning_rl` block).

---

## 5. Calibration & sanity checks

The simulator ships with **13 unit tests** in `tests/test_simulation.py`:

| Test                                              | What it verifies |
|---------------------------------------------------|------------------|
| `test_chinchilla_monotone_in_compute`             | More compute → lower loss, asymptoting to E=1.69 |
| `test_eval_predictors_in_range_and_monotone`      | MMLU stays in [0.25, 0.95]; ELO grows with capability |
| `test_cluster_failures_grow_with_size_and_time`   | 512-node cluster has more failures than 8-node |
| `test_pretrain_smoke_advances_clock_and_costs`    | Clock and cost both advance; step count matches `D/B` |
| `test_alignment_quality_grows_with_data`          | More labels → higher SFT/RLHF quality |
| `test_safety_blocks_high_capability_on_high_thresholds` | At zero thresholds, every model blocks |
| `test_orchestrator_e2e`                           | Full pipeline runs and produces sane outputs |

Plus 6 pre-existing tests covering the rest of the platform skeleton.

---

## 6. Reference results

Reproduce on any machine in <1 s each:

```
$ python scripts/simulate.py --size 1b
$ python scripts/simulate.py --size 7b
$ python scripts/simulate.py --size 70b
$ python scripts/simulate.py --size 400b --gpu-type B200
```

| Size | GPUs       | Sim wall-clock | Total cost  | MMLU  | HumanEval | GSM8K | ELO  | Safety  |
|------|-----------:|---------------:|------------:|------:|----------:|------:|-----:|---------|
| 1B   | 64 × H100  | 3.7 days       | $0.93 M     | 50.6% | 25.2%     | 20.3% | 1515 | BLOCK (jailbreak) |
| 7B   | 512 × H100 | 4.8 days       | $1.02 M     | 62.7% | 41.0%     | 36.1% | 1711 | BLOCK (jailbreak) |
| 70B  | 4096 × H100| 13.2 days      | $3.31 M     | 76.8% | 63.9%     | 63.3% | 1985 | BLOCK (cbrn, autonomy) |
| 400B | 16384 × B200| 24.4 days     | $41.87 M    | 84.2% | 76.8%     | 80.2% | 2142 | BLOCK (cbrn, cyber, persuasion, autonomy) |

### 6.1 Cost-composition inversion

This is the most important observation the simulator surfaces:

```
Phase          1B      7B      70B      400B
─────────────────────────────────────────────
labels      99.1%   91.0%   28.0%      2.2%
compute      0.9%    8.9%   71.9%     97.8%
data+other   0.1%    0.1%    0.1%      0.0%
```

At small scale you are running a **labeling operation that incidentally trains a model**. At frontier scale you are running a **datacenter that incidentally consumes labels**. Both modes need very different org structures — exactly what the platform-overview org chart in `docs/00-overview.md` calls out.

### 6.2 Failure-rate scaling

```
Run         GPU-failures   downtime      $ lost to downtime
1B  / 3.7d        0          0 s              $0
7B  / 4.8d        0          0 s              $0
70B / 13.2d      15        7,200 s         $7,200
400B/ 24.4d      94       45,120 s        $45,120
```

The downtime cost is dwarfed by the cost of *idle* but powered GPUs during recovery. Auto-rewind + hot-spares keep this <0.2% of total spend even at frontier scale.

### 6.3 Safety-threshold crossover

The default thresholds are deliberately calibrated so that **every frontier-scale run blocks** on at least one category. This is realistic — Anthropic's RSP and OpenAI's Preparedness Framework both bake in the assumption that a vanilla frontier checkpoint will not pass safety eval; mitigations (RLHF refusal training, output classifiers, monitoring) are what get a model to ship.

To experiment, raise the thresholds (e.g. set `cbrn_threshold=0.5`) and see what unblocks.

---

## 7. Extending the simulator

To add a new phase (say, **continued pretraining for long context**):

1. Create `platform/sim/longctx_sim.py` with:
   ```python
   def simulate_longctx(spec, cluster, clock, cost, bus, seed) -> dict:
       ...
       clock.advance(seconds)
       cost.charge('longctx', f'gpu_{cluster.gpu_type}', dollars)
       bus.emit('longctx.done', **summary)
       return summary
   ```
2. Add a `LongCtxSpec` dataclass.
3. Add it to `ProgramSpec` and call it from `orchestrator.run_program(...)` after pretrain.
4. Write a unit test that asserts clock + cost both advance.

To add a new metric (say, **TruthfulQA**):

1. Add `predict_truthfulqa(N, D, rlhf_q)` to `scaling.py` (sigmoid over log-FLOPs).
2. Surface it from `simulate_eval`.
3. Add a row to the CLI's report.

To swap the loss law (e.g. for Llama-3-style overtraining curves):

- Edit the `E, A, B, ALPHA, BETA` constants at the top of `scaling.py`. Every downstream prediction updates automatically.

To model new hardware:

- Add a row to `GPU_SPECS` in `cluster.py` with TFLOPs, HBM, and $/GPU-hr.

---

## 8. Limitations (deliberate)

- **No parallelism in the simulator itself.** Phases are strictly serial. In reality data prep and tokenizer training overlap with cluster bring-up. Adding overlap would change wall-clock by <10% and isn't worth the modeling complexity.
- **No memory model.** We assume the user has chosen a parallelism plan that fits; we don't simulate OOM.
- **Eval predictors are regressions, not derivations.** They will mis-predict outliers (a model with a great math curriculum will beat the GSM8K curve; a poorly-tuned model will miss the MMLU curve). They are useful for *relative* comparisons.
- **Reasoning RL is now modeled (optional).** The 2024–2025 RL-on-verifier-reward paradigm (o1, R1) runs as the `reasoning_rl_sim` phase when you pass `--reasoning-rl`; it prices rollout/update GPU compute + verifier CPU + cold-start labels and lifts the reasoning/ELO scores. A toy GRPO loop backs it in `platform/rl/` (see `docs/15-reasoning-rl-rlvr.md`). The *capability* lift is a calibrated regression, not a derivation — see eval-predictor caveat above.
- **Multimodality is still out of scope of the simulator.** A toy VLM adapter exists in `platform/model/vision.py` (see `docs/16-multimodality.md`), but vision/audio encoder pretraining, pixel/audio tokenization, and joint-training costs are not priced by the sim yet.

These are good follow-ups if you want to extend the system.

---

## 9. File index

```
platform/sim/
├── __init__.py
├── clock.py              ~25 LOC   virtual time
├── scaling.py            ~70 LOC   Chinchilla + eval predictors
├── cluster.py            ~85 LOC   GPU fleet + failure model
├── economy.py            ~30 LOC   cost book
├── events.py             ~30 LOC   JSONL event bus
├── data_sim.py           ~60 LOC   corpus pipeline
├── tokenizer_sim.py      ~25 LOC   BPE training
├── pretrain_sim.py       ~85 LOC   the loop
├── alignment_sim.py      ~75 LOC   SFT + RM + DPO/PPO
├── reasoning_rl_sim.py   ~125 LOC  RLVR / GRPO post-training (optional)
├── eval_sim.py           ~85 LOC   eval + safety
├── serving_sim.py        ~50 LOC   tiered inference
└── orchestrator.py       ~90 LOC   end-to-end glue (MoE/precision/reasoning-RL)

platform/model/
└── vision.py            ~110 LOC   toy LLaVA-style VLM adapter (multimodality stub)

scripts/
└── simulate.py          ~210 LOC   CLI

tests/
├── test_simulation.py    ~90 LOC   7 simulation tests
├── test_sim_frontier.py            MoE + FP8 + reasoning-RL + e2e tests
└── test_vision.py                  VLM forward/loss on CPU
```

Total: ~915 LOC of pure Python. Zero external dependencies beyond the standard library.

---

## 10. Quick recipes

**Sweep across model sizes:**

```bash
for s in 1b 7b 70b 400b; do
    python scripts/simulate.py --size $s
done
```

**Compare RLHF strategies:**

```bash
python scripts/simulate.py --size 7b --rlhf none --out-dir out/sim/7b_none
python scripts/simulate.py --size 7b --rlhf dpo  --out-dir out/sim/7b_dpo
python scripts/simulate.py --size 7b --rlhf ppo  --out-dir out/sim/7b_ppo
```

**Find the label-budget knee:**

```bash
for n in 10000 50000 100000 250000 500000 1000000; do
    python scripts/simulate.py --size 7b --pref-pairs $n \
        --out-dir out/sim/7b_pref_$n
done
```

Then `jq '.eval.arena_elo' out/sim/7b_pref_*/summary.json` to see ELO vs label spend.

**Plot a loss curve from `events.jsonl`:**

```python
import json, matplotlib.pyplot as plt
events = [json.loads(l) for l in open("out/sim/70b/events.jsonl")]
losses = [(e["step"], e["loss"]) for e in events if e["kind"] == "pretrain.log"]
xs, ys = zip(*losses)
plt.plot(xs, ys); plt.xlabel("step"); plt.ylabel("loss"); plt.show()
```

**Price a 2025-class frontier run (sparse MoE + FP8 + reasoning RL on GPUs we don't have):**

```bash
# 1T total-param MoE, 256 experts top-8 (~357B active), FP8 on a GB200 fleet,
# with an o1/R1-style RLVR phase bolted on after alignment.
python scripts/simulate.py --size 1t \
    --moe-experts 256 --moe-top-k 8 \
    --precision fp8 \
    --gpu-type GB200 --gpus 32768 \
    --reasoning-rl --rl-prompts 200000 --rl-steps 2000 \
    --out-dir out/sim/frontier_moe_fp8_rl
```

The report's ALIGN section gains a **REASON-RL** line, and EVAL shows the RLVR
lift (in a representative run `reasoning_quality ≈ 1.085` pushes GSM8K to ~99%
and arena ELO past ~2400, on a ~9-day simulated wall-clock). Inspect the RL
economics with:

```bash
jq 'select(.kind=="reasoning_rl.done")' \
    out/sim/frontier_moe_fp8_rl/events.jsonl
# → rl_vs_pretrain_compute shows the RL FLOPs are a small fraction of pretraining
```

**Sweep precision / sparsity to see the throughput-vs-cost tradeoff:**

```bash
for p in bf16 fp8 nvfp4; do
    python scripts/simulate.py --size 70b --precision $p \
        --out-dir out/sim/70b_$p
done
jq '{precision, clock_days, total_dollars}' out/sim/70b_*/summary.json
```

---

*End of document.*
