# October–November SOTA harvest — implementation and evidence

**Validated:** 2026-11-11 · **Scope:** bounded local implementations across all five
projects · **Git state:** working-tree changes, not a release or commit

> **Initial-harvest snapshot:** original experiment and test results are retained
> below. The subsequent [repair pass](2026-11-repair-pass.md) updates correctness,
> provenance, compatibility, and safe-execution contracts. Use that report for
> current acceptance evidence; these historical measurements were not relabeled.

The [October](2026-10-sota-llm-agi.md) and [November](2026-11-sota-llm-agi.md)
recommendations now have executable reference implementations and small CPU
experiments. They do **not** amount to a reproduction of frontier training,
production async RL, general-purpose Lean verification, or LFM3 fine-tuning.
No models or datasets were downloaded, no paid API calls were made, and no GPU
training was launched for this harvest.

## What was implemented

| Harvest | Concrete implementation | Remaining boundary |
|---------|-------------------------|--------------------|
| Complete attempt accounting | [nanoGPT provenance v2](../nanogpt-edu/research/provenance.py), [midGPT search records](../midgpt/research_search.py), [distGPT lease usage](../distgpt/distgpt/eval/rollout_queue.py), [specialization branch ledger](../coder-finetune/eval/specialization.py), [frontier tariff ledger](../frontier-platform/frontier_platform/serving/task_cost.py) | Project-local formats, no shared imports; unknown measurements are not invented. Provider instrumentation and durable billing reconciliation remain external |
| Executable verification | [midGPT local generation](../midgpt/research_model.py) plus strict JSON arithmetic checking; [Lean container adapter](../coder-finetune/eval/lean_jail.py) | Tiny GPT inference was executed. Lean adapter has unit/failure-path coverage but live Lean remains unverified |
| Mixed-size rollout work | [distGPT local workers](../distgpt/distgpt/eval/rollout_worker.py), policy/log-prob version checks, cumulative budgets, atomic snapshots and token-fenced mutation | Real callback execution, but deterministic sequence generator—not a neural rollout engine, multiprocess queue, or async learner |
| Specialization comparison | Actual tiny CPU classifier training: common baseline versus full SFT versus rank-one readout LoRA; held-out and regression tasks; Brier/ECE metrics | Educational classifier, not causal-LM fine-tuning or LFM3. Existing LM training remains a separate, explicitly sized experiment |
| Hybrid architecture ablation | [Causal convolution/local/global mixers](../nanogpt-edu/hybrid.py) and [equal-token experiment CLI](../nanogpt-edu/hybrid_ablation.py) | Tiny analogue, not LFM3 architecture/weights; dense local mask does not establish sparse-kernel memory savings |
| Cost/effort routing and deployment contracts | [Bounded route execution](../frontier-platform/frontier_platform/serving/task_routing.py) with independent checker/authorization, refusal/substitute/partial-stream handling, and shared attempt/token/dollar budgets | Injected synchronous adapters, not provider integrations or a production security boundary; see [design](../frontier-platform/docs/19-october-november-2026-harvest.md) |

## Review corrections

- **Lease identity is mandatory.** Worker identity or an attempt number alone
  cannot authorize completion, failure, or heartbeat. Restoring a pending
  checkpoint may reuse both while creating a new acquisition. Every mutation
  needs its unique lease token; duplicate completion with that token is idempotent.
  This is an intentional compatibility tightening, documented in distGPT's README.
- **Unknown usage is not a cheap failure.** midGPT withholds measured token cost
  per accepted task if any branch has unknown usage. Frontier routing retains
  admission reservations for unknown bills and blocks further dispatch until
  reconciliation. It never prices a substitute model using the requested
  model's tariff as though that were measured billing.
- **Budgets stay shared.** Frontier token, dollar, and attempt limits apply to
  the same ledger across tasks, not just one retry loop. Callbacks still need
  their own enforced runtime/output limits; a Python contract cannot preempt a
  hung synchronous function or undo provider overspending.
- **Failed experiments stay visible.** A failed hybrid arm remains in the JSON
  with observed elapsed time and unknown completed-token count. An arm's mean
  quality is unavailable if any requested seed failed. The specialization
  harness retains failed branches and all planned prediction denominators.

## Test evidence

Each suite was run from its own subproject directory with its existing `.venv`.
CUDA was hidden and CPU thread counts limited. This is not a new top-level test
runner.

| Project | Passed | Skipped | Notes |
|---------|-------:|--------:|-------|
| nanogpt-edu | 213 | 0 | Causality, gradients, equal budgets, failed-arm retention, v1/v2 accounting |
| midgpt | 151 | 0 | Real tiny-model generation, local checkpoint loading, strict verification, unknown-cost regression |
| distgpt | 172 | 4 | Acquisition fencing, snapshot migration/recovery, policy staleness, worker failure and cumulative costs |
| coder-finetune | 172 | 1 | Container-command/failure tests, split isolation, actual SFT/LoRA smoke; live Lean skipped |
| frontier-platform | 454 | 15 | Routing budgets, tariffs, authorization, refusal/substitution, incomplete responses, cancellation and ledger round trip |
| **Total** | **1,162** | **20** | No failed tests |

Commands use the following form, independently in each directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests
```

The coder-finetune run additionally set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Ruff passed on all 22 changed Python files.
Existing warnings include CUDA-disabled autocast, unfused FlexAttention,
Python 3.14 Torch/JIT/fork deprecations, single-process distributed checkpoints,
and existing scheduler/precision warnings. They are not new model results.

## Bounded experiment evidence

### Hybrid sequence modeling

Run from `nanogpt-edu/`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python hybrid_ablation.py \
  --out research/results/hybrid_cpu_smoke.json
```

The [raw artifact](../nanogpt-edu/research/results/hybrid_cpu_smoke.json) includes
source/data hashes, paired seeds, every arm, timing samples, and tensor payload
counts. Both arms trained on 4,096 tokens for each of seeds 7, 17, and 27, with
identical data, optimizer, width, and depth. Arm order alternates. Held-out data
uses an independent RNG stream. Neither arm was tuned against the results.

| Arm | Parameters | Held-out NLL, mean | Sample SD |
|-----|-----------:|------------------:|----------:|
| Global-attention baseline | 31,968 | 3.437754 | 0.010111 |
| Convolution/local/global hybrid | 31,040 | 3.368941 | 0.015052 |

This supports a working synthetic delayed-copy experiment—not a general
language-quality advantage. The comparison is token-matched, not parameter,
FLOP, or wall-time matched. Timing is full-sequence forward latency, not cached
decoding. Memory fields count live tensors, not peak RSS; unmeasured peak values
are `null`. The dense local mask is not an efficient sliding-window kernel.
The protected research harness and default GPT implementation are unchanged.

### Local model-backed candidate checking

Run from `midgpt/`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python research_search.py --tiny-smoke --seed 0 \
  --candidates 2 --max-new-tokens 4 --total-tokens 6 \
  --output out/research-search/harvest-reviewed.json
```

Observed two invalid candidates, six generated tokens, 595 forward-input tokens,
and zero accepted tasks. Accepted-task cost is `null`, not zero. This confirms
actual inference, strict checking, and bounded accounting; it demonstrates no
arithmetic competence. A supplied local midGPT checkpoint can replace the random
model without network access; GPT-2 checkpoints require a local vocabulary file.

### Resumable rollout/checker callbacks

Run from `distgpt/`:

```bash
.venv/bin/python -m distgpt.eval.rollout_worker \
  --snapshot out/harvest-reviewed-queue.json
.venv/bin/python -m distgpt.eval.rollout_worker \
  --snapshot out/harvest-reviewed-queue.json --resume
```

The deterministic short/long jobs completed with two input tokens, 67 output
tokens, and two checker calls. Resume preserved exactly two attempts and the
same usage totals: no duplicate generation or cost. The grader cost is a
synthetic accounting unit, not dollars. The queue is single-process and snapshots
are checkpoints, not a transactional log. Killed work after the last snapshot
cannot be reconstructed exactly.

### Specialization and regression

Run from `coder-finetune/`; the output directory must be new:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 120 \
  .venv/bin/python -m eval.specialization \
  --config configs/specialization_smoke.json \
  --output out/harvest-specialization-reviewed
```

All nine branches completed, with 360 planned predictions and no failures.
The experiment uses three fixed seeds, 60 common base-training steps, and 80
specialization steps per trained branch. Parity is the new task; sign
classification is the retention task. See the [complete method and local-checkpoint
instructions](../coder-finetune/examples/oct_nov_harvest.md).

| Arm | Held-out accuracy | Regression accuracy |
|-----|------------------:|--------------------:|
| Baseline | 48.33% ± 2.89 pp | 100.00% ± 0.00 pp |
| Full SFT | 96.67% ± 5.77 pp | 98.33% ± 2.89 pp |
| Rank-one readout LoRA | 55.00% ± 5.00 pp | 66.67% ± 28.87 pp |

Values are means ± sample SD across three seeds, not confidence intervals.
The constrained LoRA result is a retained regression, not evidence that LoRA is
poor in general. These are character-classifier results, not pretrained LLM
results. Full records, checkpoints, and calibration metrics remain under the
ignored local output directory; rerunning the command regenerates them.

### Routing contracts

Run from `frontier-platform/`:

```bash
.venv/bin/python tools/bench_task_routing.py --out-dir out/task-routing-reviewed
```

Both policies accept three synthetic tasks. Fixed-premium routing uses three
attempts; cheap-then-premium routing uses four, including one rejected cheap
attempt. The fixture tariff totals are 0.00030 versus 0.00016 USD. These are
**synthetic arithmetic outputs, not money spent or measured model savings**.
The artifact includes all attempts; the tests additionally cover unknown bills,
shared limits, authorization failures, refusals, substitutions, and cancellation.

## External blockers and next evidence

- **Lean:** Docker daemon access was denied during worker validation and no
  explicitly trusted local immutable Lean image was supplied. No image was
  pulled and no host fallback exists. The adapter only accepts four fixed
  certificate strings; theorem/premise source and image trust remain host-owned.
  A true and false theorem must be tested against a provisioned image before
  calling this live verification validated. Novelty is never inferred from
  certificate validity.
- **Actual LLM specialization:** choose and provision a licensed local model,
  validate tokenizer/backend compatibility, freeze realistic train/held-out/
  regression datasets, and approve a separate training budget. The tiny
  classifier is evidence for the comparison method only.
- **Production routing and RL:** real provider/rollout adapters, enforced
  cancellation, authenticated model identities, durable concurrent accounting,
  and a learner integration remain external. Native NVFP4, trillion-scale MoE,
  and thousand-agent runs were not attempted.

The working tree is deliberately uncommitted. Earlier user edits to the root
README, documentation index, and September edition were preserved; September
was not changed by this harvest.
