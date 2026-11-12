# SOTA-readiness assessment and improvement roadmap — 2026-11

**Assessed:** 2026-11-11 · **Scope:** current working tree of all five projects,
including the uncommitted October–November harvest · **Mode:** code/test audit;
no runtime implementation changes

> **Subsequent repair:** this is the pre-repair audit snapshot. The
> [repair pass and acceptance evidence](2026-11-repair-pass.md) maps every finding
> to its current status, supported configurations, and remaining external gates.
> Historical line references and probe outputs below describe the recorded audit
> fingerprint, not a claim that every defect remains open.

## Executive judgment

**The repository already has many modern ingredients. Its largest remaining gap
is not another architecture feature: it is proving that data, optimization,
restart, evaluation, and serving work correctly together.** The five projects can
become excellent, current reference projects at their respective scales. They
cannot be described as frontier-capability models merely by implementing frontier
techniques or passing unit tests.

The highest-value sequence is:

1. **Fix correctness and execution boundaries:** fresh-optimizer checkpoint
   restoration, model-parallel data/control flow, per-rank resume, and real
   isolation for generated Python.
2. **Make installations and experiments reproducible:** tested dependency
   profiles, strict CI, immutable data/model/tokenizer manifests, and explicit
   supported configuration matrices.
3. **Establish capability per budget:** canonical held-out evaluation, regression
   suites, independent final tests, repeated runs, and full cost accounting.
4. **Optimize only validated paths:** cached decoding, packing/fused losses,
   sharded vocabulary loss, low precision, and asynchronous RL, each with a
   numerical reference and an end-to-end acceptance gate.

**Important finding:** all five fresh CPU suites pass—**1,162 passed, 20 skipped**—
yet bounded probes expose material gaps. In particular, distGPT reloads weights
but loses Adam moments when restoring into a fresh optimizer; the existing
model-parallel trainer distributes different data to TP peers; and an older
frontier multi-agent helper executes work it omits from its returned cost. These
are not speculative concerns or reasons to abandon the architecture. They are
specific missing invariants to test and repair.

This report builds on, but does not blindly repeat, the
[June assessment](repository-assessment-2026-06.md). The recent
[harvest](2026-11-harvest.md) is useful: it adds better local contracts and honest
small experiments. It does **not** close older trainer, evaluator, or serving
paths automatically.

## 1. What “SOTA level” should mean here

| Project | Appropriate target | What is not the target |
|---------|--------------------|------------------------|
| `nanogpt-edu` | A readable, numerically correct modern Transformer lab with reproducible small-budget ablations | A miniature copy of every frontier system or a claim that Shakespeare loss measures general capability |
| `midgpt` | A reproducible single-node pretraining stack competitive on quality per token, FLOP, and wall time at 124M–1.5B | A collection of toggles without rank-correct restart, common evaluations, or measured interactions |
| `distgpt` | A topology-correct distributed reference with explicit supported recipes, parity tests, and measured scaling | Assuming that independently working TP, PP, and FSDP components compose correctly |
| `coder-finetune` | An auditable, safe adaptation/evaluation stack that demonstrates real coding gains at a fixed budget | Treating a toy classifier, training loss, or a few HumanEval examples as proof of LLM specialization quality |
| `frontier-platform` | A truthful executable blueprint: typed contracts, validated reference paths, and explicit production replacement boundaries | A production frontier service inferred from interfaces, simulations, or mocked backend tests |

Use five readiness dimensions: **correctness, reproducibility, capability,
efficiency, and operational safety**. A technique earns a supported status only
for the configurations and hardware actually exercised. Maintain distinct labels:

- **Implemented:** code exists.
- **Contract-tested:** unit and negative tests exercise the interface.
- **Integrated:** a real backend runs an end-to-end path.
- **Measured:** quality/cost/latency results have a reproducible manifest.
- **Production-validated:** recovery, concurrency, authorization, security, and
  operational limits have been tested in the intended environment.

Do not collapse these into one green checkbox. Do not rank the repository against
commercial models without running comparable evaluations.

## 2. Method, baseline, and limitations

### Working-tree identity

The starting Git commit was `d020dd366770379bdf04cd2ac300c915cee4c25b`, but the
audit includes the existing dirty working tree. Git SHA alone therefore does not
identify the audited code. A sorted path/content fingerprint of **372 tracked or
non-ignored runtime/config files** (`.py`, `.toml`, `.yml`, `.yaml`, and
`requirements.txt`) was recorded:

```text
7a8f6cba362bd2b187809f2d184c9231c3697c0d4b4347e0bf9c98166dc985a8
```

Source line references below describe this snapshot. No runtime files were
changed to fix the findings. Temporary synthetic data and trusted tiny-model
probes were used; no model/dataset download, paid inference, untrusted generated
Python, or GPU/multi-node workload was launched.

The knowledge graph was used for orientation, but its overlay was partial.
Findings rely on source inspection and the direct probes recorded below, not on
an exhaustive call-graph claim. Partial delegated review notes were not treated
as completed evidence; the important findings were checked directly.

### Fresh tests

Each command ran from its own project directory with its existing environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests
```

`coder-finetune` also used `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

| Project | Passed | Skipped | What this does not prove |
|---------|-------:|--------:|--------------------------|
| nanogpt-edu | 213 | 0 | Real-corpus superiority or validity of adaptive model selection |
| midgpt | 151 | 0 | Multi-rank exact resume or canonical benchmark parity |
| distgpt | 172 | 4 | Mixed DP×TP×PP optimizer parity, multi-node scaling, or crash-durable full-state restart |
| coder-finetune | 172 | 1 | Live Lean, secure Python isolation, EvalPlus integration, or a working declared-version stack |
| frontier-platform | 454 | 15 | Real vLLM weight synchronization, production concurrency, or validated frontier capability predictions |
| **Total** | **1,162** | **20** | Passing tests cover their assertions, not every advertised configuration |

The environments reported Python 3.14.4, Torch 2.12.0, Transformers 5.8.0,
PEFT 0.19.1, TRL 1.3.0, Datasets 4.8.5, Accelerate 1.13.0, and pytest 9.0.3.
Neither `lm_eval` nor `evalplus` distribution metadata was installed. This is an
observed environment, **not** a recommended lockfile. Existing warnings include
unfused FlexAttention, disabled CUDA autocast, distributed checkpoint assumptions,
Python 3.14 JIT/fork deprecations, and scheduler/precision warnings.

### Primary reference stacks

The comparison is against concrete engineering practices, not star counts or
vendor benchmark charts. Upstream revisions are pinned in §10.

- **nanochat [R1]:** tokenizer-to-chat workflow, a fixed capability target,
  explicit validation/CORE metrics, and model-size sweeps. Borrow its experiment
  discipline, not its advertised dollar costs as predictions for this hardware.
- **torchtitan [R2, R3]:** composable parallelism, verified recipes, loss comparison,
  checkpointable data, observability, and trainer/generator numerical parity.
  Its latest features have dependency/hardware requirements; it is a reference,
  not a drop-in promise of compatibility.
- **OLMo-core [R4] and DataTrove [R5]:** clear model/training building blocks and
  scalable, resumable data processing, deduplication, and decontamination.
- **TRL [R6], lm-eval [R7], and EvalPlus [R8]:** explicit training/evaluation
  contracts and reusable tools. Pin and test actual adapters rather than mimic
  their APIs with only mocks.
- **vLLM [R9]:** maintained serving, with documented reproducibility limits.
  A seed alone is not a cross-version or cross-hardware reproducibility guarantee.

## 3. Priority findings

**P0** means fix before relying on the affected scale/safety path. It does not
mean every trusted CPU example is unusable. **P1** means required for credible
research/release evidence. **P2** means optimization after the relevant invariants
are established.

### F1 — P0: a fresh distGPT optimizer loses checkpointed moments

**Evidence:** [checkpoint.py](../distgpt/distgpt/training/checkpoint.py), lines
75–105, builds the load destination from a newly created optimizer's
`state_dict()`, then calls DCP and reloads that state. The fresh Adam state
mapping is empty.

A direct one-step `torch.nn.Linear`/AdamW round trip through the real
`CheckpointManager` produced:

```text
ADAM_SAVED_STATE_ENTRIES 2 FRESH_RESTORED_ENTRIES 0 NEXT_STEP 1
MODEL_WEIGHTS_EQUAL True
```

**Impact:** a resumed job can look successful while starting its optimizer
history over. Correct weights and a resumed step number do not establish training
continuity.

**Improvement:** use a tested DCP model/optimizer state abstraction that correctly
materializes fresh optimizer state, including sharded state and multiple
optimizers. Define checkpoint-schema migration explicitly.

**Acceptance:** save after nontrivial Adam updates, reconstruct model **and
optimizer from scratch**, reload, compare every moment and step counter, and
compare the next update against uninterrupted training. Repeat for Muon+AdamW,
DDP/FSDP, and supported topology changes. Testing reload into an already-used
optimizer is insufficient.

### F2 — P0: distGPT's integrated model-parallel data/control plane is not correct

The June concern remains open:

- [trainer.py](../distgpt/distgpt/training/trainer.py), lines 173–177, passes global
  `rank` and `world` to the loader; [streaming.py](../distgpt/distgpt/data/streaming.py),
  lines 39–50, selects `files[rank::world_size]`.
- For a DP=1, TP=2 layout, TP peers should collaborate on the **same batch**.
  A two-shard loader probe returned zeros to rank 0 and ones to rank 1.
- [trainer.py](../distgpt/distgpt/training/trainer.py), lines 304–315, describes a
  DP-only loss reduction, but [utils/dist.py](../distgpt/distgpt/utils/dist.py),
  lines 42–47, reduces over the world group. Pipeline intermediates contribute
  zero loss; tail-only rewind decisions can diverge from other stages.
- The PP branch, trainer lines 279–287, fetches one `micro_batch`-sized loader
  batch while the schedule splits it into `grad_accum` microbatches. That is not
  the documented concatenated macro-batch of `micro_batch × grad_accum` examples.
- PP evaluation uses a PP-local integer as `dist.broadcast(src=...)` at trainer
  lines 423–429. `src` is a global rank for that API; nontrivial subgroup layouts
  need an explicit coordinate-to-global-rank mapping.

The loader probe is a **wiring counterexample**, not a completed multi-node
experiment. The control-flow conclusions are source-backed risks that require
mixed-layout end-to-end tests.

**Improvement:** one explicit mesh-coordinate abstraction; batch ownership by DP
coordinate; identical input/target identities within model-parallel groups;
correct stage-boundary payloads; DP-only statistics; and one collective
rewind/checkpoint decision. Validate actual macro-batch accounting.

**Acceptance:** for every supported layout, compare logits, main loss,
all parameter gradients, and one optimizer update against a single-process
reference using nonuniform data. Include DP2, TP2, TP2+SP, PP2, DP2×TP2, and a
mixed PP layout as hardware permits. Keep unsupported combinations rejected
until tested. The current [SP test](../distgpt/tests/test_sequence_parallel.py),
lines 130–164, uses all-zero inputs and compares losses **between ranks**; it
does not establish reference-gradient or optimizer-update parity.

### F3 — P0 for untrusted candidates: generated Python has no enforced isolation boundary

[coder-finetune/eval/run_humaneval.py](../coder-finetune/eval/run_humaneval.py),
lines 7–27 and 108–211, explicitly describes a subprocess safety floor. It runs
`exec(prog, ns)` in a child, applies best-effort rlimits, and defaults to `fork`
on POSIX. `spawn` avoids inheriting some parent state but is **not** filesystem,
network, or credential isolation. No hostile code was executed during this audit.

The new [Lean jail](../coder-finetune/eval/lean_jail.py) is a separate, constrained
adapter. It neither replaces this Python executor nor validates arbitrary Lean.

**Improvement:** a fail-closed evaluator service with immutable local images,
non-root execution, denied networking, no host secrets or repository mounts,
resource/output ceilings, whole-job cleanup, and a stronger runtime or dedicated
host where the threat model warrants it. Training code should send bounded jobs
and receive verdicts—not run model-generated Python in the trainer's trust domain.

**Acceptance:** controlled fixtures prove that candidates cannot read host
sentinels, access network endpoints, retain descendants, escape wall/memory/output
limits, or use inherited credentials. Run these on the actual deployment backend;
mock command checks alone do not validate containment.

### F4 — P1: midGPT DDP restart restores rank 0's randomness everywhere

[midgpt/train.py](../midgpt/train.py), lines 112–114, initializes rank-distinct
sampling generators. Lines 179–204 reload one `rng_state`, `cuda_rng_state`, and
`gen_state`. Lines 225–234 serialize one state set; lines 322–332 and 360–362
save only on the master rank.

A deterministic mechanism probe confirmed that generators with different
rank seeds emit identical next draws after receiving the same saved state. This
is not a fresh full DDP training run, but it directly exercises the state
restoration operation used by the trainer.

**Improvement:** save and restore RNG/sampler/scaler state for each rank, with
explicit world-size/topology compatibility and collective publication. Preserve
best-metric and rewind-controller state where required by the restart contract.

**Acceptance:** compare an uninterrupted multi-rank run with a mid-accumulation
or completed-step restart at a clearly specified boundary. Check next-batch IDs,
rank-distinct streams, dropout RNG, optimizer/scaler states, and subsequent
updates—not only the loss printed after load.

### F5 — P1: validation and checkpoint publication need stronger semantics

**distGPT validation:** [trainer.py](../distgpt/distgpt/training/trainer.py),
lines 345–347 and 387–405, passes the training loader into `_eval_one_batch`.
A real loader probe changed its cursor from 0 to 5. This is a training-stream
probe, not a separately defined held-out corpus; changing evaluation frequency
also changes subsequent training data consumption.

**Checkpoint publication:** [checkpoint.py](../distgpt/distgpt/training/checkpoint.py),
lines 41–73 and 146–166, creates the final step directory before all writes and
selects the largest step-named directory without a completion marker. An empty
`step_000000999` directory was returned as the latest checkpoint. GC also needs
validated names, pinned-best protection, and explicit retention semantics.
Per-rank cursor files are useful, but they are not full RNG/controller state;
loader topology changes are explicitly rejected in `streaming.py:86–95`.

**Improvement:** an immutable held-out manifest and separate validation loader;
non-mutating evaluation; checkpoint staging plus collective commit marker;
manifest/hash verification; and clearly separate **weights resharding** from
**full training-job restart across topology changes**.

**Acceptance:** eval frequency cannot change training batch IDs; validation
never enters training; interrupt each checkpoint-publication stage and ensure
`latest()` selects only a complete checkpoint; restart tests cover all promised
state. Do not advertise async publication when the path calls synchronous
`dcp.save` without that implementation.

### F6 — P1: old research-budget and selection paths remain scientifically fragile

[nanogpt-edu/research/harness.py](../nanogpt-edu/research/harness.py), around
lines 174–181, uses `max(1, token_budget // tokens_per_step)`. Tiny CPU calls to
the actual function measured **1 requested token → 8 trained**, and **10 → 8**.
Candidate batch/block changes can therefore change actual budgets despite the
“exact token budget” description. The new hybrid ablation rejects indivisible
budgets correctly; the protected older harness does not.

[research/loop.py](../nanogpt-edu/research/loop.py), lines 192–209, repeatedly
selects on the same validation score with a `1e-6` improvement threshold. This
is adaptive validation use, not independent evidence that a selected change
improves generalization. Its crash path at lines 107–111 does not record elapsed
work, while lines 227–231 default absent costs to zero. A richer provenance schema
does not automatically instrument that caller.

The web-text preparer splits at a token offset
([prepare_fineweb.py](../nanogpt-edu/prepare_fineweb.py), lines 63–90), which may
split a document and does not provide repository/document-family isolation.
The harness's bits-per-byte conversion at lines 233–238 accepts a caller-supplied
bytes-per-token factor: the default of 1 is not a valid general BPE conversion.

**Improvement:** preserve the small educational loop, but add exact-budget
validation, explicit failed/unknown costs, document-level split manifests,
separate search-validation and sealed final-test data, repeat-seed promotion,
and measured byte normalization. Since the harness is intentionally protected,
any future change to it needs an explicit reviewed harness-contract revision;
this audit does not edit it.

**Acceptance:** non-divisible/undersized budgets fail before training or have a
predeclared partial-step policy; every failed attempt retains measured elapsed
cost and marks unavailable work unknown; final-test data never drives candidate
selection; reported bits/byte reconcile with measured bytes and tokenizer rules.

### F7 — P1: frontier serving and RL need one consistent numerical/data contract

Three safe CPU counterexamples are especially actionable:

1. **Nucleus sampling:** [torch_engine.py](../frontier-platform/frontier_platform/serving/torch_engine.py),
   lines 47–56, masks the first token that crosses the cumulative-probability
   threshold. For probabilities `[0.60, 0.25, 0.15]` and `top_p=0.7`, the minimal
   nucleus is tokens 0 and 1, but the implementation retained only token 0 and
   returned log-probability 0. midGPT's implementation already shifts the mask
   correctly; keep projects independent but share the mathematical contract/tests.
2. **Reward/trajectory mismatch:** [async_rollout.py](../frontier-platform/frontier_platform/rl/async_rollout.py),
   lines 113–130, crops learner token rows to `seq_len` while decoding the entire
   generated continuation for reward. With prompt `[1,2,3,4]`, generated
   `[5,6,7]`, and `seq_len=5`, the learner received only token 5 but reward text
   was `5,6,7`. EOS handling and prompt/output ceilings also need explicit tests.
3. **Post-dispatch accounting:** [multi_agent.py](../frontier-platform/frontier_platform/serving/multi_agent.py),
   lines 42–51, calls the worker before checking its returned token count. A
   20-token callback under a 10-token limit executed once but returned
   `tokens_used=0`. The newer task-routing ledger does not replace this path.

The async actor currently hardcodes the torch backend and shares the learner
model object (`async_rollout.py:52–71`). Its Python async interface is not
proof of independent distributed actors or concurrent GPU execution. The vLLM
adapter replays a completed batch as a token stream and uses a private weight-load
path; tests with fakes are not real-backend validation. Missing token log-probs
fall back to `0.0` in `vllm_engine.py:158–165` and `async_rollout.py:83–86`, which
must not stand in for unknown behavior probabilities in RL.

**Improvement:** a canonical trajectory record containing exact generated/scored
IDs, stop/truncation cause, policy/version, sampling parameters, masks, and
observed log-probs; no fabricated probabilities. Align the sampler and learner
semantics explicitly, including temperature/top-p transformations. Consolidate
old budgeting paths around admission, metering, cancellation, and failed-work
accounting. Real actors need separate weight snapshots and synchronization.

**Acceptance:** sampling-distribution fixtures; exact reward/learner token identity;
EOS/empty/overlong/cancelled-stream cases; retained overrun costs; and
trainer/generator parity at fixed weights. Use the same distribution for an
on-policy ratio≈1 test; when sampling is intentionally transformed, test the
specified importance correction rather than assuming that equality. Follow
[R3, R6, R9] with stated numerical tolerances and performance costs.

### F8 — P1: installation, CI, and compatibility can silently change the experiment

- [CI](../.github/workflows/tests.yml), lines 30–38, swallows both requirements
  and editable-install failures, then adds a broad common dependency set.
- [distGPT metadata](../distgpt/pyproject.toml) and
  [frontier metadata](../frontier-platform/pyproject.toml) declare no runtime
  dependencies, despite the code requiring them for documented paths.
- [coder requirements](../coder-finetune/requirements.txt), lines 4–7, require
  Transformers `<5`, TRL `<0.20`, and Datasets `<4`; observed local tests ran
  5.8.0, 1.3.0, and 4.8.5 respectively. CPU CI uses Python 3.11, while the local
  environments used 3.14.4; minimum-version claims are not covered by that alone.
- [Preference config filtering](../coder-finetune/cf_pref/dpo_train.py), lines
  135–146, silently drops unsupported dataclass fields. The
  [vLLM constructor fallback](../frontier-platform/frontier_platform/serving/vllm_engine.py),
  lines 217–234, can remove requested pipeline parallelism or speculative options.
  Compatibility that silently changes semantics is not reproducibility.

**Improvement:** project-local tested dependency profiles/constraints and package
extras; fail install jobs on errors; build wheels and test imports outside the
checkout; an explicit minimum/current Python matrix; separate CPU, HF integration,
GPU, and multi-node jobs. Version adapters must emit the effective configuration
and reject unsupported mandatory semantics.

**Acceptance:** a clean environment installs each documented profile, executes
at least one real training/evaluation step with the intended libraries, and
records exact versions. Required tests cannot all disappear behind optional
skips. Wrong/unsupported options fail before expensive model loading.

### F9 — P1: benchmark names and training supervision need explicit contracts

The coding runner is **base HumanEval**, not HumanEval+: it loads
`openai_humaneval` at [run_humaneval.py](../coder-finetune/eval/run_humaneval.py),
lines 333–335, and defaults to the first 20 problems at lines 290–300. This is
useful quick feedback but is not an EvalPlus result or broad code capability
measurement. The report writer and recorded completions are useful foundations.

[SFT construction](../coder-finetune/train.py), lines 175–208, specifies packing
but not an explicit assistant-only loss policy and receives no eval dataset.
Training all conversational tokens may be intentional, but it must not be
confused with response-only adaptation. TRL's actual mask behavior depends on
the dataset, template, and version [R6]. The loaders normalize examples to
messages and can discard original metadata
([cf_data](../coder-finetune/cf_data/__init__.py), lines 90–107); model/tokenizer
loads at `train.py:79–106` do not pin a revision.

**Improvement:** task/repository/time-aware splits, contamination checks,
license/source retention, pinned model/tokenizer/template/data revisions,
explicit role-aware loss masks and EOS handling, and validation independent of
checkpoint selection. Add a real EvalPlus adapter, a rolling or private held-out
coding set, and task-level cost/regression reports. For DPO/SimPO/GRPO, add numerical
objective tests and a tiny actual trainer step under each supported dependency
profile; mocked dispatch does not prove the configured objective runs correctly.

**Acceptance:** inspect per-token labels on multi-turn/tool examples; forbid
user/system/padding loss when assistant-only training is requested; test packing
boundaries and truncation; hash split membership; compare base/SFT/LoRA on the
same isolated evaluator and generation budget; validate export/merge logits and
tokenizer identity before declaring a model artifact ready.

### F10 — P1: frontier packaging and simulation claims need a clearer boundary

The top-level import name `platform` collides with Python's stdlib module.
The proxy in [platform/__init__.py](../frontier-platform/frontier_platform/__init__.py)
only helps when the project package was imported first. A direct probe importing
stdlib `platform` before adding the project to `sys.path` produced:

```text
ModuleNotFoundError: No module named 'platform.serving'; 'platform' is not a package
```

Tests work around import order by evicting modules. Rename to a distinctive
package in a separately reviewed migration and test the installed wheel outside
the checkout, including stdlib-first and third-party-import-first cases.

The simulator is valuable for **scenario analysis**, not predicting a real
model's leaderboard position. [sim/scaling.py](../frontier-platform/frontier_platform/sim/scaling.py)
uses rough sigmoid capability fits (lines 47–83), fixed FP8/NVFP4 speedup factors
(lines 108–122), and assumed reasoning-RL lifts (lines 125–142). Add fit provenance,
calibration domains, uncertainty, sensitivity sweeps, and invalid-extrapolation
warnings. Do not use simulated capability gains to claim measured SOTA progress.

The README also calls working code architecture-only and lists some implemented
reference paths as stubs. Replace broad status claims with a generated or
manually maintained **capability × backend × evidence-level** matrix.

## 4. Project-specific improvement plans

### 4.1 `nanogpt-edu`: a modern, trustworthy teaching and research lab

**Keep:** the compact architecture, explicit RoPE/RMSNorm/GQA-style components,
train-only MTP loss, greedy/speculative correctness references, and the new
paired-seed hybrid experiment. Do not transplant a distributed framework here.

**Next milestones**

1. **Scientific correctness:** fix F6 through a reviewed harness revision; add
   document-level data identity and true byte-normalized evaluation.
2. **A stable small-budget baseline:** freeze one character task and one local
   byte/BPE corpus; compare AdamW/Muon and architecture variants under equal
   data/tokens plus separately reported FLOPs/time. Preserve a minimal baseline
   with every optimization disabled.
3. **Ablations with promotion rules:** search on development data, confirm on
   independent seeds, and use a sealed final set once per candidate family.
   Report all attempts and paired uncertainty; three toy seeds are not a general
   model-quality claim.
4. **Inference education:** optionally add a small KV-cache path in a separate
   module with cached/full-logit parity, RoPE-offset and rolling-window tests.
   Then compare cached greedy versus speculative decoding at equal output
   correctness. Avoid implying that MTP guarantees a speedup.

**Graduation artifact:** a short runnable tutorial plus a machine-readable run
manifest that reproduces quality/cost curves and both positive and negative
ablations. Model cards should state data, intended use, limits, and exact
hardware—not advertise general intelligence from a tiny corpus.

### 4.2 `midgpt`: competitive single-node pretraining at a defined budget

**Keep:** working DDP/AMP accumulation, modern/GPT-2 variants, fused-loss hooks,
HF export, and local search/checking. The current tiny random-model search is a
plumbing test, not a capability result.

**Next milestones**

1. **Restart and numerical parity:** F4, exact effective batch definitions,
   scalar versus accumulated gradient tests, and restart/AMP/scaler parity.
2. **Data-first recipes:** immutable source/tokenizer revisions, document-aware
   splits, deterministic shuffle/mixing, contamination checks, and checkpointed
   sampling state. Keep the uint16/GPT-2 storage assumption explicit before
   introducing tokenizers with vocabularies above 65,535.
3. **Canonical evaluation:** pin lm-eval and its task revisions; implement a
   native model adapter or validated export path for each architecture. A GPT-2
   export path is not a generic evaluator for every RoPE/SwiGLU variant. Compare
   against a public same-scale baseline on fixed tasks with logged samples.
4. **Quality-per-budget frontier:** compare validation bits/byte and suitable
   language/code/reasoning task accuracy at equal tokens/FLOPs, then time-to-target
   quality. Validate batch/precision/compile/fused-CE interactions rather than
   claiming the sum of individual speedups.
5. **Serving and adaptation:** cached decoding, batched evaluation, and a small
   midtraining/SFT recipe after the base model and tokenizer have validated
   export/round-trip parity.

**Graduation artifact:** one fully reproducible 124M/350M recipe, one larger
single-node recipe, and a table of supported configurations with measured
quality, throughput, memory, restart behavior, and confidence/variance. Use
nanochat's end-to-end discipline [R1], not an indiscriminate feature port.

### 4.3 `distgpt`: a reliable distributed training laboratory

**Keep:** clear parallelism modules, actual Gloo collectives in tests, DCP
integration, explicit unsupported combinations, and new fenced local workers.
The recent rollout queue improves acquisition semantics; it does not repair the
pretraining trainer's topology or optimizer checkpoint state.

**Next milestones**

1. **Correctness release:** F1/F2/F5, dense baseline parity across supported
   layouts, stage-aware batching, collective control decisions, and numerical
   gradient-norm/clipping validation.
2. **Restart release:** atomic complete checkpoints; per-rank RNG/data/controller
   state; crash injection; best-checkpoint retention; explicit rules for world-size
   changes and interrupted accumulation.
3. **Performance release:** profile compute/communication/data loading separately;
   add vocab-parallel cross-entropy, selective activation checkpointing, asynchronous
   checkpointing, and communication overlap only where traces identify a bottleneck.
   Report warm steady-state throughput and end-to-end time, including save/eval.
4. **Precision release:** the current FP8 module explicitly notes that plain
   `nn.Linear` models do not obtain FP8 throughput
   ([precision.py](../distgpt/distgpt/training/precision.py):12–25). Add real
   supported low-precision layers/kernels, state handling, and BF16 numerical
   baselines before claiming FP8 training. Treat native NVFP4 as a separate
   Blackwell-gated recipe, not a simulator multiplier made real.
5. **MoE/long-context release:** expert-parallel dispatch, grouped expert GEMMs,
   router-load/capacity/drop telemetry, deterministic router-state handling,
   context parallelism, and hybrid-attention cache/state accounting. Each needs
   an isolated reduced-model parity test before a combined configuration.
6. **RL integration:** connect neural generators and CPU verifiers to the queue,
   then add behavior-log-prob parity, policy lag, restartable actors, and a
   transactional backend. Do not label the synchronous callback demo async RL.

**Graduation artifact:** a torchtitan-style verified-recipe matrix [R2], with
single-device loss/update references, multi-rank parity results, save/restart
fault injection, and strong/weak scaling curves. Model size in YAML is not proof
that the recipe has run at that size.

### 4.4 `coder-finetune`: real coding gains with safe and auditable evaluation

**Keep:** full/LoRA/QLoRA paths, placement helpers, explicit reward components,
completion recording, and the new negative-result-preserving comparison method.
Do not treat the character classifier's SFT/LoRA numbers as LLM evidence.

**Next milestones**

1. **Safety and compatibility:** isolate Python evaluation (F3), align installed
   and declared dependencies (F8), and test a real local tiny Transformers/TRL
   model through an optimizer step. Explicitly pin model, tokenizer, template,
   adapter base, and dataset revisions; keep remote-code trust opt-in and reviewed.
2. **Data/supervision:** F9, repository/family/time splits, source/license metadata,
   near-duplicate and benchmark contamination filters, and assistant/tool masking
   tests. Add a validation set that never becomes the final reporting set.
3. **Base versus adaptation:** on a provisioned in-envelope model, compare
   prompting, SFT, LoRA, QLoRA, and only then preference/RL methods. Use identical
   tasks and generation budgets; include syntax, functional correctness,
   instruction/tool schemas, refusal behavior, and retention tasks.
4. **Reward quality before more RL:** hidden tests, verifier false-positive tests,
   reference solutions, task validity audits, and an independent holdout. Run
   objective formula tests, length/normalization checks, and trainer/generator
   parity before enabling a vLLM rollout path.
5. **Deployable artifact:** merged/adapter logit parity, quantization quality
   regression, tokenizer/chat-template identity, license manifest, reproducible
   export, and evaluated inference memory/latency at target concurrency.

**Graduation artifact:** a genuine same-base coding evaluation with all attempts,
proper task denominators, uncertainty, retention results, and cost per accepted
solution. Start with a manageable held-out suite and EvalPlus [R8]; expand to
repository-level or time-separated tasks only as evaluator validity improves.
No fixed gain threshold is promised before the baseline distribution is measured.

### 4.5 `frontier-platform`: an executable and honest systems blueprint

**Keep:** the broad system decomposition, CPU reference modules, design docs,
research-artifact gates, serving abstraction, and explicit source-backed cost
ledger. The goal is to make the relationships executable, not to implement a
commercial platform inside this repository.

**Next milestones**

1. **Truthful capability map and packaging:** F10; distinguish design-only,
   CPU-reference, real-backend-tested, and production-target capabilities. Rename
   the stdlib-colliding package in a reviewed migration, not an incidental cleanup.
2. **One vertical reference path:** versioned dataset → tokenizer → tiny pretrain
   → complete restart → SFT or a small verified RL task → cached serving → held-out
   evaluation → artifact/review gate. Pass the same identities and budgets through
   every stage; do not reset them at adapter boundaries.
3. **Serving/RL contract repair:** F7, request validation, actual finish reasons,
   error/cancellation propagation, no missing log-prob substitution, true backend
   streaming where claimed, and effective-configuration reporting.
4. **Operational adapters:** a durable lease/reservation store, isolated workers,
   policy/weight synchronization, explicit authorization revocation, metrics/traces,
   failure injection, and a supported real serving backend. Implement only alongside
   the corresponding design-doc updates, preserving the project's rule.
5. **Calibrated simulation:** separate measured hardware profiles, assumed
   economics, and hypothetical capability scenarios. Show intervals and sensitivity;
   model extrapolation error rather than plotting an invented score as evidence.
6. **Research and safety validation:** real verifier/literature/reviewer adapters,
   independent graders, held-out red teams, and measured false-positive/negative
   behavior. Certificate validity, novelty, significance, and deployment approval
   remain different decisions.

**Graduation artifact:** a reproducible vertical-slice demo plus a clear replacement
map for production services, with audited failure behavior. A small end-to-end
system with honest limits is a better blueprint than an expansive feature table
whose adapters silently change behavior.

## 5. Cross-project architecture and experiment contracts

Preserve the five independent projects. **Do not add cross-subproject imports or
a new top-level test runner.** Share concepts through versioned specifications,
fixtures, and documentation; keep implementations local where independence is the
teaching goal.

A common **manifest vocabulary**, not a common runtime package, should cover:

- Code commit **and dirty source hashes**, environment/lock profile, hardware,
  precision/kernel settings, seed list, and exact effective config.
- Model/base/adapter/tokenizer/chat-template revisions and licenses.
- Dataset snapshot, document/repository IDs, filters, dedup/contamination report,
  split membership, task/checker versions, and known exclusions.
- Every candidate/branch, prompt/completion IDs, parent/retry lineage, sampler
  parameters, actual usage, unknown-cost flags, and verifier time.
- Checkpoint schema, complete-state manifest, publication status, and supported
  resume topology.
- Quality, regressions, calibration where appropriate, per-domain results,
  paired uncertainty, prefill/decode distributions, peak memory, throughput,
  total wall time, and time/cost to a predeclared accepted-quality target.

Use different denominators honestly: model training tokens, generated tokens,
verifier calls, CPU/GPU time, and dollars are not interchangeable. Cache and
reasoning counters may be subsets of other counters; normalize provider fields
before aggregation. Never substitute zero for unknown usage. A saved checkpoint,
benchmark aggregate, and signed release artifact should carry the same lineage.

## 6. Minimal and ideal hardware envelopes

These are planning envelopes, not measured fit guarantees or one-workstation
limits. Sequence length, batch, optimizer, precision, activations, cache, storage,
and interconnect change capacity requirements substantially.

| Project | Minimal meaningful envelope | Ideal envelope for its next milestones | What ideal unlocks |
|---------|-----------------------------|------------------------------------------|--------------------|
| nanogpt-edu | CPU contracts/tiny probes; one 8–16 GB GPU for small training | One 24–48 GB GPU plus enough CPU/storage for repeated runs | Larger ablations and realistic-corpus sweeps without sacrificing simplicity |
| midgpt | One 16–24 GB GPU with appropriately sized model/context | 4–8 high-memory GPUs in one node | Real DDP restart/throughput validation, longer context, and larger single-node recipes |
| distgpt | CPU/Gloo parity; 2–4 GPUs for real sharding/kernel integration | Multiple 8-GPU high-memory nodes with fast fabric and checkpoint storage | Mixed DP×TP×PP/CP/EP, 1B–70B recipe validation, fault recovery and scaling; size each model separately |
| coder-finetune | One 16–24 GB GPU for bounded-context adapters; isolated CPU evaluator workers | 4–8 × 80 GB-class GPUs plus separate rollout/evaluation capacity | Full fine-tuning, repeated adaptation comparisons, and bounded multi-domain RL |
| frontier-platform | CPU vertical-slice references and design/contract tests | Separately provisioned training, serving, verifier and storage pools | Operational integration/failure tests; 500B+ training remains a separately costed cluster-scale program |

Hopper/Blackwell kernels, native NVFP4, and large MoE remain legitimate ideal-tier
work. They should be gated by numerical and workload evidence, not excluded
because the minimal machine cannot run them.

## 7. Recommended delivery order and acceptance gates

These are dependency stages, not time estimates or permission to launch paid jobs.
Some independent packaging/data work can proceed in parallel.

| Stage | Highest-value work | Exit gate |
|-------|--------------------|-----------|
| **A — correctness and containment** | F1/F2/F3/F4/F5/F7; fail unsupported configurations explicitly | Fresh-optimizer and rank-local restart parity; same-batch model-parallel invariants; isolated code execution; sampler/trajectory/cost counterexamples become regression tests |
| **B — reproducible execution** | Strict installs, tested constraints, effective configs, manifests, package rename plan | Clean-environment real dependency integration; no silent feature degradation; installed imports independent of cwd/order |
| **C — credible quality evidence** | Document/repository splits, sealed holdouts, real coding adaptation, canonical benchmark adapters | Same-base/budget comparisons with all attempts, regressions, uncertainty, and replayable artifacts |
| **D — performance at fixed quality** | Cached decoding, packing/fused CE, vocab-parallel CE, compile/overlap, real FP8 | Numerically validated improvements on representative workloads; distributions and total time/cost, not one favorable microbenchmark |
| **E — larger systems and research** | Native NVFP4, expert/context parallelism, asynchronous RL, advanced scientific agents | Stable reduced-scale parity first; real-backend recovery, security and scaling evidence before larger claims |

**First implementation batch I recommend:** fix fresh-optimizer restoration,
introduce an explicit supported-topology/batch-ownership contract, separate the
distGPT validation loader, and repair the three frontier serving/trajectory/cost
counterexamples. In parallel, make dependency installation fail loudly and
inventory the declared-versus-observed stack. Then address Python isolation and
rank-local DDP restart before running a serious new adaptation or multi-node job.

## 8. What not to prioritize yet

- Adding another MoE, MLA, MTP, or RL objective merely because it appears in a
  release report; many are already present but not validated in combination.
- A large-model download or long run before the evaluator, data split, and
  restart state are trustworthy.
- Generic “FP8 enabled” or “async” labels without kernel traces, actual backend
  integration, and effective-configuration records.
- More synthetic callback demos as substitutes for real adapter integration.
  Keep them as fast contract tests; promote one meaningful vertical path next.
- Replacing all five projects with one shared framework, or letting educational
  code inherit the blueprint's operational complexity.
- Optimizing simulator leaderboard numbers; calibrate assumptions and report
  uncertainty instead.

## 9. Evidence index and June-to-November status

The [machine-readable probe record](research/sota-readiness-audit-2026-11.json)
contains exact observed outputs, environment metadata, test counts, and the
runtime fingerprint. Probes used trusted synthetic inputs and temporary files;
they are not GPU performance measurements or full distributed-training validation.

| June theme | November status |
|------------|-----------------|
| Integrated DP×TP×PP correctness | **Still open**, revalidated by source and data-distribution probe |
| Rank-local midGPT RNG restart | **Still open**, restored-state mechanism reproduced |
| Generated-code isolation | **Still open** for Python; constrained Lean adapter is separate and live integration remains unverified |
| Dependency metadata/CI strictness | **Still open**, with an observed declared/installed HF-stack mismatch |
| Frontier `platform` import collision | **Still open**, reproduced with stdlib-first import |
| Checkpoint publication/full state | **Still open**; fresh-optimizer moment loss additionally reproduced |
| Full experiment provenance | **Partial progress** from the November harvest; older callers still omit some failed-work costs and final holdout controls |
| Architecture-only documentation | **Still inconsistent** with working CPU/reference code and backend adapters |
| Hybrid/verification/routing experiments | **New progress**, with bounded tests and honest toy results; not frontier capability evidence |

The review does not certify security or correctness outside these tests. It
identifies high-leverage improvements and their acceptance criteria. Any future
implementation should preserve existing user work and be followed by the
appropriate project-local tests and evidence updates.

## 10. Primary references

These are repository/documentation sources inspected for this assessment. GitHub
sources are pinned no later than November 11, 2026 UTC. Upstream performance
claims were **not** reproduced locally. Reference numbering is local to this
assessment, separate from the monthly SOTA Watch source sequence.

**[R1] nanochat**, README and linked experiment workflow, revision
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.
<https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md>

**[R2] PyTorch torchtitan**, README: composable parallelism, verified recipes,
testing, data, checkpointing and profiling, revision
`b8cebbfb740c84d2efccacf1ea12ee2e43664be8`.
<https://github.com/pytorch/torchtitan/blob/b8cebbfb740c84d2efccacf1ea12ee2e43664be8/README.md>

**[R3] torchtitan**, *Bitwise Numerics and Batch Invariance*, same revision.
Explicit trainer/generator operation, dtype, parallelism and determinism
requirements, limitations, and performance trade-offs.
<https://github.com/pytorch/torchtitan/blob/b8cebbfb740c84d2efccacf1ea12ee2e43664be8/torchtitan/rl/BITWISE_PARITY.md>

**[R4] AllenAI OLMo-core**, README, revision
`2289b819b8f709e37e3e042ebb019f411fb6b742`.
<https://github.com/allenai/OLMo-core/blob/2289b819b8f709e37e3e042ebb019f411fb6b742/README.md>

**[R5] Hugging Face DataTrove**, README: filtering, deduplication, scalable
pipelines and checkpointed generation, revision
`5484e67ab81daf66c689106054d5133910d731c2`.
<https://github.com/huggingface/datatrove/blob/5484e67ab81daf66c689106054d5133910d731c2/README.md>

**[R6] Hugging Face TRL**, SFT and GRPO trainer documentation, revision
`a76cab378aa69d2ca222aff38e3ec7794279c0b7`. Assistant masks/template requirements,
packing, loss definitions, and training/inference mismatch metrics. These are
reference contracts, not evidence that the local declared TRL range supports
all the current APIs.
<https://github.com/huggingface/trl/blob/a76cab378aa69d2ca222aff38e3ec7794279c0b7/docs/source/sft_trainer.md>
<https://github.com/huggingface/trl/blob/a76cab378aa69d2ca222aff38e3ec7794279c0b7/docs/source/grpo_trainer.md>

**[R7] EleutherAI lm-evaluation-harness**, README, revision
`d6de81643928d653435c431bae19945d41d32520`. Task/model interfaces, separate
backend dependencies, revisions, logged samples, and distributed limitations.
<https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/README.md>

**[R8] EvalPlus**, README, revision
`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`. HumanEval+/MBPP+ and coding-evaluation
methodology. Adopting its benchmark does not remove the need for an enforced
execution sandbox.
<https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/README.md>

**[R9] vLLM**, *Reproducibility*, revision
`3bc0f430883c575d4076280d8d10d77cd4bfc07c`. Default nondeterminism, scheduling/batch
invariance, and same-hardware/same-version limits.
<https://github.com/vllm-project/vllm/blob/3bc0f430883c575d4076280d8d10d77cd4bfc07c/docs/usage/reproducibility.md>
