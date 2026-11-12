# November repair and enhancement pass

**Validated:** 2026-11-11 · **State:** uncommitted working-tree changes on base
`d020dd3` · **Scope:** the [SOTA-readiness audit](repository-assessment-2026-11.md),
not a claim of frontier-model capability

The pass repairs the verified correctness and integration problems, adds explicit
support gates, and upgrades reproducibility and real-backend CPU testing. It
preserves the five independent projects. No new top-level build/test runner or
cross-subproject runtime imports were introduced.

**Acceptance:** **1,465 tests passed, 33 skipped in the independent CPU suites**,
plus **six tiny two-GPU FSDP checks passed**. Six CPU skips are those separately
executed GPU checks. Both installable project wheels import outside the checkout;
stdlib `platform` is loaded first. The documented synthetic CPU recipe completes
four updates and publishes a verified checkpoint. Ruff, workflow syntax, direct
version constraints and whitespace checks pass.

No pretrained model or dataset rows were downloaded, no paid inference was used,
and no large training job was launched. Public model/dataset **metadata** was
queried to pin recipe revisions. The only added environment dependency was the
small pinned `pytest-timeout` plugin. Existing user changes were retained, and
the branch is left uncommitted.

## 1. Audit-to-repair status

| Finding | Implemented repair | Evidence / remaining boundary |
|---------|--------------------|-------------------------------|
| **F1 — optimizer history lost on restart** | Normalized DCP state restores fresh AdamW and Muon+AdamW state; completed-update semantics | Moment/step and next-update parity, including actual two-GPU FSDP restart |
| **F2 — integrated topology errors** | DP-coordinate data ownership, topology-aware reductions/control, logical gradient clipping and explicit supported configurations | Dense DDP and restricted TP/SP CPU reference parity; dense DP-only FSDP GPU evidence. PP/mixed DP×TP remain rejected |
| **F3 — generated Python on host** | Coding evaluator/reward entrypoints require an explicit local immutable-image Docker executor; frontier code verifier requires an explicit execution adapter | Command/negative tests and early denial pass. Live Python/Lean containment remains unverified; no unsafe fallback |
| **F4 — DDP ranks share restored RNG** | Collective rank-local RNG/generator/scaler/optimizer/control capture and atomic publication in midGPT | Real two-rank Gloo dropout, next-batch and subsequent-update parity; unsafe legacy/topology resumes rejected |
| **F5 — eval consumes training data / incomplete checkpoint selected** | Separate content-identified validation; non-mutating evaluation; staged, hashed complete-checkpoint publication; collective discovery and pinned-best retention | Eval cadence/update identity, failure/publication tests, complete-state restart. Shared-filesystem synchronous scope, not an object-store service |
| **F6 — rounded budgets and omitted failed costs** | Exact preflight token budgets, whole-document splits, measured byte normalization, separate final-holdout role, schema-3 nullable costs and cleanup | Actual tiny CPU runs and negative tests. Repeated-seed promotion, near-dedup and statistically rigorous final quality evaluation remain future work |
| **F7 — sampler/trajectory/budget mismatch** | Correct top-p support, explicit sampling semantics, exact reward/learner token alignment, observed behavior probabilities, frozen actor snapshots, complete attempted-work accounting | Reference/negative tests pass. vLLM is not live-validated; callbacks are not a production concurrency or cancellation service |
| **F8 — hidden dependency/configuration drift** | Strict installs, per-project profiles, runtime metadata, real HF imports/steps, wheel checks, unsupported mandatory settings rejected | Installed direct pins match. Python 3.11/3.14 CI matrix is configured but was not run remotely; local tests used 3.14.4 |
| **F9 — supervision, benchmark and artifact identity** | Explicit SFT loss scope and validation, grouped split/provenance, pinned references, honest benchmark names, safe reward outcomes, no-clobber merge/export receipts | Real tiny SFT/DPO/GRPO steps and nonzero LoRA merge/reload parity. No Qwen quality benchmark, EvalPlus integration result, or quantized-serving claim |
| **F10 — stdlib collision / overstated blueprint evidence** | Package renamed to `frontier_platform`, import hacks removed, restricted legacy conversion, capability/evidence matrix | Source and wheel import-order tests and conversion tests pass. Simulator capability/precision coefficients remain assumptions unless separately calibrated |

## 2. Important behavioral and compatibility changes

### Checkpoint and distributed training

- distGPT checkpoints denote **completed updates**. A controlled stop uses
  `train.stop_after`; changing `optim.total_steps` is not an exact resume because
  it changes the schedule.
- Full restart restores model/optimizer, rank RNG, loader and controller state
  and validates data/config/topology before mutation. Tensor resharding or a
  weights-only warm start is not equivalent to exact full-job restart.
- `latest()` is local inspection; `discover_latest()` collectively agrees on
  visibility before startup resume and spike rewind. Rank disagreement raises
  before DCP branches diverge.
- Unmarked/incomplete or unsafe legacy state cannot silently become an exact
  resume or a new training run. Reviewed weights-only migration needs a new run
  and explicit settings.
- Held-out validation must have distinct paths/content. Set `eval_every: 0` to
  disable it explicitly; there is no training-stream fallback.
- The current trainer permits dense CUDA DP/FSDP with AdamW, FP32 master state,
  FP32 or BF16 autocast, no distributed compilation, and no activation-checkpoint
  composition. PP, mixed DP×TP and unvalidated combinations fail early. Ordinary
  `nn.Linear` under an FP8 context is not advertised as FP8 training.

See the [distGPT matrix and runnable recipes](../distgpt/docs/repair-contracts-2026-11.md)
and [midGPT restart contract](../midgpt/docs/repair-contracts-2026-11.md).

### Research experiments

The protected nanoGPT harness was revised as an explicitly authorized maintainer
change. Candidate-search agents still may not edit it. Token budgets must divide
into complete candidate updates; no rounding up/down is hidden. The loop retains
whole-process elapsed time, rejects malformed or final-holdout selection results,
cleans temporary files, and records unavailable work as unknown.

Provenance schema 3 reads legacy v1/v2 but permits nullable training-token and
verifier costs. Cost-per-accepted-result is unavailable when its inputs are
unmeasured. Document splitting precedes token concatenation; exact repeats stay
in one split; artifacts/source/tokenizer identities and UTF-8 normalization are
recorded. This is not a near-duplicate filter, a malicious-candidate filesystem
sandbox, or an automatic statistical promotion procedure.

See the [harness/data/provenance contract](../nanogpt-edu/research/REPAIR_CONTRACT.md).

### Coding adaptation and evaluation

- The tested HF families and exact CPU profile now agree. CPU base installation
  does not pull bitsandbytes; QLoRA is a separate optional profile.
- SFT requires an explicit all-token or assistant-only objective and real held-out
  data. Assistant-only loss requires working template generation masks and
  surviving assistant targets. SDPA recipes disable cross-example packing.
- Active recipes pin observed model/tokenizer revisions and Magicoder metadata.
  They default to locally cached weights. Remote preference and MBPP loaders also
  require/forward revisions; empty or non-assertion MBPP task tests are rejected.
- Only sigmoid DPO is an integrated preference trainer in this profile. Full DPO
  receives a frozen copy of supplied weights: a real integration test exposed
  that TRL otherwise attempted an implicit reference reload. ORPO/SimPO are not
  silently mapped to unavailable trainer modes.
- GRPO code rewards require a reviewed local image and working execution backend
  before data/model load. Completions and verdicts are retained; unavailable
  execution is not a zero reward. Deterministic rewards can still be gamed.
- The coding runner is base HumanEval or explicitly named local tasks, not an
  EvalPlus result. Subsets and sample counts are recorded. The old host-execution
  switches and Boolean/message result tuple are not the current API.
- Merge preserves recorded or explicitly supplied base identity, tokenizer/template
  fingerprints and payload hashes, and refuses existing output paths. Export
  emits configuration only; it does not quantize or certify vLLM compatibility.

See the [active coding repair profile](../coder-finetune/examples/repair_profile.md).
Historical speed/quality tables retain their original meaning; they are not new
measurements of changed recipes. Provision reviewed model/data/image artifacts
and use fresh output paths before substantive training/evaluation.

### Frontier package and reference contracts

The Python import is `frontier_platform`. The old `platform` proxy and module
cache eviction were removed; top-level examples and internal imports migrated.
A restricted converter handles common legacy `model`/`ModelConfig` checkpoints
with weights-only loading and an explicit class-name alias, without restoring a
stdlib-shadowing package or unrestricted pickle fallback. Source files are
preserved and publication refuses overwrite.

Torch sampling retains the top-p threshold-crossing token and reports its actual
sampling distribution. GRPO requires positive temperature and untruncated
softmax (`top_p=1`) in this reference; unsupported distributions are rejected.
Actor snapshots/provenance and response masks refer to the exact rewarded tokens;
missing behavior log-probabilities are errors, never fabricated zeros. Dropout
cannot silently change the learner distribution being compared to behavior.

Reserved multi-agent workers are admitted before dispatch. Legacy callbacks
cannot promise hard pre-dispatch limits, but overruns/errors/scorer failures and
unknown costs remain visible and stop additional work. Code-verifier/evaluator
construction requires an explicit execution adapter; only named trusted-fixture
opt-in permits the old subprocess helper. Those helpers are not a security jail.

See the [serving/RL contracts](../frontier-platform/docs/20-sota-repair-contracts.md)
and [capability/migration matrix](../frontier-platform/docs/21-capability-evidence.md).

## 3. Acceptance evidence

### Independent CPU suites

From each project directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest -q -ra --tb=short --timeout=180 tests
```

| Project | Passed | Skipped |
|---------|-------:|--------:|
| nanogpt-edu | 236 | 0 |
| midgpt | 162 | 0 |
| distgpt | 189 | 10 |
| coder-finetune | 318 | 8 |
| frontier-platform | 560 | 15 |
| **CPU total** | **1,465** | **33** |

Skips are explicit: GPU/NCCL paths, optional Liger/warcio, absent jailer binaries,
a gated distributed experiment, and live Python/Lean images. Six skipped FSDP
GPU cases were executed separately and passed. Required real HF integration does
not depend on silently missing libraries.

The local environment was Python **3.14.4**, Torch **2.12.0**, NumPy **2.4.6**,
Transformers **5.8.0**, PEFT **0.19.1**, TRL **1.3.0**, Datasets **4.8.5**,
Accelerate **1.13.0**, pytest **9.0.3**, and pytest-timeout **2.4.0**. Existing
warnings include unfused FlexAttention, deliberately disabled CUDA paths,
Python 3.14 Torch/JIT deprecations, single-process DCP assumptions and a scheduler
unit test. No warning is presented as model-performance evidence.

### Real tiny FSDP checks

Two local **RTX 5060 Ti** GPUs, each reporting **16,311 MiB** total memory, were
checked idle before the runs. NVIDIA driver: **595.91.07**. The first unwrapped
attempt failed in NCCL with an NVML driver/library mismatch, before parity
assertions. Using the **existing** user-local runtime wrapper resolved that
process-level library selection; no system reconfiguration was performed.

Final command from `distgpt/` on this host:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  timeout --kill-after=10s 180 /home/support/utils/nvidia-live.sh \
  "$PWD/.venv/bin/python" -m pytest -q --tb=short --timeout=90 \
  tests/test_fsdp_cuda.py \
  tests/test_repair_acceptance.py::test_real_fsdp_trainer_interrupted_restart
```

**Six passed.** Four cover unsharded-reference gradient/clipping comparisons,
fresh Adam moments and the next update in FP32/BF16-autocast modes with both
reshard settings. Two run the actual trainer uninterrupted versus interrupted/
resumed and compare weights, optimizer, RNG, loader and runtime state. Models are
below one million parameters with only a few updates. This is correctness
evidence—not multi-node scaling, 70B support evidence, or a throughput benchmark.

### Actual HF and artifact integrations

`tests/test_real_sft_repair.py` runs actual offline CPU SFT, DPO and GRPO optimizer
steps on a locally constructed tiny GPT-2 and tokenizer. SFT checks both masking
policies, parameter updates and held-out loss. DPO checks a frozen, identical,
non-aliased reference. GRPO uses a clearly labeled **non-executing reward double**:
it tests the trainer/generator API, not Python execution or reward quality.

`tests/test_export_provenance.py` applies nonzero LoRA deltas and verifies
save/merge/reload logit parity, tokenizer/template identity, hashes, stored/pinned
revisions and no-clobber publication. Recipe tests construct every shipped YAML
configuration under the real installed APIs without downloading weights/data.

### Installation, smoke commands and CI

- Both `distgpt` and `frontier-platform` wheels were built without network build
  dependencies, installed temporarily, and imported outside the checkout with
  stdlib `platform` first. The package itself came from the wheel, not an editable
  path. Existing validated dependencies were reused; this was not a fresh
  transitive dependency reinstall.
- The documented `repair_cpu.yaml` CLI recipe completed four synthetic updates;
  the real checkpoint manager verified step 4 as committed. Inputs/outputs were
  temporary, not overwrites of user data.
- Installed direct dependencies match every project's constraints. CI now fails
  on installs, runs real-stack checks, records environments/skip reasons, and
  checks wheels. Workflow YAML and embedded Bash/Python syntax were validated.
- **The GitHub-hosted Python 3.11/3.14 matrix was not executed in this session.**
  Direct constraints are not a complete transitive/hash lockfile.
- Ruff and `git diff --check` pass. Local test/recipe logs are under
  `.jaaicode/artifacts/repair-validation/`; portable counts/contracts are in the
  [machine-readable receipt](research/sota-repair-evidence-2026-11.json).

## 4. Remaining gates and next meaningful enhancements

1. **Live isolation:** Docker daemon access was denied and no operator-approved
   Python/Lean image was supplied. No image was pulled. Run real filesystem,
   network, descendant, output, timeout, cleanup and grader-tampering tests before
   untrusted deployment. Default denial is implemented; containment is not certified.
2. **Larger distributed recipes:** PP, mixed DP×TP, context/expert parallelism,
   distributed compilation and additional precision/optimizer compositions need
   their own reference-update, restart, fault and scaling evidence.
3. **Real capability per budget:** provision a licensed local pretrained model,
   freeze realistic train/validation/regression/final-test data, and run
   same-base/budget comparisons. No Qwen, LFM3 or other pretrained-model quality
   improvement is claimed by this repair.
4. **Production serving/RL:** real vLLM parity and streaming, authenticated
   identities, durable concurrent reservations, enforced cancellation, isolated
   workers and independently validated reward services remain integrations.
5. **Scientific evidence:** adaptive search still needs independent repeated-seed
   promotion and final evaluation; data needs near-dedup/contamination review;
   simulator capability/precision/economic outputs need calibration and uncertainty.

The code is a stronger, tested foundation for those milestones—not a declaration
that all five projects have reached frontier performance or production maturity.
