# Repository architecture assessment and improvement roadmap — 2026-06

> **Scope:** repository-wide architecture, implementation, testing, packaging,
> safety, reproducibility, and documentation review of `LLM-playground`.
>
> **Bottom line:** this is an unusually strong educational LLM-systems
> repository. Its main opportunity is now **consolidation, correctness, and
> reproducibility rather than adding more frontier features**. The first work to
> prioritize is topology-correct combined TP/PP training in `distgpt`, per-rank
> DDP resume state in `midgpt`, an actual isolation boundary for generated-code
> execution, and complete install metadata.

## Executive summary

The repository succeeds because it is not just a collection of transformer
implementations. It presents a deliberate progression from a readable
10M-parameter model to single-node DDP, distributed pretraining, post-training,
and an executable frontier-system reference:

```text
FROM-SCRATCH PRETRAINING
nanogpt-edu
    ↓ add BPE, DDP, accumulation, robust resume, external eval/export
midgpt
    ↓ add FSDP2, TP, PP, DCP, streaming data, cluster launch
 distgpt

POST-TRAINING
pretrained HF model
    ↓ SFT / LoRA / QLoRA
coder-finetune
    ↓ DPO / SimPO / KTO / GRPO-RLVR
code eval and verifier-backed rewards

FULL-SYSTEM VIEW
data → tokenizer → pretrain → alignment/RL → eval → safety → serving
                         frontier-platform
                   executable reference + simulator
```

The strongest aspects are:

1. **A coherent educational ladder.** Each project introduces a distinct class
   of production concern while remaining independently runnable.
2. **Honest measured results.** Negative scaling, regressions, failed
   interventions, memory trade-offs, and bug post-mortems are documented rather
   than hidden.
3. **Tests that encode real bug history.** The suite captures RNG-resume,
   distributed checkpoint, shard-boundary, DDP topology, pipeline construction,
   and sandbox regressions.
4. **Meaningful distributed coverage.** `distgpt` uses real multiprocess gloo
   tests rather than only mocks.
5. **Broad system coverage.** `frontier-platform` represents data, training,
   alignment, evaluation, safety, serving, infrastructure, and economics.

The highest-priority gaps are:

1. **`distgpt`'s full trainer is not topology-correct for combined TP or PP.**
   The loader uses global rank/world size, reductions use the world process
   group, and pipeline-only rewind control is not collective.
2. **`midgpt` DDP resume restores rank 0's RNG and batch-generator state on
   every rank**, which can duplicate batches and dropout streams after restart.
3. **Generated code runs in a resource-limited local subprocess, not a Docker
   sandbox**, despite higher-level wording that can imply otherwise.
4. **Install metadata and quickstarts are inconsistent.** In particular,
   editable installs of `distgpt` and `frontier-platform` do not declare all
   runtime dependencies.
5. **Documentation and project doctrine trail the implementation.** The
   frontier project is no longer merely a set of skeleton interfaces.
6. **The autonomous research harness repeatedly selects on one validation
   split**, creating adaptive validation-overfitting risk.

## Review method and validation

This assessment used:

- The root README and `JAAICODE.md` doctrine.
- All five project READMEs.
- The frontier architecture overview and autonomous-research documentation.
- Representative critical paths in model, data, training, checkpoint,
  distributed, fine-tuning, serving, and simulation modules.
- CI, packaging, dependency manifests, repository hygiene, and recent Git
  history.
- Static architecture, complexity, duplication, type-coverage, and quality
  reports.
- The repository knowledge graph (approximately 6,000 nodes and 13,500 edges).

Every subproject test suite was run using that project's own virtual
environment:

| Project | Result |
|---|---:|
| `nanogpt-edu` | **53 passed** |
| `midgpt` | **87 passed** |
| `distgpt` | **121 passed, 4 skipped** |
| `coder-finetune` | **106 passed** |
| `frontier-platform` | **425 passed, 5 skipped** |
| **Total** | **792 passed, 9 skipped** |

The runs produced roughly 113 warnings, mostly from PyTorch/DCP behavior on
Python 3.14, multithreaded `fork()` use in `coder-finetune`, expected precision
fallbacks, and uncompiled FlexAttention test paths.

### Review limitation

This review did **not** rerun the long GPU experiments or a combined multi-node
DP×TP×PP job. Published benchmark numbers remain documented repository
artifacts rather than independently reproduced measurements in this review.
Findings about combined `distgpt` topologies are based on direct code-path and
test-coverage analysis.

## Architecture by project

### `nanogpt-edu`

```text
TinyShakespeare / addition / FineWeb-Edu
    → uint16 token files + metadata
    → random fixed-length batches
    → RoPE + RMSNorm + SwiGLU GPT
    → AdamW or Muon
    → deterministic held-out evaluation
    → latest and best checkpoints
    → greedy/top-k or MTP-speculative generation
```

**Assessment:** the best starting point for understanding transformer training.
It has grown beyond its original minimal scope through FineWeb preparation,
Muon, MTP, DeepConf, speculative decoding, and the autonomous research harness.
Those additions are valuable, but the core learning path should remain visibly
separate from research extensions.

### `midgpt`

```text
WikiText / OpenWebText / FineWeb-Edu
    → GPT-2 BPE and mmap shards
    → rank-seeded random sampling
    → GPT-2 or configurable Llama-style model
    → DDP-aware gradient accumulation
    → AdamW/Muon + AMP + optional fused CE
    → deterministic cross-rank validation
    → checkpoint/resume and spike rewind
    → HF export → lm-eval / vLLM-compatible directory
```

**Assessment:** the repository's cleanest practical trainer. It strikes the
best balance between readability and production mechanics. Its most important
correctness gap is preserving rank-local RNG and data-generator state across a
DDP resume.

### `distgpt`

```text
Streaming binary shards
    → 3D DeviceMesh (PP × DP × TP)
    → DTensor TP
    → pipeline-stage trimming and 1F1B
    → FSDP2 over DP
    → AdamW or Muon
    → DCP sharded checkpoints
    → spike detection and rewind
    → held-out eval / HF export / lm-eval
```

**Assessment:** the DP/FSDP path is strong, tested, and supported by real
single- and two-GPU measurements. TP and PP primitives also have real
multiprocess tests. The remaining gap is the full trainer's topology semantics
when those primitives are combined.

### `coder-finetune`

```text
HF/builtin instruction or preference data
    → Qwen2.5-Coder base model
    → full FT / LoRA / QLoRA
    → TRL SFT, DPO/SimPO/ORPO, or GRPO
    → optional verifier execution
    → adapter checkpoint
    → merge/export/generate
    → HumanEval-style pass@k
```

**Assessment:** a pragmatic use of HuggingFace, PEFT, and TRL. It correctly
focuses custom code on data shape, adapters, distributed placement, reward
correctness, and evaluation rather than rewriting Trainer. The main weakness is
that generated-code execution still lacks a true isolation boundary.

### `frontier-platform`

```text
data acquisition/filter/dedup/mix/shard
    → tokenizer
    → transformer and training engine
    → SFT/RM/DPO/PPO/GRPO
    → eval and safety gates
    → serving abstraction
    → telemetry feedback

plus:

model/data/cluster specification
    → discrete-event simulator
    → wall time, cost, failures, predicted capability, serving economics
```

**Assessment:** this is no longer accurately described as only architecture
interfaces. It is an executable small-scale reference and simulator with
selected production integrations intentionally left stubbed. The documentation
should make that boundary explicit.

## Maturity summary

| Project | Current maturity | Strongest aspect | Main gap |
|---|---|---|---|
| `nanogpt-edu` | Runnable and well tested | Pedagogical clarity plus real experiments | Core simplicity is being diluted; adaptive validation risk |
| `midgpt` | Strong single-node trainer | Best balance of readability and production mechanics | Multi-rank checkpoint/RNG resume semantics |
| `distgpt` | Strong DP/FSDP framework; partial 3D integration confidence | Real FSDP2, DCP, loader resume, metrics, two-GPU calibration | Full TP/PP trainer topology semantics |
| `coder-finetune` | Practical and broad | Correct HF/TRL/PEFT use plus verifier-oriented testing | Local executor is not a security boundary |
| `frontier-platform` | Large executable reference plus simulator | Broadest system view and subsystem coverage | Identity/status confusion, package collision, uncertainty framing |

## What the repository does especially well

### 1. It preserves an effective complexity ladder

A reader does not need FSDP2 to understand RoPE, and does not need TRL to
understand a pretraining loop. Keeping the projects standalone is the correct
architectural choice.

Static analysis found extensive duplicate blocks, especially Muon and plotting
code. Much of this duplication is intentional and follows the no-cross-project
import invariant. It should be managed with provenance and parity tests rather
than eliminated through a shared runtime package.

### 2. It reports negative findings honestly

Examples include:

- Non-trivial models overfitting TinyShakespeare.
- Muon failing to beat AdamW at an inappropriate data scale.
- Liger saving memory but reducing throughput on Blackwell.
- Naive FSDP2 over PCIe scaling negatively.
- High-rank LoRA needing a larger optimization budget.
- DeepConf separating confidence without reproducing large-vote gains.
- Spike-rewind loops and distributed deadlocks found during real runs.

This is better scientific communication than presenting every implemented
technique as a win.

### 3. Regression tests carry design history

The tests and comments explain the concrete failures behind safeguards:

- RNG `ByteTensor` restore failures.
- Evaluation randomness contaminating training reproducibility.
- Rank-0-only collective checkpoint deadlocks.
- DDP world size incorrectly inferred from visible GPU count.
- LoRA gradient checkpointing losing input gradients.
- Pipeline schedules missing `PipelineStage` wrappers.
- Shard-end sampling bias.
- Queue flakiness when combining threads and fork-based multiprocessing.

### 4. Distributed tests execute real collectives

`distgpt/tests/test_distributed_smoke.py` covers:

- Two-process FSDP2 on gloo.
- TP sharding with forward/backward and rank-consistent loss.
- Pipeline-stage construction.
- A two-process full trainer run using DP/FSDP and DCP.

This is substantially stronger than tests that only inspect a planned mesh.

### 5. Safety limitations are stated in source-level documentation

`coder-finetune` correctly calls its subprocess and rlimit layer a safety floor,
not a jail. The issue is not the local warning; it is higher-level wording that
can imply Docker is provided when it is not.

### 6. The frontier reference covers systems commonly omitted

Data acquisition, synthetic lineage, decontamination, eval contamination, red
teaming, serving, infrastructure, cost, failures, and capability gates all have
explicit module boundaries. This gives readers a much better map of what
surrounds a production training loop.

# Priority findings and recommendations

## P0 — Make `distgpt` topology-correct for the full DP×TP×PP trainer

This is the most important finding.

### Finding 1: data is sharded by global rank instead of DP coordinate

`distgpt/distgpt/training/trainer.py` constructs the streaming loader using the
global rank and global world size:

```python
loader = StreamingLoader(
    data_dir,
    cfg["data"]["seq_len"],
    cfg["train"]["micro_batch"],
    rank=rank,
    world_size=world,
    seed=cfg["seed"],
    device=device,
)
```

`StreamingLoader` then assigns:

```python
self.my_files = self.files[rank::world_size] or self.files
```

That is correct for pure data parallelism. It is not correct for model
parallelism:

- TP ranks cooperating on one model replica need identical token IDs.
- PP stages in one pipeline need one coherent macro-batch and target set.
- Only distinct DP coordinates should consume different data.

With the current design, every global rank can receive a different shard
subset.

### Finding 2: pipeline stages independently draw batches

Every PP rank advances its own loader. Because those loaders are global-rank
sharded, stage 0 can receive inputs from one document while the tail stage has
targets from another. The trainer also supplies local `x` values to every
stage's schedule call rather than making first-stage input and last-stage target
ownership explicit.

### Finding 3: reductions described as DP-only use the world group

`distgpt/distgpt/utils/dist.py::all_reduce_mean` reduces over the default world
process group. The trainer uses it for training and evaluation while comments
describe DP-only aggregation. Under PP, intermediate stages contribute zero
loss, so a world reduction can dilute the tail-stage result.

### Finding 4: spike rewind is not collective across pipeline stages

Only the final PP rank observes the loss and invokes rewind. Rewind enters a DCP
checkpoint load, which is collective. Other pipeline stages can continue
training while the tail rank enters checkpoint collectives, creating a
rank-divergence or deadlock risk.

### Finding 5: PP tests stop at construction

The PP smoke test verifies layer trimming and schedule construction but does not
run the complete trainer or a real `schedule.step()` with topology-correct
inputs and targets. The test's prose is stronger than the executed assertion
surface.

### Recommended implementation

Introduce an explicit topology context:

```python
@dataclass(frozen=True)
class ParallelContext:
    global_rank: int
    dp_rank: int
    dp_size: int
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    dp_group: ProcessGroup
    tp_group: ProcessGroup
    pp_group: ProcessGroup
```

Then:

1. Construct loaders with `dp_rank` and `dp_size`.
2. Ensure TP and PP peers with the same DP coordinate advance identical loader
   state.
3. Pass inputs only where the pipeline schedule expects first-stage inputs and
   targets only where it expects final-stage targets.
4. Reduce train/eval losses over the DP group.
5. Broadcast the spike decision across the relevant model-parallel replica.
6. Make every rank enter checkpoint rewind collectively.
7. Save loader state by DP replica rather than blindly by global rank.

Add full-trainer integration tests for:

- TP=2.
- PP=2 with an actual schedule step.
- DP=2 × PP=2.
- A forced spike under PP proving all ranks rewind to the same step.
- Batch fingerprints proving TP/PP peers match while DP replicas differ.

Until that lands, use the more precise project claim:

> Proven DP/FSDP2 trainer with tested TP and PP primitives; combined full 3D
> trainer integration remains experimental.

## P0 — Preserve per-rank state in `midgpt` DDP checkpoints

At startup, each rank receives different RNG streams:

```python
torch.manual_seed(cfg["seed"] + rank)
gen.manual_seed(cfg["seed"] + rank * 1000003)
```

Only rank 0 writes the shared checkpoint. That checkpoint includes rank 0's:

- CPU RNG state.
- CUDA RNG state.
- Training batch-generator state.

Every rank subsequently loads those same values. After resume, ranks can use
identical data-sampling and dropout streams, reducing effective data-parallel
diversity and breaking interrupted-versus-uninterrupted equivalence.

### Recommended implementation

Either:

1. Gather rank-local RNG and generator states to rank 0 and store a
   `rank_states` array in the checkpoint; or
2. Write a small `state_rankNN.pt` sidecar per rank beside the shared
   model/optimizer checkpoint.

Add a two-rank gloo regression:

1. Run N steps uninterrupted.
2. Run K steps, checkpoint, restart, then run N−K steps.
3. Compare final model and optimizer state.
4. Record batch fingerprints and prove resumed ranks remain distinct.
5. Verify interrupted and uninterrupted results match within the intended
   determinism tolerance.

## P0 — Provide a real generated-code isolation boundary

The repository has no Dockerfile, container executor, or gVisor/Firecracker
adapter. The implemented boundary is a resource-limited subprocess. Source code
correctly warns that this is not a jail, but some README wording describes the
evaluator as being in a Docker sandbox.

### Immediate documentation correction

Use:

> HumanEval-style evaluation in a resource-limited subprocess. Run the
> evaluator itself inside Docker, gVisor, or a microVM for untrusted model
> output.

### Recommended implementation

Add a container executor with:

- Read-only root filesystem.
- No network.
- Disposable empty work directory.
- Non-root UID.
- Dropped Linux capabilities.
- Seccomp/AppArmor policy.
- PID, CPU, memory, file-size, and descriptor limits.
- Hard wall-clock timeout.
- One completion per disposable container or worker VM.
- An explicit `--unsafe-local` mode for trusted development.

The local safer path should prefer multiprocessing `spawn`. Python 3.14 already
warns about calling `fork()` from a multithreaded process in the current tests.

## P1 — Make installation metadata complete and reproducible

### Current state

- `distgpt/pyproject.toml` and `frontier-platform/pyproject.toml` contain package
  metadata but no runtime dependency list.
- `pip install -e .` therefore does not install all packages needed by tested
  runtime code.
- `frontier-platform` has no requirements file.
- Several projects use broad lower bounds without a tested lock or constraints
  file.

### Recommended structure

Declare core dependencies in PEP 621 metadata for packaged projects:

```toml
[project]
dependencies = [
    "torch>=2.4",
    "numpy>=...",
    "PyYAML>=...",
]

[project.optional-dependencies]
test = ["pytest", "pytest-timeout"]
eval = ["lm-eval", "transformers", "safetensors"]
gpu = ["transformer-engine"]
serving = ["vllm"]
tokenizer = ["tokenizers"]
```

For requirements-based projects, publish tested constraints or lock files:

```text
requirements.in
requirements.txt
requirements-test.txt
requirements-optional.txt
```

Every benchmark should also record:

- Python, PyTorch, CUDA, driver, and NCCL versions.
- GPU model and count.
- Exact Git commit and config hash.
- Dataset revision and tokenizer revision.
- Seed and dependency freeze.

## P1 — Rename the frontier import package away from `platform`

The `frontier-platform/platform` package shadows Python's standard-library
`platform` module. Its `__init__.py` manually proxies missing attributes to the
stdlib implementation so that pip, pytest, PyTorch, and other tools keep
working.

The workaround is clever but fragile. Rename the import package to:

```text
frontier_platform/
```

A temporary compatibility package can re-export the new package with a
deprecation warning. This migration is easier now than after more downstream
users depend on `from platform...` imports.

## P1 — Harden distributed checkpoint publication and retention

`CheckpointManager.save()` creates the final step directory before DCP
finishes. `latest()` treats any `step_*` directory as valid. A crash can leave a
partial directory that is chosen on restart.

`best.txt` can also reference a checkpoint removed by keep-last garbage
collection.

### Recommended changes

- Save into a temporary step directory.
- Complete DCP and per-rank metadata writes.
- Write a `_SUCCESS` manifest with schema and topology metadata.
- Atomically publish the completed directory.
- Make `latest()` ignore directories without `_SUCCESS`.
- Exempt the best checkpoint from garbage collection.
- Validate `best.txt` on startup.
- Version model, optimizer, loader, topology, and checkpoint schemas.

Also qualify the phrase “reshardable across topologies”:

- DCP model and optimizer state may be reshardable.
- `StreamingLoader.load_state_dict()` rejects a changed world size.
- A complete exact resume across a topology change is therefore not currently
  supported.

Provide an explicit model/optimizer warm-resume mode that resets the loader.

## P1 — Refactor monolithic trainer entry points

Static complexity analysis identified the highest-complexity critical paths:

| Function | Cyclomatic complexity | Approximate length |
|---|---:|---:|
| `midgpt.train.main` | **67** | 292 lines |
| `distgpt.training.trainer.train` | **55** | 282 lines |
| `nanogpt-edu.train.main` | **35** | 186 lines |

Compactness remains educationally useful in `nanogpt-edu`. The larger trainers
now mix device setup, process groups, model construction, optimizers,
checkpointing, evaluation, logging, stability control, and cleanup.

Extract small units such as:

```python
RuntimeContext
TrainingState
build_model(...)
build_optimizers(...)
load_or_initialize_state(...)
train_step(...)
evaluate(...)
save_checkpoint(...)
close_runtime(...)
```

Use `try/finally` so exceptions close JSONL logs, W&B sessions, and process
groups.

## P1 — Add typed config validation

Nested dictionaries currently carry configuration through most training entry
points. Misspelled or incompatible values often fail late.

Each project can remain standalone while defining its own validated dataclasses.
Validate before model allocation:

- `d_model % n_head == 0`.
- `n_head % n_kv_head == 0`.
- `dp * tp * pp == world_size`.
- Context and sequence-length compatibility.
- Pipeline-stage and microbatch compatibility.
- GRPO group size versus effective distributed batch.
- Precision support on the selected hardware.
- Optional kernel availability.
- Checkpoint/config compatibility.

Static analysis estimated roughly 40% return-type coverage and 72% parameter
type coverage across the repository. Public builders and configuration APIs
should be improved first; tests do not need maximal annotation coverage.

## P1 — Strengthen CI and stop swallowing install failures

The workflow currently treats some dependency and editable-install failures as
warnings:

```bash
pip install -r requirements.txt || echo "WARN: some deps missing; tests may skip"
pip install -e . || true
```

This can hide a broken installation behind test path shims or skipped tests.

### Required CPU CI

- Fail if core dependency installation fails.
- Fail if editable package installation fails.
- Run `pip check`.
- Test the supported Python range, not only one version.
- Import packages and run CLI help from outside project directories.
- Run existing tests and Ruff.
- Track a warning budget.

### Optional dependency CI

Use separate jobs for:

- HF/TRL integrations.
- Tokenizer extras.
- lm-eval and HF export.
- Serving or fused-kernel adapters where the environment supports them.

### Scheduled GPU CI

On a self-hosted runner, run:

- One real CUDA training step per runnable trainer.
- Two-rank NCCL DDP.
- Two-rank FSDP2.
- A combined model-parallel smoke after the topology fix.
- HF export followed by one-token serving generation.

## P1 — Add a machine-readable experiment registry

The prose results are strong, but metrics are copied manually across multiple
README files and plots. Add normalized records, for example:

```text
results/
├── schema.json
├── nanogpt-edu/tiny-clean-5060ti.json
├── midgpt/llamafied-ab.json
├── distgpt/fsdp2-2gpu-pcie.json
└── coder-finetune/lora-rank-ab.json
```

A record should include:

```json
{
  "project": "midgpt",
  "commit": "...",
  "config_sha256": "...",
  "dataset": {
    "name": "HuggingFaceFW/fineweb-edu",
    "revision": "...",
    "tokenizer": "gpt2",
    "tokens_seen": 131072000
  },
  "hardware": {
    "gpus": 2,
    "model": "RTX 5060 Ti",
    "driver": "...",
    "torch": "...",
    "cuda": "...",
    "nccl": "..."
  },
  "seeds": [1337],
  "metrics": {
    "best_val_loss": 3.873,
    "best_val_ppl": 48.1,
    "tok_per_s": 14800
  },
  "artifact_paths": []
}
```

Generate README tables and plots from these records to reduce documentation
drift.

# Documentation corrections

## Root README

1. `configs/tiny.py` writes to `out/tiny`; the sample command should use
   `out/tiny/ckpt.pt`, not `out/ckpt.pt`.
2. Four projects—not three—show real single-GPU training runs:
   `nanogpt-edu`, `midgpt`, `distgpt`, and `coder-finetune`.
3. `pip install -e .` is insufficient for the currently declared `distgpt` and
   `frontier-platform` package metadata.
4. The README says “See individual subprojects” for licensing, but no
   `LICENSE*` files were found. Add and scope a real license.
5. The frontier status should describe executable local references plus
   production integration stubs rather than “most bodies raise
   `NotImplementedError`.”

## `distgpt`

The README advertises `distgpt train|eval|sample`, but `cmd_sample()` reports
that sampling is not implemented. Implement it or remove it from the advertised
CLI.

The “every code path is implemented” claim should explicitly account for:

- Unsupported MoE+TP.
- Unsupported MLA+TP.
- Unsupported HF export combinations.
- Experimental combined full-trainer 3D integration.

## `coder-finetune`

Replace claims of a provided Docker sandbox with precise subprocess wording
until a container executor exists. Also distinguish HumanEval from HumanEval+
and built-in toy task sets from EvalPlus-backed evaluation.

## `frontier-platform`

Replace “architecture-only” with:

> An executable reference architecture and discrete-event simulator. Local
> toy implementations validate interfaces; production-scale adapters remain
> explicit stubs.

## `JAAICODE.md`

The doctrine is stale in several places:

- It forbids a top-level test runner although `tools/orchestrate.py` exists and
  is documented.
- It describes frontier modules as mostly `NotImplementedError` interfaces.
- It understates the number of design documents.
- It lists `coder-finetune/data/` instead of `cf_data/`.
- It implies all projects have project-local pytest configuration.

Update the doctrine to permit **top-level orchestration without a shared build
or runtime dependency**.

# Scientific and evaluation improvements

## Protect autonomous research from adaptive validation overfitting

The `nanogpt-edu/research` harness repeatedly keeps or rejects candidates using
one validation metric. The train-versus-validation gap gate catches model
overfitting to training data, but not research-policy overfitting to validation
after many adaptive experiments.

Use three splits:

```text
train
search-validation       # visible to the agent
confirmation-test       # hidden from routine search
```

Recommended protocol:

1. Search against `search-validation`.
2. Apply finite, descent, and train-gap gates.
3. Re-run retained candidates under multiple seeds.
4. Evaluate promoted candidates once on the hidden confirmation split.
5. Do not feed confirmation results into routine candidate selection.
6. Rotate or replenish the search-validation split periodically.

Report means and standard deviations across 3–5 seeds for promoted results. A
single fixed seed provides deterministic comparison, not variance estimation.

## Add uncertainty and sensitivity to simulator results

The frontier simulator emits precise point estimates for cost, wall time,
failures, and capability. Add:

- Low/base/high MFU scenarios.
- GPU-price ranges.
- Network and checkpoint-overhead distributions.
- Scaling-law coefficient uncertainty.
- Monte Carlo intervals.
- Sensitivity or tornado charts.
- Backtests against real `nanogpt-edu`, `midgpt`, and `distgpt` runs.

Prefer outputs such as:

```text
7B wall time:
  p10: 3.8 days
  median: 5.1 days
  p90: 8.7 days

Dominant uncertainty:
  48% MFU assumption
  31% checkpoint/recovery overhead
  14% data-pipeline stalls
```

## Standardize evaluation without coupling projects

Preserve standalone execution, but adopt a common result schema with
project-local adapters. Useful fields include:

- Validation loss, perplexity, and bits per byte.
- Tokens seen.
- Total and active parameters.
- Throughput and MFU.
- Allocated and reserved peak VRAM.
- Evaluation task and dataset revision.
- Contamination status.
- Seed and uncertainty interval.
- Checkpoint, config, and dataset fingerprints.

## Label evidence levels explicitly

Use consistent evidence labels:

- **Unit:** pure helper behavior.
- **Integration:** components wired together.
- **Distributed smoke:** real process groups and collectives execute.
- **Training smoke:** loss is finite or descends briefly.
- **Reproduction:** a published metric is recreated.
- **Benchmark:** controlled performance measurement.
- **Scientific A/B:** controlled comparison with seed/uncertainty reporting.

# Maintainability and repository hygiene

## Manage intentional duplication with provenance

Do not create a shared runtime package merely to remove duplicated Muon,
transformer, schedule, or plotting code. Instead use:

- “Ported from” headers with source commit IDs.
- Golden-vector tests in every copy.
- A matrix documenting intentional divergences.
- A periodic parity checker that reports drift without coupling projects.

## Burn down warnings

Current warning themes include:

- Python 3.14 and PyTorch JIT deprecations.
- Future DCP overwrite behavior.
- Multithreaded `fork()` safety.
- Scheduler call order.
- Uncompiled FlexAttention.
- Tensor-to-float conversion while gradients are attached.

CI should reject new warnings while maintaining a small explicit baseline for
known issues.

## Make top-level examples portable

The top-level examples pin host-specific GPU UUIDs. Support conventional device
selection:

```bash
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

and optionally:

```bash
./run.sh --device 0
./run.sh --device-uuid GPU-...
```

Probe and print the selected device rather than requiring source edits.

## Ignore transient local logs precisely

Two `tmux-*.log` files were untracked during the review. Add focused patterns
such as:

```gitignore
tmux-*.log
*.local.log
```

Avoid blanket `*.log` ignores if benchmark logs are intentionally committed.

## Add command and documentation smoke tests

Run cheap command tests in CI:

```bash
python train.py --help
python sample.py --help
python -m distgpt.cli train --help
python -m distgpt.cli eval --help
python -m distgpt.cli sample --help
python scripts/simulate.py --help
```

Also validate:

- Markdown links.
- Referenced image existence.
- Referenced config paths.
- README command paths.
- Test counts generated from actual test output rather than manually copied.

# Implementation roadmap

## Phase 1 — correctness and safety

1. Fix `distgpt` loader rank semantics for DP×TP×PP.
2. Replace world-group reductions with explicit topology-group reductions.
3. Make PP spike rewind collective.
4. Add full TP/PP trainer integration tests.
5. Fix `midgpt` per-rank DDP resume state.
6. Correct Docker sandbox wording.
7. Add a real isolated code executor or an explicit `--unsafe-local` mode.

## Phase 2 — reproducibility and packaging

1. Add complete PEP 621 dependencies and extras.
2. Add constraints or lock files.
3. Add benchmark environment manifests.
4. Add atomic DCP completion markers and protect best checkpoints.
5. Add a machine-readable results schema.
6. Rename `platform` to `frontier_platform`.
7. Add a repository license.

## Phase 3 — maintainability and evidence

1. Refactor `midgpt.main` and `distgpt.train`.
2. Introduce typed, validated config objects.
3. Add a supported-Python CI matrix.
4. Stop swallowing installation failures.
5. Add warning budgets.
6. Add scheduled GPU/NCCL tests.
7. Add hidden confirmation data and multi-seed promotion to the research
   harness.
8. Add uncertainty intervals and sensitivity analysis to the simulator.

# Recommended target state

| Project | Defensible target claim |
|---|---|
| `nanogpt-edu` | Minimal educational GPT plus a controlled small-scale research workbench |
| `midgpt` | Reproducible single-node GPT pretrainer with correct DDP resume and external eval/export |
| `distgpt` | Tested DP×TP×PP trainer with topology-correct data, metrics, checkpointing, and failure recovery |
| `coder-finetune` | Consumer-GPU SFT/preference/RLVR stack with a real isolated verifier boundary |
| `frontier-platform` | Executable frontier-system reference architecture and uncertainty-aware program simulator |

# Final assessment

`LLM-playground` is already substantially stronger than a typical educational
LLM monorepo. It combines readable model code, distributed systems, real
measurements, post-training, evaluation, safety, and frontier-scale systems
thinking without hiding negative results.

The next stage should be:

> **Make every maturity claim precise, every resume and topology path correct,
> every benchmark reproducible, and every security boundary real.**

The most urgent engineering task is the `distgpt` 3D topology audit. The
fastest high-value cleanup is the documentation and packaging pass. The most
important scientific improvement is adding hidden confirmation evaluation and
uncertainty to the research and simulation layers.
