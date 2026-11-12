# October–November harvest: bounded verification and specialization

**Two runnable, deliberately narrow additions:** a fail-closed Docker/Lean
adapter for structured proof submissions, and an offline CPU comparison of a
tiny baseline classifier against full SFT and rank-one readout LoRA. Neither is
a frontier-model result. The Lean adapter has hermetic coverage but **real Lean
execution remains unverified on this machine**: Docker daemon access is denied,
and no trusted local Lean image ID was supplied.

## 1. Structured proofs with a real jailed checker adapter

### Contract and accepted subset

Run from `coder-finetune/`. `configs/lean_contract.json` is **host-owned**, not
model output. It pins the proposition and an ordered list of premises. The
wrapper names those premises `h0`, `h1`, etc., imports only `Init`, declares
`theorem certified`, and prints its axioms. The model supplies the JSON format
already accepted by `eval/structured_proof.py`.

`eval/lean_jail.py` accepts exactly these four certificate strings:

| Certificate | Intended use |
|---|---|
| `by rfl` | Definitional equality |
| `by decide` | Kernel-checked decidable propositions |
| `by intro n; rfl` | One introduced binder followed by reflexivity |
| `by assumption` | A proposition already among the explicit premises |

This is a **closed language**, not an extensible tactic parser. Newlines,
comments, extra declarations, arbitrary identifiers, `sorry`, `admit`, `axiom`,
`native_decide`, `run_tac`, macros, and embedded commands are rejected before
Docker is called. No substring denylist is used. Even alternate whitespace is
rejected. The adapter compares the submitted theorem to the host theorem
byte-for-byte; the older generic evaluator retains its whitespace-normalizing
comparison, but cannot make the stricter adapter accept a different target.

The theorem and premise types themselves are trusted Lean source. Review their
meaning, syntax, and consistency independently; do not generate the host
contract from an untrusted submission. An inconsistent premise can prove
anything. A malicious host contract or modified image is outside this boundary.
The allowlist blocks certificate injection; it does not establish that the
host asked a meaningful theorem or that its image contains an honest kernel.

### External prerequisites, without downloads

An operator must provision and independently trust a **local** Linux container
image containing Lean 4 and its `Init` library. Its Lean executable must run as
UID/GID `65534:65534` with a read-only root. The default executable path is
`/opt/lean/bin/lean`; use `--lean-path` for a different installation. The image
must not require a toolchain-manager download on first invocation. There is no
bundled image, fabricated digest, automatic pull, host Lean fallback, or claim
that an image was verified here.

Set `LEAN_IMAGE` to the immutable `sha256:...` image ID from the trusted local
inventory (`docker image ls --no-trunc`). Tags are not accepted. The adapter
checks that `docker image inspect` resolves to that exact ID. Once the operator
has supplied it, these are complete commands:

```bash
# From coder-finetune/, using the already installed Python environment.
: "${LEAN_IMAGE:?Set LEAN_IMAGE to a trusted locally installed sha256 image ID}"
.venv/bin/python -m eval.structured_proof \
  --contract configs/lean_contract.json \
  --submission examples/lean_submission.json \
  --image "$LEAN_IMAGE" --timeout 10

LEAN_TEST_IMAGE="$LEAN_IMAGE" .venv/bin/python -m pytest -q \
  tests/test_lean_jail.py -k local_lean_integration
```

The CLI prints a `ProofEvaluation`. Its `message` contains a JSON `LeanResult`
for submissions reaching the adapter. A zero exit means certificate acceptance,
**not** novelty or release approval. The CLI never sets `novelty_reviewed`.
Malformed schemas or a mismatched target fail before invocation. Python callers
can use `DockerLeanChecker(...).check(theorem, certificate)` for the typed record
or pass the checker directly to `evaluate_submission` for the existing API.

### Jail and evidence

| Control | Setting |
|---|---|
| Image / network | Immutable local image ID, `--pull=never`, `--network=none` |
| Identity | UID/GID 65534, all capabilities dropped, no-new-privileges |
| Filesystem | Read-only root and source mount; 16 MiB noexec/nosuid `/tmp` |
| Memory / CPU / tasks | 256 MiB memory and memory+swap, one CPU, 32 PIDs |
| Kernel limits | CPU seconds bounded by configured wall limit; 1 MiB file limit |
| Lean limits | One thread, 100,000 heartbeats, recursion depth 512 |
| Wall time | Default 10 s, maximum 30 s; inspect and cleanup each get 5 s |
| Output | Default 16 KiB, maximum 64 KiB; incremental bounded capture |
| Cleanup | Unique container name, forced removal in `finally`, temporary source removed |

The Docker CLI is run without a shell or inherited stdin. Output overflow or
wall timeout kills its process group; the `finally` block separately removes
the container, because killing the CLI alone is insufficient. Docker log storage
is disabled to avoid an unbounded daemon log. Only the temporary proof directory
is mounted, read-only; no Docker socket or repository is exposed in the container.
An inaccessible daemon/image fails closed. Failed cleanup prevents acceptance
and is reported rather than silently ignored.

Each adapter record pins the exact theorem, ordered premises, certificate,
rendered source, checker implementation, and requested/inspected immutable image
identity. It also records the command, limits, captured output, exit code,
outcome (`accepted`, `lean_rejected`, `timeout`, `output_limit`, `runner_error`,
`image_unavailable`, or `cleanup_failed`) and cleanup result. Invalid targets and
certificates have separate preflight outcomes. An image-preflight failure is
reported as `image_unavailable` with its exit code and diagnostic, not as a
failed proof. The source is reconstructible from the contract and certificate;
its temporary copy is removed after checking.

Docker shares the host kernel. This is not a VM-grade multi-tenant jail; use a
dedicated worker or stronger runtime for hostile workloads. A daemon outage can
prevent removal, and an interrupted asynchronous Docker creation may require
operator reconciliation of `proof-*` containers. Do not retry failed cleanup as
though it were a valid proof. Image/kernel supply-chain review and human novelty
review remain external requirements. The old HumanEval/GRPO executor is still a
host-subprocess safety floor, **not** this Lean-specific container adapter.

## 2. Offline specialization comparison

### Question and method

**Question:** does supervised specialization on parity improve held-out parity
accuracy without losing positive-number classification? The hypothesis is that
full SFT learns the new task, while constrained rank-one readout adaptation may
trade specialization against retention. The harness keeps all predeclared
branches, including unfavorable results; it does not select a winning seed.

`eval/specialization.py` trains a real, small character-embedding/MLP classifier
on CPU. “SFT” here means supervised binary cross-entropy training (implemented as
two-class cross entropy), **not** TRL causal-LM fine-tuning. LoRA freezes the
backbone and classifier bias and learns only a rank-one `B @ A` update on the
readout. This is an educational example of the comparison protocol, not an LFM3
benchmark, code-generation evaluation, or claim about PEFT on a large model.

| Split | Fixed examples | Role |
|---|---:|---|
| `base_train` | 80 | Positive-number classifier warm-start, -40 through 39 |
| `train` | 80 | Parity specialization, 0 through 79 |
| `heldout` | 20 | Parity evaluation only, 80 through 99 |
| `regression` | 20 | Sign evaluation only, -50 through -41 and 40 through 49 |

Task identity includes the operation, not just its numeric operand. All task IDs
and canonical prompts are distinct across and within splits; collisions fail
before creating artifacts. Neither held-out nor regression examples enter any
training call. Fixed seeds `[0, 1, 2]`, batch size 16, learning rate .02, 60 common
base steps and 80 specialization steps are declared in
`configs/specialization_smoke.json`. Baseline/SFT/LoRA start from identical base
weights within each seed. Both trained branches see the same seeded minibatch
schedule and learning rate. Neither receives a tuning search, early stopping,
retry, or extra branch budget. Baseline gets no specialization steps.

### Run entirely offline

Requires the project's existing PyTorch installation; no new dependencies are
introduced. The existing HumanEval JSON summary writer is reused, but no
HumanEval dataset download or generated-code execution is invoked. The tiny
training loop is independent of TRL because this fixed classifier is not a
causal language model.

```bash
# From coder-finetune/. Output directories must not already exist.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout 120 .venv/bin/python -m eval.specialization \
  --config configs/specialization_smoke.json \
  --output out/harvest-specialization-smoke

# Load the saved local baseline, with no Hub/model-download fallback.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout 120 .venv/bin/python -m eval.specialization \
  --config configs/specialization_smoke.json \
  --model-dir out/harvest-specialization-smoke/models/seed-0/baseline \
  --output out/harvest-specialization-local

CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest -q tests
```

Local model directories use the documented `tiny-character-classifier/v1`
format: `config.json` (format, fixed vocabulary/width, weights SHA-256) and
`model.pt` (state dictionary only, loaded on CPU with `weights_only=True`). The
harness writes that format for every successful arm and merges LoRA on save.
It accepts only that tiny architecture, caps checkpoint bytes, verifies the
hash, tensor shapes and finite weights, and never accepts a remote model ID.
Arbitrary Hugging Face model directories are deliberately unsupported: loading
one would defeat the fixed CPU/memory budget. Use existing `train.py` separately
for real LM SFT/LoRA; this smoke does not silently launch it.

Configuration validates at most ten distinct seeds, 200 steps per training
phase, batch size 32, and 30 seconds per phase/branch. The example uses 15 seconds.
Deadlines are checked between bounded-size CPU operations, not a hard in-kernel
preemption mechanism; the outer `timeout 120` provides a process-level ceiling
for the example. Forced process termination cannot produce a complete ledger;
a missing final artifact is not a successful run. No GPU, paid API, download,
arbitrary model code, or large training is involved.

### Artifacts and scoring

- `dataset.json`: complete fixed split manifest; its canonical hash is in `run.json`.
- `branches.jsonl`: one flushed record per seed/arm, with every optimizer step
  attempted, its status/loss, and **every planned evaluation**. A failed training
  branch still emits all 40 predictions as failed; nothing is dropped to improve
  the denominator. The CLI exits nonzero if any branch fails.
- `run.json`: budget, source/dataset hashes, starting Git revision, Python/Torch/OS
  environment, base-preparation costs, initial/final model-state hashes, full
  branch records, and across-seed accuracy mean/sample standard deviation.
- `models/seed-N/{baseline,sft,lora}/`: local reusable merged checkpoints with
  artifact hashes. Output directories are never overwritten.

Each split reports accuracy with **all planned examples** in the denominator,
failed count, Brier score and ten-bin ECE with an explicit `calibration_n`.
Failed predictions have no made-up probability and are excluded only from
calibration. Costs include total branch wall seconds, attempted/completed
optimizer steps, padded training character tokens, planned predictions, and
zero API calls. Shared warm-start work is separately logged in `base_preparation`.
One unscored training-prompt warm-up precedes evaluation. Per-example latency
p50/p95/stdev are diagnostic, not a speed ranking: samples vary by prompt, there
are only three training seeds, and setup/serialization are included in branch
wall cost. Character tokens are not language-model tokenizer tokens or dollars.

### Observed local smoke

Executed using the commands above on CPU, one Torch thread, Python 3.14.4,
Torch 2.12.0+cu130, Linux x86-64. No CUDA work was requested. The platform API did
not return a CPU model name (recorded as empty, not invented). The initial smoke
used source SHA-256
`acac8bd943f9cb737b7dfce8407e13b3a971fb8195b77bc4e878aa79bf1cf4db`
and dataset SHA-256
`fc000d3aaf07a46521b0fcf35fa46b1f60d41793c57ee0194705ed01aad7e297`.
All nine branches completed: **360 predictions, no failures**. Accuracy is
mean ± sample standard deviation over three seeds, not a confidence interval.

| Arm | Held-out parity accuracy | Regression accuracy |
|---|---:|---:|
| Baseline | 48.33% ± 2.89 pp | 100.00% ± 0.00 pp |
| Full SFT | 96.67% ± 5.77 pp | 98.33% ± 2.89 pp |
| Rank-one readout LoRA | 55.00% ± 5.00 pp | 66.67% ± 28.87 pp |

Full SFT improved this toy specialization but did not perfectly retain the
regression task. The intentionally constrained LoRA readout did not match SFT
and showed substantial retention variance; that result is retained, not tuned
away. Held-out Brier scores for seeds 0/1/2 were baseline
`.420305/.433123/.279835`, SFT `.132721/.000828/.001788`, and LoRA
`.247180/.271636/.215748`. The JSON contains all calibration metrics.

Total branch wall costs across the three seeds were baseline 0.0086 s, SFT
0.0710 s and LoRA 0.0477 s; common base preparation took 0.4876 s. These tiny
one-run timings include setup effects and **do not support a throughput claim**.
The full run consumed 180 base steps, 240 SFT steps and 240 LoRA steps. The second
CLI run successfully loaded the saved seed-0 baseline locally: nine branches,
no failures; mean parity accuracies were 50.00% baseline, 80.00% SFT, 56.67%
LoRA, with 100% regression for each arm. These runs differ in base initialization
policy, so do not pool them into one benchmark.

Validation: **172 passed, 1 skipped**, with 77 existing Python 3.14 Torch/JIT and
fork-from-multithreaded-process deprecation warnings. The skipped case is the
optional real Lean integration. Mock jail tests cover command controls,
immutable image validation, schema/target rejection, declaration injection,
proof holes, timeout/output/runner failures, and cleanup. No real Lean theorem
was claimed as checked here. Result artifacts remain under `out/` (ignored by
Git); this document records the bounded smoke evidence, not a monthly SOTA score.
