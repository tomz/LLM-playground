# Project-local testing and reproducibility

The five projects remain independently installed and tested. There is no new
top-level build or test runner. Each project provides:

- `requirements-test.txt`: the dependencies needed by its CPU tests, including
  actual integration libraries where the tests exercise them;
- `constraints-ci.txt`: exact versions of the direct dependencies selected for
  the CPU test profile;
- its existing runtime `requirements.txt` or package metadata, with optional
  backends kept separate from the CPU baseline.

## Install and test one project

From the chosen subproject directory, in a fresh virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -c constraints-ci.txt setuptools wheel
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.0'
.venv/bin/python -m pip install --no-build-isolation -r requirements-test.txt
.venv/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -ra --timeout=180 tests
```

These are CPU-profile commands, not GPU installation instructions. The profile's
NumPy version requires Python 3.11 or newer. CI is configured for Python 3.11 and
3.14; a locally passing 3.14 run is not proof that the remote 3.11 job has passed.
Do not overwrite an existing environment just to follow this example: inspect
its dependencies or use a separate environment for a compatibility experiment.

`torch==2.12.0` accepts the matching CPU local-version build. Installing it from
the CPU index first keeps a normal requirements installation from selecting a
CUDA build for CPU CI. GPU precision, FSDP/NCCL, QLoRA, and serving backends need
separately tested hardware/software profiles; CPU test results do not validate
them.

## What the constraints mean

The constraints are **tested direct-dependency profiles, not complete transitive
or hash-verified lockfiles**. Transitive resolution and platform wheels can differ.
CI records `pip freeze --all` and runtime metadata so the resolved environment is
visible. A publication-quality experiment should retain that complete manifest
and artifact hashes, plus the code revision and dirty source hashes.

An upgrade is a deliberate compatibility change:

1. Update runtime bounds and the affected project-local profile together.
2. Install into a clean environment without suppressing errors.
3. Run real integration tests, not only mocked constructor dispatch.
4. Check effective configuration and fail on unsupported mandatory settings.
5. Retain the old profile or record migration constraints when checkpoint,
   tokenizer, loss, sampling, or backend behavior changes.

Projects deliberately do not import a shared runtime dependency helper. Similar
profiles are duplicated so each project remains self-contained.

## CI acceptance contract

[The workflow](../.github/workflows/tests.yml) has a project × Python matrix and
per-project working directories. It:

- fails on dependency or editable-install errors;
- runs `pip check` and checks installed versions against the direct constraints;
- requires actual Transformers/PEFT/TRL imports for coding integration, and
  actual Transformers/safetensors imports for export integration;
- runs tests offline with CUDA hidden, bounded CPU threads, per-test timeouts,
  skip reasons, and a JUnit report;
- builds the two installable projects without network build dependencies, installs
  their wheels into temporary environments, and imports them outside the checkout;
- imports stdlib `platform` before `frontier_platform`, guarding against the old
  package-name collision;
- uploads resolved dependencies, runtime metadata, and test results even when a
  later step fails.

The temporary wheel-check environment inherits the already installed job
dependencies, but the project package itself must load from the new wheel's
installation location—not the checkout or an editable path. This tests wheel
packaging separately from dependency installation.

The workflow definition and its embedded Bash/Python can be checked locally.
That is **not** a substitute for observing the GitHub-hosted matrix complete.

## Required versus external integration tests

Required CPU/HF integration paths should not silently skip because a declared
dependency failed to install. External prerequisites remain distinct:

- a locally provisioned, explicitly trusted immutable container image and a
  functioning isolation backend for generated-code evaluation;
- supported GPUs and kernels for low precision and accelerator parallelism;
- independently provisioned model/data artifacts for non-synthetic capability
  benchmarks;
- real serving engines and distributed worker infrastructure for operational
  tests.

A missing prerequisite should produce an explicit unavailable/skipped result,
never a fabricated pass, zero-cost failure, unsafe host fallback, or a claim that
a production security boundary was tested. Tests using trusted toy programs and
mocked container invocations remain contract evidence only.

## Evidence to retain for a repair or experiment

Keep project-local test commands and counts, expected skip reasons, exact
configuration, dataset/task identities, code and checkpoint hashes, and all
attempts—including failures. Report unknown usage as unknown. Compare the next
optimizer update and next data samples for restart validation; successful file
loading alone is not a restart test.

For quality or performance work, add independent holdouts, repeated seeds,
regression metrics, precision/kernel/backend identity, warm-up rules, timing
samples, memory definitions, and the complete generated/graded-token budget.
Separate measured results, synthetic fixtures, and simulator assumptions.
