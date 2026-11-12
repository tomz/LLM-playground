# November maintainer repair: research budgets, data and provenance

This is an explicitly authorized **maintainer revision** of the protected
harness contract. It does not permit candidate-search agents to edit the harness
or choose their own evaluation rules. Existing quality numbers describe their
original code/data snapshot and must not be relabeled as results of this revision.

## Exact training budgets

`run_candidate` validates the budget before opening data or building a model.
Token budgets must be positive integers, at least one full update, and divisible
by `batch_size × grad_accum × block_size`. There is no round-up or silent
round-down. A divergent run reports actual partial work. Timer and normalization
inputs must be finite and positive. The loop's default is now 2,097,152 tokens;
a candidate with incompatible step geometry is rejected, not given extra work.

Wall-clock mode is a separate benchmark contract. The research loop records the
whole harness process's elapsed time, separately from any reported training or
verifier duration. It does not claim to preempt arbitrary native/GPU work.

## Document-level data and sealed holdout

`prepare_fineweb.py` assigns a whole document to train, search-validation, or
final-holdout by a seeded content hash **before token concatenation**. Exact
repeats stay in the same split; they are not deduplicated. This is not a
near-duplicate/benchmark contamination filter or license approval.

The streaming writer limits individual document size and buffering, records
source/tokenizer revisions and hashes, per-document offsets/byte counts, and
per-split artifact hashes. Its collection token target may finish the current
whole document; that collection policy is explicitly different from an exact
training-token budget. It never overwrites an existing dataset directory.

UTF-8 bytes per stored token are measured, including zero-byte EOT tokens in the
denominator. The harness verifies artifact identities. Legacy character shards
can be measured from their UTF-8 mapping; unmeasured BPE data requires an explicit,
recorded factor. Sampled NLL scaled by this corpus ratio is an estimated BPB,
not an exact whole-corpus byte likelihood.

The normal harness reads train/val only. `final_holdout=True` is a separate
frozen-candidate evaluation requiring a prepared manifest. The loop rejects any
final-holdout result as a selection signal. This separation does not make
adaptive validation selection statistically unbiased: independent confirmations,
sealed final tests, and all-attempt reporting are still necessary.
This role separation is not a filesystem sandbox or a defense against malicious
candidate code.

## Failed attempts and schema 3

`run_once` removes its temporary result file on success, malformed output,
process failure, timeout, and launch errors. It always retains observed elapsed
time. Missing training-token/verifier usage is **unknown**, never inferred zero.
A malformed or failed result cannot become a keep; final-holdout receipts cannot
enter search selection.

Provenance schema 3 permits nullable training-token and verifier costs. The reader
still accepts v1/v2. Known legacy totals retain their meaning; aggregates and
per-accepted ratios become `null` when the required cost is incomplete. Schema-2
records cannot retroactively claim nullable core costs. Old defaults exist for
constructor compatibility; instrumented callers must supply `None` when work is
unmeasured. Inference cache/reasoning counters remain subsets, not extra tokens.

## Offline acceptance

From `nanogpt-edu/`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q --timeout=180 \
  tests/test_repair_contracts.py tests/test_provenance.py
```

Tests exercise the actual tiny CPU training path, exact-budget rejection,
deterministic whole-document shards, duplicate split isolation, artifact hashes,
legacy byte normalization, process-failure cleanup and sealed-holdout exclusion.
Injected document/tokenizer fixtures need no downloads. No new corpus-quality
or hybrid-architecture superiority result is claimed by this repair.
