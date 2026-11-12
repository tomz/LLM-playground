# 20 — Serving, rollout, and execution-cost repair contracts

## Scope and status

This repair addresses the CPU counterexamples in the repository assessment's
F7, plus vLLM configuration fallback from F8. It does not establish frontier
quality, concurrent GPU execution, distributed actors, or real-vLLM compatibility.
The existing `frontier_platform` namespace is unchanged.

| Path | Contract | Evidence boundary |
|---|---|---|
| Torch serving | Token-exact requests, minimal top-p nucleus, actual behavior probabilities, explicit finish/usage | CPU reference tests, including cached/full-forward parity |
| Sync/async rollout | Pre-generation context admission, exact reward/learner IDs, frozen actor snapshot, complete stream required | In-process CPU tests; no distributed weight transport |
| GRPO/GSPO | Observed behavior probabilities; matching positive-temperature softmax; eval-mode forward with gradients | CPU numerical fixtures; greedy/top-p training rejected |
| Multi-agent dispatch | Declared bounds admitted before dispatch; attempted work always accounted for | Cooperative local callbacks, not a process sandbox or durable ledger |
| vLLM adapter | Mandatory options never silently removed; raw model probabilities never labelled behavior probabilities | Fake-backend tests only; batch replay, not live token streaming |

## Exact trajectories and actor lifecycle

Each prompt must be nonempty and fit both the learner `seq_len` and the model's
context limit. Invalid prompts fail before any request in the group starts.
Generation is capped to `min(max_new_tokens, context_limit - prompt_length)` per
row. A full-context prompt or zero-output request produces an empty response;
an empty prompt batch produces `[0, 0]` tensors. The learner rejects a batch with
no response tokens rather than performing a meaningless update.

The returned generated IDs, response mask, behavior log-probabilities, and reward
text all describe the same continuation. EOS is retained once in the IDs/mask
and passed to the tokenizer with the other generated IDs. The tokenizer may
suppress special tokens during decoding; no additional continuation is generated
and then discarded. Reward text is decoded once from the entire continuation,
not assembled from per-token byte/BPE fragments.

A successful stream has exactly one terminal event. Its finish reason, stop ID,
prompt usage, completion usage, and probability kind must agree with the observed
tokens. Early exhaustion, extra tokens, duplicate/missing completion, errors,
cancellation, missing log-probabilities, and nonfinite/positive log-probabilities
are not training examples. Any group failure cancels and drains its siblings
before the actor permits synchronization again.

The actor deep-copies model parameters and buffers, disables gradients, and keeps
its own eval-mode model. Learner updates do not alter this snapshot. Only an
explicit `sync_weights()` replaces it and advances its version. A guard rejects
sync or a second group while generation is in flight; streams within the group
share one snapshot and version. Callers must serialize learner mutation with the
synchronous copy itself and must not mutate the actor model directly. The sync
sampler also uses an isolated snapshot. Copying costs memory and time; it is an
intentional correctness-first choice for this tiny reference.

`GroupRollout.provenance` is optional for constructor compatibility. New rollouts
carry an immutable receipt with the policy version (async actor only), sampling
parameters, exact response IDs, generation ceilings, finish reasons, stop IDs,
and log-probability kind. CPU forwards block the event loop: an async API alone
does not demonstrate concurrent generation, continuous batching, or GPU scaling.

## Learner distribution

The supported training distribution is `softmax(logits / temperature)` with
finite `temperature > 0` and `top_p == 1`. The actor, learner, reference KL, and
entropy computation use that same transformation. Learner/reference forwards
run in eval mode to avoid a dropout-induced distribution change, while learner
gradients remain enabled. Caller training modes are restored.

The learner checks provenance against its configuration and rejects greedy or
nucleus-truncated training. Truncated supports can change between updates and
require an explicitly designed importance/support contract, not an assumed
on-policy ratio of one. Generation itself still supports greedy/top-p.

Observed, finite behavior log-probabilities are mandatory. Older records with
no provenance can supply these under the legacy unit-temperature/full-softmax
contract only. A record with no behavior probabilities is still constructible,
but cannot be trained: `logp - logp.detach()` is not a substitute for missing
sampling evidence. Fixed weights should yield ratios near one; skewed weights
must use the actual observed denominator.

## Worker admission and attempted-work accounting

`ReservedWorker(max_tokens, run)` declares an integer upper bound before work;
`run(task, max_tokens)` receives that bound. The dispatcher admits it only when
the entire bound fits the remaining shared budget. Unused reservation is released
when the measured count returns. The callback must enforce its own limit; a
local Python wrapper cannot preempt arbitrary code. An overrun is recorded at
its actual count, even beyond the shared budget, and stops further dispatch.

Legacy `worker(task) -> (answer, tokens)` callbacks remain usable but have no
pre-dispatch bound. They cannot promise hard containment. Every attempted branch
is retained, including worker/scorer exceptions, invalid/missing counts, and
reservation/budget overruns. `tokens_used` is `None` when any attempted work is
unmeasured; `measured_tokens` preserves the known subtotal. Zero denotes measured
zero or no dispatch, never unknown execution. Per-branch elapsed time and error
status remain visible. Only completed, valid branches enter reconciliation.
The reference executes branches sequentially despite the historical
`run_parallel` name; durable reservations, cancellation, and external metering
remain production work.

## vLLM compatibility boundary

The adapter submits the requested TP, PP, dtype, context/concurrency limits,
prefix cache, chunked prefill, and speculative draft options without retrying a
reduced configuration. An unsupported constructor option fails explicitly. A
constructed backend exposes the submitted effective configuration; prebuilt
injected backends cannot prove their configuration and mark it unknown.

Returned chosen-token probabilities are labelled `raw_model`, not `behavior`:
temperature, nucleus filtering, or version-specific processing can make these
different. Missing entries remain `None`/`unavailable`; nonfinite entries fail.
This adapter is therefore not a verified source of GRPO trajectories. It runs a
batch call in an executor and replays completed tokens; cancelling the Python
consumer does not prove that the underlying backend job stopped. The private
weight-loading hook is version-specific and remains unverified on real vLLM.

## Code-verifier execution boundary

`CodeUnitTestVerifier` now requires an explicit `executor(code, test, limits)`
adapter. An absent backend fails at construction, before generation/training;
backend exceptions propagate instead of becoming ordinary zero rewards. The
adapter must return an explicit Boolean verdict. Its isolation, cleanup and
grader integrity require separate operational validation—dependency injection
is not proof of containment.

`trusted_test_only=True` retains the old host-subprocess path solely for
host-authored test fixtures. It is not safe for model outputs: host filesystem
and network access, descendants, pipe output and test tampering are not fully
contained. The low-level sandbox/jailer reference helpers remain available, but
auto-detection or a warning is not an authorization to use them on untrusted
code. Container/VM deployment validation remains external; no cross-project
imports from coder-finetune were added.

## Validation and rollout

Run from this subproject, using the existing environment without installation:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests
```

The pre-repair CPU baseline is **454 passed, 15 skipped**. Focused regressions live
in `tests/test_rollout_contracts.py`, `tests/test_multi_agent_contracts.py`,
`tests/test_serving_contracts.py`, and `tests/test_vllm_engine.py`. Acceptance is
numerical/behavioral, not a throughput or model-quality claim. A real backend
must separately establish its supported version/configuration, probability
semantics, stop/cancellation behavior, and weight-sync safety before promotion.
