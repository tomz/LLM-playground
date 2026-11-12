# October–November 2026 harvest: measured routing and deployment contracts

This note accompanies `frontier_platform.serving.task_cost`, `task_routing`, and the
stdlib-only `tools/bench_task_routing.py` demonstration. It extends the earlier
[July–August design](15-july-august-2026-harvest.md), without implementing any
production `NotImplementedError` interfaces. Sources and motivation are in the
[October](../../docs/2026-10-sota-llm-agi.md) and
[November](../../docs/2026-11-sota-llm-agi.md) SOTA Watch editions.

## Decision

Use explicit, caller-ordered model/effort routes with an independent answer
checker and authorization callback. Admit each call against a conservative
per-bucket token/tool-cost ceiling. Escalate only after a completed, in-contract
answer fails the quality checker. A refusal, substitute model, incomplete stream,
checker failure, or unknown-cost backend failure stops this task. Escalation
must not become a way to evade safeguards.

This is a deterministic control-plane reference, not a learned router, provider
client, authorization service, sandbox, distributed scheduler, or concurrency-safe
budget bank. The existing heuristic `Router` and multi-agent helper remain
unchanged. No vendor rate or model benchmark is hard-coded into the new modules.

## Accounting contract

- Input, cache-read, cache-write, and output token buckets are disjoint. Reasoning
  tokens are included in output and recorded as a subset, never billed twice.
  Adapters must normalize the provider's overlapping usage fields before entry.
- Tariffs have model identity, rate revision, source, start date, and optional end
  date. The supplied observation date must lie in the tariff's validity interval.
  This checks internal consistency; it does not fetch or certify current prices.
- Each attempt records task/domain, requested and actual model/revision/safeguard
  mode, requested effort, grader version, parent attempt, duration, verifier time,
  outcome, actual usage when known, and the admission reservation.
- Every rejected or failed attempt counts. The denominator for accepted-task cost
  is distinct accepted task IDs, not branches. Repeated trials need different IDs.
- Unknown usage is `null`, not zero. Its reservation stays in budget accounting;
  measured cost per accepted task becomes unavailable if any cost is unknown.
  A substitute model's usage cannot be priced with the requested model's tariff.
  An unknown or substituted bill is not guaranteed to stay below the reservation.
  Further tasks on that ledger are rejected until usage/tariffs are reconciled.
- JSONL stores schema version and decimal money as strings; reload rejects unknown
  versions, invalid amounts, and duplicate attempt IDs. The writer is a local
  export, not an atomic transactional audit log.

## Enforcement and failure boundaries

The adapter receives a `Route.ceiling` and must actually enforce output limits,
input bounds, tool costs, timeouts, and cancellation in its execution environment.
The reference checks reservations before calling it and rejects responses that
exceed ceilings; it cannot undo overspending by a broken adapter. Such usage is
retained instead of discarded. Cumulative budgets cover all tasks in the supplied
ledger. A production concurrent executor needs atomic reservations, durable
usage reconciliation, and provider-specific billing adjustments.

Authorization is a separate callback and must return the Boolean `True`; exceptions
fail closed. It runs before every dispatch. Cooperative cancellation is checked
before dispatch and after generation/checking, but cannot interrupt a hung
synchronous callback. Tool execution is not performed by this module. Provider
identities are adapter-reported; authenticated provenance and enforcement must
be implemented outside it. Prompt instructions are not an authorization boundary.

## Acceptance and reproduction

From `frontier-platform/`:

```bash
.venv/bin/python -m pytest -q tests/test_task_routing.py
.venv/bin/python tools/bench_task_routing.py --out-dir out/task-routing
```

Tests cover exact tariff arithmetic, cache/reasoning normalization, invalid
counts and amounts, tariff expiry, pre-dispatch budgets, shared budgets,
authorization denial/errors, refused and incomplete responses, substitutions,
revision/safeguard mismatch, output overruns, checker exceptions, cancellation,
failed-call costs, escalation lineage, unique identities, and JSONL round trips.

The demonstration compares fixed-premium versus cheap-then-premium policies on
three **synthetic callback fixtures**. It exercises failed-branch accounting and
produces per-attempt JSONL plus a summary. It does not show real model routing
accuracy, task latency, or savings. A real experiment must use a pinned local
model or explicitly approved API adapter, disjoint held-out tasks, repeated
trials, independent grading, all attempts, and confidence intervals.

## Next production gates

1. Implement authenticated provider adapters with correct usage normalization,
   output ceilings, full stream termination semantics, and actual timeout/stop enforcement.
2. Reconcile unknown usage before admitting further work against a shared budget.
3. Evaluate routing on task-specific holdouts and include regression/refusal tests.
4. Add durable reservations, audit storage, and approval revocation for concurrency.
5. Keep scientific novelty and review gates independent of model/checker success.
