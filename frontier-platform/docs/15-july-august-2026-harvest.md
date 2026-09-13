# July–August 2026 harvest: stateful agents and auditable research

This design note accompanies the pure-Python reference implementations added
from the July and August SOTA Watch editions. Production adapters remain
intentionally external; existing `NotImplementedError` platform interfaces are
unchanged.

## Goals

1. Preserve and compact long-running agent state without silent rolling loss.
2. Execute model-written tool plans under explicit call/output budgets.
3. Fan out multiple agents under one token budget and reconcile deterministically.
4. Keep self-play red-team training separate from a frozen held-out release gate.
5. Record every research attempt, human edit, verifier cost, and certificate.
6. Require formal validity, theorem/claim identity, prior-art review, significance
   review, and independent reviewers before a research artifact can be released.

## Reference modules

| Concern | Module | Production replacement boundary |
|---|---|---|
| Research lineage and release gate | `platform.infra.research_artifacts` | Durable object store/catalog and organization review workflow |
| Programmatic tools | `platform.serving.programmatic_tools` | Sandboxed code runner plus asynchronous tool engine |
| Multi-agent budget/reconciliation | `platform.serving.multi_agent` | Concurrent scheduler, cancellation, distributed tracing |
| Self-play red teams | `platform.safety.selfplay_redteam` | Attack model farm and immutable external held-out corpus |

Related implementations are project-local by design:

- `nanogpt-edu/research/provenance.py` adds full attempt/cost JSONL beside its
  chart-oriented TSV ledger.
- `midgpt/context.py` provides retained reasoning, rolling truncation,
  compaction, and fact-fidelity measurement.
- `distgpt/distgpt/eval/rollout_queue.py` pins leases, retries, idempotency, and
  snapshot recovery for rollout/verifier workers.
- `coder-finetune/eval/task_contract.py` audits SWE task contracts, while
  `structured_proof.py` validates proof schemas and injected checker results.

## Invariants

- A tool plan stops on errors or budget exhaustion; raw output size and retained
  output size are both recorded.
- Multi-agent branches share one token ceiling. A branch that would exceed it is
  not partially admitted.
- Held-out attacks never enter self-play curriculum generation. Duplicate train/
  held-out prompts fail the gate through normalized fingerprints.
- Passing a formal certificate is not enough. The certificate must target the
  released claim, and prior-art/significance reviews require independent humans.
- Failed and discarded research attempts count toward total cost.

## Production rollout

1. Keep deterministic CPU tests as backend conformance tests.
2. Add sandboxed/jailed tool adapters and a durable lease store.
3. Add OpenTelemetry spans keyed by artifact, branch, tool call, and work-item ID.
4. Freeze and access-control held-out red-team corpora before training begins.
5. Require artifact JSONL, checker logs, reviewer decisions, and cost summaries
   as release-gate inputs.

## Non-goals

- The reference executor is not a security boundary.
- The in-memory queue is not a distributed database.
- A checker result does not establish novelty or mathematical importance.
- Parallel-agent helpers do not claim frontier-scale speedups without hardware
  measurements.
