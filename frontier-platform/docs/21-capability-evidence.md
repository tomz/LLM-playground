# Capability and evidence boundaries after the November repair

The Python package is now **`frontier_platform`**, not `platform`. Executable CPU
references coexist with intentionally unimplemented production interfaces. This
is not an architecture-only empty skeleton, and it is not a production frontier
platform. See [the repair contracts](20-sota-repair-contracts.md) and
[repository evidence](../../docs/2026-11-repair-pass.md).

| Component | Current evidence level | Boundary |
|-----------|------------------------|----------|
| Transformer, KV cache, torch serving | CPU reference/integration tests | No frontier quality or large-model serving throughput claim |
| SFT/DPO/GRPO references | Tiny real CPU updates and numerical tests | Supported sampling distribution only; not a distributed learner service |
| Async actor interface | Frozen in-process snapshots, exact token/reward receipts, guarded synchronization | CPU forwards block the event loop; no concurrent GPU/continuous-batching claim |
| Multi-agent dispatch | Reserved admission plus measured/unknown attempted-work accounting | Sequential cooperative callbacks; legacy workers cannot promise hard limits |
| Task routing | Tariff, budget, authorization and protocol contract tests | Provider-reported identities; no authenticated provider or durable reservation service |
| vLLM adapter | Fake-backend contract tests | Batch replay, raw-model log-probs, private/version-specific weight loading; not live validation |
| Code verifier | Default denial without an explicit executor | Trusted-fixture subprocess is not isolation; deployment and grader integrity need independent proof |
| Data/infra/review interfaces | Mix of pure helpers, references and deliberate stubs | Interface presence does not establish ingestion scale, review quality or operational enforcement |
| Simulator | Tested scenario/accounting calculations and some calibration helpers | Capability curves, precision speedups and economic inputs can be assumptions; not measured leaderboard results |

## Import migration

Replace `from platform...` with `from frontier_platform...`. The old stdlib proxy
and import-eviction tricks were removed. Both source-tree and installed-wheel
imports are checked with stdlib `platform` loaded first. Do not recreate a global
`sys.modules['platform']` alias to make old application code work.

For the common legacy checkpoint shape containing `model` state plus a pickled
`platform.model.config.ModelConfig`, there is a narrow converter:

```bash
.venv/bin/python tools/convert_legacy_checkpoint.py \
  --source out/legacy.pt --destination out/converted.pt
```

It uses PyTorch **weights-only** loading with only the old ModelConfig name
allowlisted to the new class. Unsupported Python objects are rejected; there is
no unrestricted-pickle fallback. The source is hashed/preserved, output must not
exist, and publication is no-clobber. Tests compare tensor/config values and
prove stdlib imports remain untouched. Use trusted local artifacts and size RAM,
disk and time for their contents; this is not a general untrusted archive service.
Other historical checkpoint formats require a separately reviewed migration.

## What the simulator may claim

A simulated GPU's throughput, a fixed low-precision multiplier, an extrapolated
loss curve and a sigmoid capability conversion are different kinds of evidence.
Retain input assumptions, fit domain and source/version alongside each output.
A code repair does not turn fixed NVFP4/RL gains into measurements. Sensitivity,
uncertainty and real-hardware/model calibration remain next work, and results
outside the calibration domain should be treated as scenarios.

## Next production gates

Durable concurrent reservations/queues; enforced cancellation; authenticated
provider identity; real streaming and weight synchronization; isolated verifier
workers; independent held-out evaluations; and calibrated scale/performance
experiments remain external integrations. Any implementation of an intentional
stub still requires a matching design-doc update. Projects remain independent;
no cross-subproject imports were introduced.
