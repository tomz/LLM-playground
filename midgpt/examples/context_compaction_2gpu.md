# Context compaction prefill A/B — 2× RTX 5060 Ti

A systems-only measurement of the July retained-state/compaction harvest using
`gpt2_350m_llamafied_fweb_5060ti_2gpu/ckpt_best.pt`. Semantic fidelity is tested
separately in `tests/test_context.py`; this run asks how much GPU prefill cost is
saved after a controller reduces retained context from 1,024 to 256 tokens.

## Method

- One arm per RTX 5060 Ti 16 GB, launched concurrently in separate processes.
- Same 350M checkpoint, bf16 autocast, batch size 1, last-token logits.
- Three warmups and 20 measured forwards per arm.
- Script: `tools/bench_context_compaction.py`.
- Raw artifact: `examples/context_compaction_2gpu.json`.

## Result

| Arm | Context | Mean prefill | Tokens/s | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| Full retained history | 1,024 | 27.81 ms | 36,821 | 1,483 MiB |
| Compacted state | 256 | 11.97 ms | 21,386 | 1,480 MiB |

Compaction removed **75% of input tokens** and reduced wall-clock prefill by
**57.0%**. It did **not** materially reduce peak VRAM at this batch/model shape:
weights dominate the ~1.48 GiB allocation and the shorter activations save only
~3 MiB. Tokens/s is lower for the short arm because fixed kernel/launch overhead
is amortized over fewer tokens; per-request latency is the relevant metric.

## Interpretation and limits

This validates the systems mechanism, not the ARC-AGI-3 capability claim. The
model has no KV cache and the benchmark performs a full prefill each time, so a
production cached decoder will have a different absolute curve. The companion
unit tests establish the controller invariant: unlike rolling truncation,
compaction retains pinned constraints and old decisions under the same budget.
A task-level agent benchmark is still needed to measure whether those summaries
preserve downstream success.
