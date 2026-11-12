# November repair: rank-local, completed-update restart

`utils/checkpoint.py` now owns training checkpoint capture/restoration. Every DDP
rank participates; only the master publishes the file. The checkpoint retains
per-rank CPU/CUDA RNG and sampling-generator state, scaler state, all optimizer
states, the next update boundary, and best/spike/rewind controller state.
Publication uses the existing atomic writer and propagates failures collectively.

Restart validates architecture, data identity, precision, accumulation and
optimizer/schedule contracts before mutation. It restores each rank's own
random stream instead of broadcasting rank zero's generator state to all ranks.
The supported boundary is a completed optimizer update, not an arbitrary
mid-accumulation interruption.

Legacy single-rank checkpoints remain readable where their state is sufficient.
Legacy DDP or changed-topology resumes that cannot recover rank-local randomness
fail explicitly. Use a reviewed weights-only import/new experiment rather than
calling such a restart exact. Checkpoints remain trusted local training artifacts;
this is not an untrusted-pickle ingestion service.

The real two-rank Gloo tests include nonzero dropout, distinct next-batch IDs,
AdamW and Muon+AdamW, exact subsequent parameter/optimizer state, and collective
capture/publication failures. CPU-enabled GradScaler state is also tested. These
are small correctness experiments, not multi-GPU throughput or pretrained-model
quality results.

From `midgpt/`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q --timeout=180 tests/test_checkpoint.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q -ra --timeout=180 tests
```

Use [the project-local test profile](../../docs/testing-and-reproducibility.md)
for the declared environment. The existing model-backed candidate search remains
a separate bounded experiment; its smoke result does not establish reasoning
quality. Cost per accepted task stays unavailable if any attempt has unknown
usage. Cached decoding, canonical benchmark adapters for every architecture, and
real-corpus quality/efficiency comparisons remain separate next milestones.
