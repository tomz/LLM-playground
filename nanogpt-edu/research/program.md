# Research program — the agent's brief

You are an automated ML researcher. Your job: **lower `val_bpb`** on this
nanogpt-edu character-LM by editing **one file** — `candidate.py` — under a
fixed compute budget, keeping only changes that *measurably* help.

This is the prompt an LLM agent reads each iteration (the analogue of
`autoresearch`'s task spec / `auto-improving-kernel`'s `program.md`). A human can
follow it too.

## The one rule

**Only `candidate.py` may change.** `harness.py` (training + measurement + gates)
and `loop.py` (keep/revert) are fixed — treat them as the laws of physics. If
you think the harness is wrong, say so; do **not** edit around it.

## The loop (one iteration)

1. Read `candidate.py` and the tail of `ledger.tsv` (what's been tried + the
   running best).
2. Form **one** hypothesis ("qk_norm will stabilise and let me raise LR").
3. Make the **smallest** edit that tests it — flip a knob in `KNOBS`, or write a
   small `patch_model(model)`. Update `DESCRIPTION` to name the change.
4. Run `python loop.py` (one experiment) or let `python loop.py --auto N` drive
   N self-mutating iterations.
5. The harness trains under the budget, measures, and gates. `loop.py` **keeps**
   the edit (lower `val_bpb` + gates pass) or **reverts** it. Either way a row
   lands in `ledger.tsv` and `progress.png` refreshes.
6. Repeat. Change **one thing at a time** — confounded edits teach nothing.

## What "better" means (the metric contract)

- **Headline:** `val_bpb` — validation **bits/byte**, lower is better. Vocab-
  independent, so architecture changes compare fairly (a BPE swap stays honest).
- **Gates (cannot be faked):** a kept run must (a) have finite loss, (b) have
  actually descended (val well below the ~ln(vocab) random ceiling), and
  (c) not overfit (train↔val gap under `max_gen_gap`). Lowering `val_bpb` by
  memorising the train set is rejected — this is nanogpt-edu's `tiny` vs
  `tiny_clean` lesson turned into a guardrail.
- **Secondary (shown, not gated):** throughput, peak VRAM, params. The bottom
  panel of `progress.png` plots quality vs speed so you can see when a win costs
  too much — chase the **Pareto frontier**, not just the lowest point.

## The search space (start here)

The seeded `KNOBS` menu mirrors nanogpt-edu's documented harvest. High-signal
first moves, roughly in order of expected ROI:

1. **`qk_norm: True`** — per-head RMSNorm on Q,K. Stabilises attention logits;
   usually a free win and lets you push LR.
2. **`zero_init_proj: True`** — identity-init each block; stable high-LR warmup.
3. **`optimizer: "muon"`** (+ `muon_lr`) — orthogonalised update, ~1.35×
   sample-efficiency on the FineWeb speedrun. Watch the throughput cost.
4. **Capacity** — `d_model`, `d_ffn`, `n_layer`. Real gains, but they cost
   tok/s and VRAM (see the Pareto panel). Pair a bigger model with a lower LR.
5. **`tie_embeddings: False`** — untie once the model is big enough to use the
   params.
6. **`mtp_tokens: 2`** — Multi-Token Prediction aux heads; denser gradient,
   train-only (zero inference cost).
7. **`dropout`** — your overfit dial if the gen-gap gate trips.

The `patch_model(model)` hook is the open-ended lever: re-init a layer, add a
hook, rescale the residual stream. A crash there is a clean revert — experiment.

## Good practice

- One variable per experiment; write a real `DESCRIPTION`.
- Prefer cheap, high-signal knobs before expensive capacity bumps.
- If a change ties (no improvement), revert and try a different axis — don't
  stack neutral changes.
- Read the discards too: a regression is a result (e.g. "deeper hurt at this
  token budget" is real signal about under-training).

## Budget & reproducibility

Default budget is **token-based** (`--tokens`, default 2M), not wall-clock, so a
5060 Ti and an H100 land on the *same* `val_bpb` curve — results are comparable
across machines. `--minutes` switches to a wall-clock budget if you want the
autoresearch-style "how far can you get in N minutes" framing. Everything is
seeded; the same `candidate.py` gives the same number.
