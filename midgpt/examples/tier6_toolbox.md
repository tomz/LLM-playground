# Tier 6 toolbox: bug fixes, eval, and the Llama-style flips

This is a *narrative* document covering what landed in midgpt's Tier 6
refresh (commits `482f51f` … `ed97459` + the MTP commit). The headline run
[`5060ti_350m_fineweb.md`](./5060ti_350m_fineweb.md) is still the right
demo of the trainer itself; this doc walks through what the toolbox around
it does and when to use each piece.

If you only read one section: the **bug fixes** in Tier 6.1 affect every
existing midgpt run (they're not opt-in — they're plain corrections to the
sampler bias and the DDP eval-loss aggregation). Everything else is
default-off and only activates if you flip a YAML flag.

## What's new (in commit order)

| Tier | Subject | Tests | Surface area |
|---|---|---|---|
| 6.1 | Bug fixes + DDP test suite | +15 | `data.py`, `train.py`, `sample.py`, `model.py` (generate) |
| 6.2 | Spike-rewind + HF export + lm-eval CLI | +27 | `stability.py`, `export_hf.py`, `lm_eval_runner.py` |
| 6.3 | Opt-in Llama-style architecture flags | +17 | `model.py` (RoPE/RMSNorm/SwiGLU) |
| 6.4 | Multi-Token Prediction + docs | +9 | `model.py` (MTP heads), `README.md`, this doc |

Test count: **11 → 79** (and the baseline 11 still pass on every tier).

---

## 6.1 — Bugs you can stop hitting

Three latent bugs fixed; each had a real-world failure mode.

### 1. Shard-sampler bias

`ShardDataset._locate` used to fall back to `off = 0` whenever the random
landing index couldn't fit a full block at the end of a shard. That
deterministically biased sampling toward each shard's first `block_size+1`
tokens. Tiny per-batch effect, but it added up: on a `block_size=1024`
shard of 1M tokens, ~0.1% of every batch was the same 1025-token prefix.
Over a 4000-step run that's ~50× the natural sample rate for those tokens.

**Fix:** return `None` and re-sample. Pinned by
`tests/test_data.py::test_sampling_distribution_unbiased`, which would
have shown a ~14% spike on the old behaviour.

### 2. DDP eval-loss measured only rank-0's slice of val

The old `evaluate()` ran inside `if is_master:` and used a `val_ds` that
was nominally constructed with `rank=rank, world_size=world` — so on a
4-GPU run, the reported val ppl was measured on **1/4** of the val set.
The actual val ppl was almost certainly different (sampling noise reduces
as N rises) and worse if any rank's slice was harder.

**Fix:** `evaluate()` now runs on **every** rank with disjoint shuffles,
and the per-rank means are all-reduced before logging. Reported val now
covers `world * eval_iters` distinct batches.

Pinned by `tests/test_distributed_smoke.py::test_eval_allreduces_across_ranks`
(2-rank gloo CPU).

### 3. CUDA-resume crashed on the RNG state

`torch.load(ckpt_path, map_location=device, ...)` on CUDA silently promoted
the saved CPU `ByteTensor` RNG states to CUDA tensors. `torch.set_rng_state`
then crashed with `TypeError: RNG state must be a torch.ByteTensor`. **Any
real CUDA resume would have hit this.** The 11 baseline tests didn't because
they ran on CPU.

**Fix:** load to CPU first, restore the RNG ByteTensors explicitly.

### + smaller goodies

- `sample.py` gains `--temperature 0` (greedy / argmax) and `--top-p` (nucleus),
  plus an optional `--seed` for reproducible non-greedy samples.
- The dead `rank=`/`world_size=` kwargs on `ShardDataset` are gone.
- `setup_ddp()` picks the backend by env (`MIDGPT_BACKEND`, default `nccl`
  on CUDA / `gloo` on CPU) so the DDP loop runs under CPU CI.

---

## 6.2 — The production toolbox

### Spike-rewind (`stability.py`)

```yaml
# Opt-in: drop this into a config to enable.
stability:
  spike_monitor: true
  spike_window: 200          # rolling window for mean/std
  spike_sigma: 5.0           # z-score threshold
  spike_min_abs_jump: 2.0    # absolute-loss threshold (BOTH must trigger)
  max_rewinds: 5             # cap — chronically spiky runs still finish
  rewind_cooldown_steps: 1000
```

The two-threshold detector (z-score AND absolute jump) is the load-bearing
trick that prevents the runaway rewind loop documented in distgpt's
`stability.py` — on a converged loss curve a single 0.3-loss blip is ~7σ but
absolutely tiny, and without the absolute floor the trainer would rewind →
train back to the same plateau → rewind → … indefinitely. `max_rewinds`
caps total rewinds so even if both thresholds keep firing the run finishes.

The 5060 Ti 354M FineWeb-Edu run didn't spike (the loss curve in
[`5060ti_350m_fineweb.md`](./5060ti_350m_fineweb.md) is textbook), but
this is the right knob for longer runs / harder data where one spike could
otherwise burn hours of compute.

### HuggingFace export (`export_hf.py`)

```bash
python export_hf.py \
    --ckpt out/gpt2_350m_fweb_5060ti/ckpt_best.pt \
    --out-dir out/hf_export \
    --verify
```

Bidirectional weight conversion between midgpt's flat `nn.Linear` layout and
HF's `GPT2LMHeadModel` (which uses `Conv1D` — `nn.Linear` *with the weight
transposed*). The `--verify` flag round-trips through
`GPT2LMHeadModel.from_pretrained` and asserts logit-agreement to `<1e-3` in
fp32. The round-trip catches one-axis transpose bugs that shapes alone would
not (`(d, d)` swapped is still `(d, d)`).

**Caveats:**
- `qk_norm=True` raises (stock GPT-2 has no QK-norm layer).
- The new Llama-style flags (RoPE/RMSNorm/SwiGLU) raise too — target is
  `GPT2LMHeadModel`; a `LlamaForCausalLM` export is its own future job.
- MTP heads are dropped silently (train-only auxiliary; the message
  `[export_to_hf] dropped N MTP-head key(s)` prints to stderr).

### lm-evaluation-harness driver (`lm_eval_runner.py`)

```bash
pip install lm-eval                                           # optional dep
python lm_eval_runner.py \
    --ckpt out/best.pt \
    --tasks hellaswag,lambada_openai,arc_easy \
    --device cuda --output results.json
```

Wraps `export_hf.py` and hands the resulting HF dir to
`lm_eval.simple_evaluate`. Lazy import so lm-eval stays an optional dep.
Use for side-by-side numbers against published GPT-2 / Pythia / GPT-J on
the canonical task list; for mid-training spot checks the existing
`eval.py` HellaSwag harness is still faster (no HF round-trip).

---

## 6.3 — Llama-flavored architecture flags

Three orthogonal config knobs flip parts of the GPT-2 model toward Llama.
All default to GPT-2; the existing 354M FineWeb-Edu run is bit-identical
unless you flip them.

| Flag | GPT-2 default | Llama-style | What it changes |
|---|---|---|---|
| `pos_kind`  | `learned`   | `rope`     | Drops `pos_emb` (learned position table); applies RoPE to Q/K per head before SDPA |
| `norm_kind` | `layernorm` | `rmsnorm`  | Replaces all block + final norms with weight-only RMSNorm |
| `mlp_kind`  | `gelu`      | `swiglu`   | Replaces `proj(gelu(fc(x)))` with gated `proj(silu(w1 x) * w3 x)` |

The recipe [`configs/gpt2_350m_llamafied_fweb_5060ti.yaml`](../configs/gpt2_350m_llamafied_fweb_5060ti.yaml)
flips all three on. To keep it iso-param with the GPT-2 baseline, `d_ffn`
is reduced from 4096 to 2730 (= `8/3 * d_model`, the Llama heuristic) so
SwiGLU's 3-matrix FFN matches the GELU 2-matrix FFN at the same total
parameter count.

The intent is a head-to-head ablation on identical data + step budget — same
loop, same loader, same eval, only the architecture differs.

---

## 6.4 — Multi-Token Prediction

```yaml
model:
  mtp_tokens: 4       # extra heads predicting tokens at +2, +3, +4, +5
  mtp_weight: 0.3     # scalar on the averaged auxiliary CE
```

Default `mtp_tokens: 0` (off). When on:

- N extra `nn.Linear(d_model, vocab_size)` heads sit alongside `lm_head`.
- During *training*, head `j` predicts the token at offset `(j+2)` from the
  same final hidden state. The averaged-per-head CE is scaled by
  `mtp_weight` and added to the main loss.
- At *inference* the heads never fire (`if self.mtp_heads and
  self.training:`); eval val ppl is pure next-token CE.
- The HF export silently strips them (GPT-2 has no MTP).

DeepSeek-V3 reports ~5-10% sample-efficiency from this on their ablation.
A free perf bump that doubles as a draft-token source for future
speculative-decoding work.

---

## Putting it together

A "frontier-style" midgpt config combines everything orthogonally:

```yaml
optim:
  optimizer: muon         # Muon for 2D hidden weights
  ...
model:
  pos_kind: rope          # Llama position encoding
  norm_kind: rmsnorm      # Llama norm
  mlp_kind: swiglu        # Llama FFN
  qk_norm: true           # cheap attention-logit stabilizer
  mtp_tokens: 4           # train-only auxiliary heads
fused_ce: true            # Liger fused linear-CE (Triton GPU only)
stability:
  spike_monitor: true     # rewind on a real spike
```

Each flag is config-gated and exhaustively tested in isolation. Use the
recipe [`gpt2_350m_llamafied_fweb_5060ti.yaml`](../configs/gpt2_350m_llamafied_fweb_5060ti.yaml)
as the starting point; flip flags one at a time to see what each one buys
you on your data.
