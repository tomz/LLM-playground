# Improvements log — Q1 2026

A 4-tier overhaul of distgpt that took test coverage from ~11 to 83 tests,
fixed 3 latent distributed bugs, and added Muon / FP8 / sequence-parallel /
HF-export / recipe-based training. This file documents what changed and
why, organized by tier in the order the commits landed.

## Tier 1 — Distributed tests + observability (commit c29e132)

Added `tests/test_distributed_smoke.py` with 2-rank gloo CPU tests for
FSDP-DP, TP, PP, and end-to-end trainer. Writing the TP test caught **3
real latent bugs** that were hiding under our single-process test suite:

1. **`GQAttention.forward` shape bug under tp>1.** The `view()` call was
   hardcoded to `self.n_head`, but each rank only owns `n_head // tp`
   local heads after column-sharding q/k/v. Fix: derive the local head
   count from `q_proj.out_features // self.head_dim`.

2. **`build_pipeline` skipped wrapping in `PipelineStage`.** It passed
   the raw model to `Schedule1F1B`, which expects a `PipelineStage`
   instance. Symptom: `AttributeError: 'GPT' object has no attribute
   'num_stages'` on any pp>1 run. Fix: wrap as
   `PipelineStage(stage_module, stage_index, num_stages, device)`.

3. **`ckpt.save` final-save deadlock.** It was gated on `is_master()`,
   but `dcp.save` is a collective across all ranks — only-master called
   it would deadlock everyone else. Fix: all ranks call save; only the
   per-rank meta JSON write is gated.

Other Tier 1 deliverables:
* `distgpt/utils/metrics.py` — `grad_norm`, `param_norm`, `MFU` against
  per-device peak FLOPs (Hopper sm_90, Blackwell sm_100/120, Ampere).
* `distgpt/model/parallel_layers.py` — explicit
  `ColumnParallelLinear`/`RowParallelLinear`/`VocabParallelEmbedding` (the
  fallback path the README referenced).
* `distgpt/data/mixture.py` — weighted multi-source sampler.
* Trainer now logs grad_norm / param_norm / tok_per_s_per_gpu / MFU
  every step, has a `train.compile: bool` knob, and runs real PP eval
  via `Schedule1F1B(n_microbatches=1)` instead of returning NaN.

## Tier 2 — Modern training tricks

### 2.6 Muon optimizer (commit 94d4c81)

`distgpt/training/muon.py` ships:
* `newton_schulz5(G, steps=5)` — Keller Jordan's tuned quintic
  iteration (a, b, c) = (3.4445, -4.7750, 2.0315). Runs in bf16 on
  CUDA, fp32 on CPU.
* `Muon` optimizer with FSDP2-safe distributed step: `p.full_tensor()`
  all-gathers the DTensor shard, N-S runs on the global matrix, the
  update is then `distribute_tensor`-ed back to the same placement.
  Per-rank: identical work, no compute saving, but composes cleanly
  with FSDP2.
* `split_muon_params(model)` — partitions named params into
  (2D hidden weights → Muon, embeddings/lm_head/MTP heads/routing
  gates/1-D → AdamW). The split is by name (IO markers) + ndim.
* `build_muon_and_adamw(model, ...)` — convenience builder returning
  both optimizers on one shared cosine schedule.

Trainer integration:
* `optim.optimizer: muon|adamw` in YAML chooses the path.
* The cosine schedule preserves the per-optimizer LR ratio (Muon's LR
  is typically 50–100× AdamW's; scaling both by the same cosine factor
  keeps the ratio constant through warmup and decay).
* `CheckpointManager.save`/`load` made polymorphic over single-or-list
  optimizers; legacy single-optim checkpoints still round-trip via the
  `"optim"` key, multi-optim uses `"optim_0"`, `"optim_1"`, ….

12 tests (`tests/test_muon.py`) covering algorithm, scope, end-to-end,
and a 2-rank gloo DTensor smoke.

### 2.7 Fused linear-cross-entropy (commit 92f37b4)

`model.fused_ce: bool` knob (default-off) routes the training loss
through `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`,
fusing the lm_head matmul + CE into one Triton kernel so the
`[B*T, vocab]` logits tensor is never materialized — the single largest
activation in the forward pass.

Numerics measured in midgpt on a 350M GPT-2: matches dense CE to
~1.8e-3 absolute, grads to ~8e-3 relative (bf16 rounding inside the
kernel).

Throughput is HW-dependent. On the RTX 5060 Ti (Blackwell sm_120, torch
2.11) midgpt measured the fused path ~26% SLOWER than dense + CE
because the matmul kernel is already well-tuned. So we treat fused-CE
as a **VRAM-headroom lever** (fit bigger batch / vocab / model), not a
speedup. Peak VRAM dropped from 12.8 GiB to 10.1 GiB (-20%) at 350M.

Contract: when `fused_ce=True` AND `targets is not None`, the forward
returns `(None, loss)` — callers needing logits must set
`fused_ce=False`. Both `trainer.py` and `eval/harness.py` already
discarded logits, so no caller changes were needed. The inference path
(`targets=None`) is unchanged: still returns last-token logits.

5 tests (`tests/test_fused_ce.py`), including a GPU numerics test that
ran on the local 5060 Ti and matched dense within 5e-3 relative.

### 2.8 Sequence parallelism (commit 017282c)

`parallel.sequence_parallel: bool` adds Megatron-style SP to the TP path.
When on:
* `attn_norm` and `ffn_norm` are wrapped with `SequenceParallel`,
  making each norm a 1/tp-local op over a `Shard(seq)` slice.
* o_proj and ffn.w2 use `output_layouts=Shard(1)` so the residual stays
  sequence-sharded into the next block's SP-wrapped norm.
* tok_emb emits `Shard(1)` directly into the first SP norm; lm_head
  consumes `Shard(1)` from the final replicated norm and gathers logits
  for CE loss.

SP is a memory + compute win when T >> d/tp (rule of thumb: enable when
`seq_len >= 4096`). Below that, the extra colwise gather right before
attention/FFN matmuls swamps the savings.

Marker attribute is `_dgpt_sp_enabled` (short prefix; the earlier
hypothesis about `torch.distributed.checkpoint` monkey-patching `_dist*`
attribute names turned out NOT to apply in torch 2.11 — see
`tests/test_pytorch_quirks.py`).

4 tests (`tests/test_sequence_parallel.py`) including a 2-rank gloo
TP+SP smoke that pins the per-rank loss agreement (an O(1) divergence
would fire here if the SP gather/scatter were miswired).

### 2.9 FP8 / Transformer Engine path (commit e2ba931)

`distgpt/training/precision.py` ships three primitives:
* `device_supports_fp8(device)` — capability table (sm_90 Hopper,
  sm_100 Blackwell-DC, sm_120 Blackwell-consumer).
* `resolve_fp8_recipe(setting, device, dtype)` — validates the
  `train.fp8: off|e4m3|hybrid` knob. Raises on typos. Warns + falls
  back to None on non-bf16 dtype or non-FP8 HW so a misconfigured
  launch keeps training in bf16 instead of dying.
* `autocast_fp8_context(recipe)` — returns `te.fp8_autocast(...)` (lazy
  import of transformer_engine) or `nullcontext` for recipe=None.
  Missing TE + real recipe → clear ImportError (no silent fallback —
  silent fallback would hide the throughput win the user expected).

Trainer integration: the existing `autocast` value-context is now a
function returning a stacked `fp8_autocast() ∘ bf16_autocast()`. fp8
is OFF by default, keeping the dense bf16 path bit-for-bit identical.

The win is gated on (a) installing TE and (b) swapping the model's
`nn.Linear` layers for `te.Linear` — out of scope here. Landing the
config plumbing now means downstream users who do the swap can flip
`train.fp8: hybrid` without patching the trainer.

12 tests (`tests/test_precision.py`).

## Tier 3 — Eval + recipes

### 3.10 HuggingFace export + lm-evaluation-harness (commit 253b4cc)

`distgpt/eval/export_hf.py`:
* `export_to_hf(model, cfg, out_dir)` writes config.json,
  model.safetensors (or pytorch_model.bin), and generation_config.json
  in the layout `transformers.LlamaForCausalLM` expects.
* `load_hf_state_into_distgpt(model, hf_dir)` is the inverse for
  continued pre-training of a published HF base model.
* Architecture mapping documented inline (distgpt is structurally
  Llama; the rename table is mechanical).
* Tied-embedding wart: safetensors refuses to write shared storage
  twice (`tok_emb.weight is lm_head.weight` under
  `tie_embeddings=True`), so we drop the lm_head duplicate and rely on
  HF's `tie_word_embeddings=True` config flag to recreate the tie on
  load.
* `qk_norm=True` rejected at export with a clear error rather than
  silently stripped — stock LlamaForCausalLM has no QK-norm layer.

`distgpt/eval/lm_eval_runner.py`:
* `run_lm_eval(...)` exports to a temp HF dir, instantiates
  `lm_eval.models.huggingface.HFLM`, runs `simple_evaluate(tasks=...)`,
  prints results, optionally writes them to JSON.
* Tokenizer-files resolution falls back to GPT-2 BPE via transformers
  when the user doesn't supply their own.
* Lazy imports of `lm_eval` and `transformers` — distgpt's base deps
  stay torch-only.

CLI: `distgpt eval --config X --ckpt Y --lm-eval-tasks hellaswag,arc_easy`
switches from in-cluster held-out-loss to the HF + lm-eval path. New
flags: `--tokenizer-dir`, `--num-fewshot`, `--limit`, `--batch-size`,
`--output-path`.

10 tests (`tests/test_eval_export.py`) including a real
`LlamaForCausalLM.from_pretrained` round-trip when transformers is
installed.

### 3.11 Recipe configs + warm-start (commit d0e19b0)

Three production recipes under `configs/recipes/`:
* `cooldown.yaml` — post-pretrain LR-decay anneal (5–10% of base steps
  at ~0.05×min_lr on high-quality data). The
  DeepSeek-V3/Llama-3/Qwen-2 pattern; ~1–2 pp on benchmarks for ~5%
  extra compute.
* `longctx_finetune.yaml` — 4K → 32K context extension via rope_base
  scaling (10000 → 500000) + `sequence_parallel: true`. Fresh optim,
  fine-tune LR (5e-5). Memory-aware: full activation ckpt,
  micro_batch=1, grad_accum=32.
* `muon_speedrun_1b.yaml` — 1B with Muon + qk_norm + zero_init_proj.
  Highest-sample-efficiency configuration we ship.
* `README.md` documenting the `load_ckpt:` vs native-resume distinction.

`CheckpointManager.load_weights_only(model, ckpt_dir)`:
* Loads ONLY model weights from an arbitrary step dir; no optim, no
  loader cursor, no step counter.
* Validates `ckpt_dir` up-front (raises FileNotFoundError with a
  helpful message) — DCP's own missing-path error is a `BaseException`
  subclass with a confusing "metadata is None" message.

Trainer resume / warm-start priority order:
1. **Native resume** — if `out_dir/run_id/ckpts/` has a step dir, pick
   up where we left off.
2. **Warm start** — `load_ckpt: <path>` loads weights only, starts at
   step 0 with fresh optim/loader/schedule.
3. **Cold start** — brand-new model, step 0.

Native resume takes priority over `load_ckpt:` — the safety property
that lets you put `load_ckpt:` in a recipe and re-run after an
interruption without overwriting your cooldown progress.

9 tests (`tests/test_recipes.py`).

## Tier 4 — Pinned regression tests + this doc

### 4.12 PyTorch quirks (this commit)

`tests/test_pytorch_quirks.py` pins three upstream behaviours we work
around:

1. **`torch.distributed.checkpoint.api.CheckpointException` is a
   `BaseException` subclass, not `Exception`.** So
   `pytest.raises(Exception)` and `try/except Exception` don't catch
   it. Worked around in `CheckpointManager.load_weights_only` by
   validating the path up-front and raising `FileNotFoundError`.

2. **safetensors refuses to write shared storage.** Worked around in
   HF export by dropping the `lm_head.weight` duplicate when tied.

3. **`nn.Module.__getattr__` still raises AttributeError for any unset
   attribute.** (Sanity check; an earlier hypothesis about a
   `_dist*`-prefix monkey-patch turned out to be incorrect for torch
   2.11+; this test would fail if such a monkey-patch ever returned.)

When PyTorch fixes any of these, the corresponding test fails, we get
notified, we can delete the workaround.

---

## Final test count

```
83 passed, 2 skipped, 26 warnings in ~24s
```

The 2 skips:
* `test_pp2_two_stage_pipeline` — gloo doesn't support all the
  pipelining collectives.
* `test_fused_ce_true_raises_importerror_when_liger_missing` — skipped
  because liger-kernel IS installed in this venv (the test only runs
  in environments without it).

## Real bugs caught by writing tests

| # | Where | Symptom | Fix |
|---|---|---|---|
| 1 | `GQAttention.forward` | shape-asserts under tp>1 | derive local head count from `q_proj.out_features` |
| 2 | `build_pipeline` | `AttributeError: num_stages` under pp>1 | wrap stage in `PipelineStage(...)` |
| 3 | `ckpt.save` | deadlock at final save | all ranks call dcp.save (collective) |

All three would have surfaced for any user running on >1 GPU. None
were caught by the original single-process test suite.
