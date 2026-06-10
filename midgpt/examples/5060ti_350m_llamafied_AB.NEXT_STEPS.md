# Track B1 — llamafied-350M A/B — NEXT STEPS (pickup doc)

> **Status: ready to launch. One decision pending (harness rigor, §4).**
> This is a *pickup* doc, not a results writeup. Everything below was
> verified on the machine on the session this was written; paths, param
> counts, and the A-side baseline numbers are real. Resume here.

## 0. TL;DR

We want a clean **architecture A/B** at 350M scale on FineWeb-Edu:

| Arm | Pos enc | Norm | MLP | QK-norm | Config |
|-----|---------|------|-----|:-------:|--------|
| **A** (baseline, GPT-2) | learned table | LayerNorm | GELU (d_ffn 4096) | ✗ | `configs/gpt2_350m_fweb_5060ti.yaml` |
| **B** (llamafied) | RoPE (θ=10k) | RMSNorm | SwiGLU (d_ffn 2730) | ✓ | `configs/gpt2_350m_llamafied_fweb_5060ti.yaml` |

The configs are **iso-param** and **iso-token** by construction (verified
below). A-side is **already trained**; B-side has **never been run**.
The only open question is how rigorous to make the comparison harness
(single-GPU baseline-as-is vs re-running both arms under identical DDP).

## 1. What's already done ✅

**A-side baseline is complete and verified** (`out/gpt2_350m_fweb_5060ti/`):

| Metric | Value |
|---|---|
| Model | GPT-2 354.60M (24L × 1024d × 16H, GELU d_ffn 4096, tied emb) |
| Tokens/step | 32 768 (micro_batch 4 × grad_accum 8 × world 1 × block 1024) |
| Iters | 4 000 → 131M tokens |
| Train loss | 11.00 → **3.97** |
| **Best val** | **loss 4.0641 / ppl 58.2** @ iter 3800 |
| Wall-clock | **2.57 h** (median 2204 ms/it) |
| Throughput | **14.8k tok/s** (single 5060 Ti) |
| Evals | 19 (cadence 200) |
| Artifacts | `log.jsonl`, `loss.png`, `ckpt_best.pt` |

Numbers extracted from `out/gpt2_350m_fweb_5060ti/log.jsonl`. Eval rows use
key `eval_val` (NOT `val_loss`); train rows have keys
`{iter, loss, lr, ms, tok_per_s, wall}`.

## 2. Verified preconditions ✅

Checked live this session — all green:

- **GPUs**: 2× RTX 5060 Ti 16 GB, both idle (13 MiB used).
- **venv + torch**: `.venv/bin/python` works; `import torch` OK (the
  system `python3` has **no torch** — always use `.venv/bin/python`).
- **Data**: `data/fineweb-edu/{train,val}/` populated (train = many
  128 MB shards, val = 1 shard). Same corpus as A-side.
- **model.py supports every B-side switch**: `pos_kind=rope`,
  `norm_kind=rmsnorm`, `mlp_kind=swiglu`, `qk_norm=true`, `rope_base`
  (see `GPTConfig` + `_apply_rope` / `RMSNorm` / `SwiGLU` / `make_mlp`).
- **iso-param confirmed** (built both models, counted params):
  - A: **354.60M**
  - B: **353.51M** → **0.31% apart** ✓ (SwiGLU's 3 matmuls at d_ffn=2730
    ≈ GELU's 2 matmuls at d_ffn=4096).
- **train.py DDP**: full support via `torchrun`. Effective batch =
  `micro_batch × grad_accum × world × block` (train.py line ~223). DDP
  eval averages across `world × eval_iters` distinct batches.
- **train.py has NO CLI hyperparameter overrides** — only `--config` and
  `--resume`. So changing grad_accum for iso-token requires a *new config
  file* (can't override on the command line).

## 3. The iso-token gotcha (read before launching) ⚠️

A-side ran **single-GPU**: `4 × 8 × 1 × 1024 = 32 768` tok/step.

If you launch the **existing** B-side config on **2 GPUs**, you get
`4 × 8 × 2 × 1024 = 65 536` tok/step — **2× the tokens per step**, i.e.
262M tokens over 4000 iters instead of 131M. That confounds the
architecture comparison with a token-budget difference.

**To keep iso-token on 2 GPUs, halve grad_accum to 4** →
`4 × 4 × 2 × 1024 = 32 768` tok/step ✓ (same global batch, same 131M
tokens, same 4000-iter LR schedule). This is the config in §5.

## 4. THE OPEN DECISION (user deferred — pick one) 🔀

> User said: *"write this up in a next steps doc and we will pick up in
> another session."* So this is unmade. The three options, verbatim:

### Option 1 — Fast: B-side only on 2 GPUs (~1.3h)
Run B-side on 2 GPUs (iso-token config from §5), compare to the
**existing single-GPU A-side**. One run, ~1.3 h.
*Confound*: A-side is single-GPU, B-side is 2-GPU → data-sharding order
differs. Architecture is still the dominant variable; note the caveat.

### Option 2 — Rigorous: re-run BOTH arms on 2 GPUs (~2.6h) ⭐ recommended
Re-run A-side *and* B-side under the identical 2-GPU DDP harness
(both with grad_accum=4). Confound-free: same harness, same data
ordering, only architecture differs. Bonus: the existing single-GPU
A-side becomes a 3rd data point that doubles as a **DDP-reproducibility
check** (single-GPU 58.2 ppl vs 2-GPU A-side should match within noise).

### Option 3 — Mirror exactly: B-side single-GPU like A-side (~2.5h)
Run B-side single-GPU with the **existing** config unchanged
(`CUDA_VISIBLE_DEVICES=0`). Tightest arch-only iso (identical harness to
A-side), but **does NOT exercise DDP** — so it doesn't satisfy the
"via DDP" part of the Track B1 brief. Only pick if DDP is not a goal.

**Recommendation: Option 2.** Costs ~1.3h extra, removes the only
confound, and gives a free DDP-determinism check. If GPU time is tight,
Option 1 is a defensible study-grade A/B.

## 5. Config to create (needed for Options 1 & 2)

Both 2-GPU options need an **iso-token** B-side config. Create
`configs/gpt2_350m_llamafied_fweb_5060ti_2gpu.yaml` — it's the existing
llamafied config with `out_dir` bumped and **grad_accum 8 → 4**:

```yaml
# 350M llamafied A/B — 2-GPU iso-token variant of
# gpt2_350m_llamafied_fweb_5060ti.yaml. grad_accum halved 8->4 so that
# 4(mb) x 4(ga) x 2(world) x 1024(block) = 32768 tok/step matches the
# single-GPU GPT-2 baseline exactly. Same 4000-iter schedule => 131M tokens.
out_dir: out/gpt2_350m_llamafied_fweb_5060ti_2gpu
dataset: fineweb-edu
tokenizer: gpt2
seed: 1337
dtype: bfloat16
compile: false
grad_checkpoint: false
log: {jsonl: true, wandb_project: null}

model:
  vocab_size: 50304
  block_size: 1024
  n_layer: 24
  n_head: 16
  d_model: 1024
  d_ffn: 2730              # 8/3 * 1024 — iso-param with d_ffn=4096 GELU
  dropout: 0.0
  bias: false
  tie_embeddings: true
  pos_kind: rope
  norm_kind: rmsnorm
  mlp_kind: swiglu
  rope_base: 10000.0
  qk_norm: true

optim:
  optimizer: adamw
  lr: 3.0e-4
  min_lr: 3.0e-5
  betas: [0.9, 0.95]
  weight_decay: 0.1
  grad_clip: 1.0
  warmup_iters: 200
  lr_decay_iters: 4000
  max_iters: 4000

train:
  micro_batch: 4
  grad_accum: 4            # 4*4*2*1024 = 32768 tok/step (iso with baseline)
  eval_interval: 200
  eval_iters: 50
  log_interval: 10
  ckpt_interval: 500
```

For **Option 2** you also need a 2-GPU iso-token *A-side* config — copy
`gpt2_350m_fweb_5060ti.yaml` to `..._2gpu.yaml`, set
`out_dir: out/gpt2_350m_fweb_5060ti_2gpu`, and `grad_accum: 4`.

## 6. Launch commands

All commands from `midgpt/`. Use `.venv/bin/python` (system python has no
torch). Tee to a `_train.log` so the plotter has step-time lines.

```bash
# --- Option 1 (B-side only, 2 GPUs) ---
torchrun --standalone --nproc_per_node 2 train.py \
    --config configs/gpt2_350m_llamafied_fweb_5060ti_2gpu.yaml \
    2>&1 | tee out/gpt2_350m_llamafied_fweb_5060ti_2gpu_train.log

# --- Option 2 (both arms, 2 GPUs) — run sequentially ---
torchrun --standalone --nproc_per_node 2 train.py \
    --config configs/gpt2_350m_fweb_5060ti_2gpu.yaml \
    2>&1 | tee out/gpt2_350m_fweb_5060ti_2gpu_train.log
torchrun --standalone --nproc_per_node 2 train.py \
    --config configs/gpt2_350m_llamafied_fweb_5060ti_2gpu.yaml \
    2>&1 | tee out/gpt2_350m_llamafied_fweb_5060ti_2gpu_train.log

# --- Option 3 (B-side single-GPU, existing config) ---
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
    --config configs/gpt2_350m_llamafied_fweb_5060ti.yaml \
    2>&1 | tee out/gpt2_350m_llamafied_fweb_5060ti_train.log
```

torchrun invokes `.venv/bin/python` automatically if it's the active venv;
if not, use `.venv/bin/torchrun`. Each arm is ~1.3 h (2-GPU) / ~2.5 h
(single-GPU). Watch with `watch(target="pid:<torchrun_pid>")` or tail the
`_train.log`.

## 7. Plot the A/B

`tools/plot_midgpt_compare.py` already does exactly this (it was built for
the fused-CE comparison). Signature: `--run JSONL LABEL`, `--base JSONL
LABEL`, `--out`, `--hardware`, `--dataset`, `--note`, `--title`.

```bash
# Option 1/3 (B-side run vs existing single-GPU baseline):
.venv/bin/python tools/plot_midgpt_compare.py \
    --run  out/gpt2_350m_llamafied_fweb_5060ti_2gpu/log.jsonl "llamafied (RoPE+RMSNorm+SwiGLU+QKnorm)" \
    --base out/gpt2_350m_fweb_5060ti/log.jsonl "GPT-2 (learned-pos+LN+GELU)" \
    --out  out/gpt2_350m_llamafied_fweb_5060ti_2gpu/compare_llamafied.png \
    --hardware "2× RTX 5060 Ti 16 GB (Blackwell sm_120, bf16)" \
    --dataset  "FineWeb-Edu (1B-token slice), iso-param 354.6M vs 353.5M, iso-token 32768/step" \
    --title    "midgpt · llamafied vs GPT-2 (350M, iso-param, iso-token)"

# Option 2: point --base at out/gpt2_350m_fweb_5060ti_2gpu/log.jsonl instead,
# and mention the single-GPU 58.2-ppl point as a DDP-reproducibility check.
```

## 8. Writeup checklist (after the run)

- [ ] Create the config(s) from §5.
- [ ] Run the chosen option (§6); confirm `world_size=2` printed and both
      GPUs ~100% in `nvidia-smi`.
- [ ] Extract B-side best val (`eval_val` min) + wall + tok/s the same way
      §1 did for A-side.
- [ ] Generate the compare plot (§7).
- [ ] Write `examples/5060ti_350m_llamafied_AB.md` (drop the
      `.NEXT_STEPS` doc) — A vs B table, the plot, a few sampled
      completions from each (`sample.py --ckpt .../ckpt_best.pt`), and the
      verdict (does the Llama recipe actually win at 350M / 131M tokens?).
- [ ] Add a "Llamafied A/B" row/section to `midgpt/README.md` and a bullet
      to the top-level `README.md` results gallery.
- [ ] Mark task **#3 (Track B1)** completed.

## 9. Key facts & paths (so you don't re-derive)

| Thing | Value |
|---|---|
| A-side config | `configs/gpt2_350m_fweb_5060ti.yaml` |
| A-side output | `out/gpt2_350m_fweb_5060ti/{log.jsonl,loss.png,ckpt_best.pt}` |
| A-side best val | **ppl 58.2 / loss 4.0641 @ iter 3800**, 2.57h, 14.8k tok/s |
| B-side config (single-GPU, exists) | `configs/gpt2_350m_llamafied_fweb_5060ti.yaml` |
| B-side config (2-GPU iso-token, TO CREATE) | `configs/gpt2_350m_llamafied_fweb_5060ti_2gpu.yaml` (§5) |
| Param counts | A 354.60M · B 353.51M (0.31% apart) |
| Iso-token target | 32 768 tok/step × 4000 iters = 131M tokens |
| 2-GPU grad_accum | **4** (not 8) — see §3 |
| Eval JSONL key | `eval_val` (min = best val) |
| Plotter | `tools/plot_midgpt_compare.py` (`--run/--base/--out/...`) |
| Python | `.venv/bin/python` (system python lacks torch) |
| DDP launch | `torchrun --standalone --nproc_per_node 2 train.py --config ...` |
| Task id | **#3 Track B1** (NOTE: tracker's "Track C" = coder-finetune, different work) |
