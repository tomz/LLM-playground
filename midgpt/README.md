# midgpt — Mid-scale single-node GPT-2 trainer

GPT-2 scale (124M–1.5B). Real BPE tokenizer (`tiktoken`), WikiText-103 or OpenWebText, mixed precision, gradient checkpointing, gradient accumulation, cosine LR, DDP across 1–8 GPUs on a single node.

## What's different from `nanogpt-edu`

| Feature                       | nanogpt-edu | midgpt |
|-------------------------------|:----------:|:------:|
| Tokenizer                     | char-level | tiktoken (GPT-2 BPE, 50257) |
| Dataset                       | TinyShakespeare | WikiText-103 / OpenWebText |
| Mixed precision               | autocast   | autocast + GradScaler |
| Gradient checkpointing        | —          | ✅ (per-block) |
| Gradient accumulation         | basic      | DDP-aware (no_sync) |
| Multi-GPU                     | —          | torchrun + DDP |
| Eval                          | held-out loss | loss + perplexity + HellaSwag (zero-shot) |
| FlashAttention                | SDPA picks it | SDPA picks it; checkpoint-friendly |
| Attention backend             | SDPA / opt-in FlexAttention | SDPA / opt-in FlexAttention (`attention_backend`) |
| Logging                       | print      | print + JSON lines + optional W&B |
| Resume                        | last ckpt  | last ckpt with full RNG/optim/loader state |
| Optimizer                     | AdamW / Muon | AdamW / Muon (orthogonalized 2D-weight updates) |
| QK-Norm stabilizer            | ✅ opt-in   | ✅ opt-in (`model.qk_norm`) |
| Zero-init residual projections | ✅ opt-in  | ✅ opt-in (`model.zero_init_proj`) |
| Fused linear-CE (Liger)       | —          | ✅ opt-in (`fused_ce`, GPU+Triton) |
| Loss-spike rewind             | —          | ✅ opt-in (`stability.spike_monitor`) |
| HuggingFace export            | —          | ✅ (`export_hf.py` → `GPT2LMHeadModel`) |
| `lm-evaluation-harness`       | —          | ✅ (`lm_eval_runner.py`) |
| Llama-style arch knobs        | RoPE/RMSNorm/SwiGLU fixed | ✅ opt-in (`pos_kind`/`norm_kind`/`mlp_kind`) |
| Multi-Token Prediction        | ✅ opt-in   | ✅ opt-in (`model.mtp_tokens`) |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Tokenize a dataset (writes shards to data/<name>/)
python prepare.py --dataset wikitext103             # ~500 MB tokens, ~10 min
# or — stream a 1B-token slice of FineWeb-Edu (~2 GB on disk, ~9 min)
python prepare.py --dataset fineweb-edu --streaming --max-tokens 1000000000
# or — full OpenWebText (~9B tokens, hours)
python prepare.py --dataset openwebtext --num-proc 16

# 2. Train (single GPU)
python train.py --config configs/gpt2_124m.yaml

# 3. Train (8 GPUs on one node)
torchrun --standalone --nproc_per_node 8 train.py --config configs/gpt2_350m.yaml

# 4. Evaluate
python eval.py --ckpt out/gpt2_124m/ckpt.pt --tasks ppl,hellaswag

# 5. Generate
python sample.py --ckpt out/gpt2_124m/ckpt.pt --prompt "Once upon a time"
```

## Configs

| File                          | Params | Layers | d_model | Tokens trained | GPU         | Wall  | Best val ppl |
|-------------------------------|-------:|-------:|--------:|---------------:|-------------|------:|-------------:|
| `smoke_124m.yaml`             | 124M   | 12     | 768     | ~0.2M (smoke)  | M1 Pro MPS  | 1.5 min | 1031        |
| **`gpt2_350m_fweb_5060ti.yaml`** | **354M** | **24** | **1024** | **131M**       | **RTX 5060 Ti 16 GB** | **2 h 27 min** | **58.2** |
| `gpt2_124m.yaml`              | 124M   | 12     | 768     | 10B (target)   | 1×H100      | ~30 h | —            |
| `gpt2_350m.yaml`              | 350M   | 24     | 1024    | 20B (target)   | 8×H100      | ~16 h | —            |
| `gpt2_774m.yaml`              | 774M   | 36     | 1280    | 40B (target)   | 8×H100      | ~4 d  | —            |
| `gpt2_1558m.yaml`             | 1.5B   | 48     | 1600    | 60B (target)   | 8×H100      | ~10 d | —            |

The 5060 Ti row is a *real measured run* (see
[`examples/5060ti_350m_fineweb.md`](examples/5060ti_350m_fineweb.md));
the others are napkin estimates at ~50 % MFU.

## Speed/quality knobs (opt-in, default-off for GPT-2 parity)

Ported from the modded-nanogpt / Liger / DeepSeek-V3 work harvested in
[`../docs/2026-05-sota-llm-agi.md`](../docs/2026-05-sota-llm-agi.md). All are
config-gated and off by default so existing runs are bit-for-bit unchanged.

### Training

- **Muon optimizer** — set `optim.optimizer: muon`. Replaces the Adam update for
  2D *hidden* weight matrices (attn qkv/proj, MLP) with the nearest
  semi-orthogonal matrix via a 5-step Newton-Schulz iteration; embeddings,
  the learned position table, `lm_head`, and all 1-D params stay on AdamW.
  ~1.35× sample-efficiency on the FineWeb GPT-2 task. Tune with `optim.muon_lr`
  (default 0.02, higher than AdamW's) and `optim.muon_momentum`. A single cosine
  schedule scales both optimizers by the same multiplier. See `muon.py`.
- **Liger fused linear-cross-entropy** — set `fused_ce: true` (needs
  `pip install liger-kernel` + a Triton GPU). Computes the `lm_head` matmul and
  the cross-entropy in one kernel *without* materializing the `[B·T, vocab]`
  logits — the largest forward activation. Exact (not an approximation),
  ~20% peak-VRAM saving. Throughput is hardware-dependent and can REGRESS on
  Blackwell (5060 Ti measured ~26% slower than dense matmul + CE); treat as a
  VRAM-headroom lever, not a speedup.
- **Loss-spike rewind** — set `stability.spike_monitor: true`. A two-threshold
  detector (z-score AND absolute jump) watches the per-step loss; on a spike
  the trainer rewinds to the last `ckpt.pt` and halves the LR for a cooldown
  window. Bounded by `max_rewinds` so a chronically-spiky model still
  finishes. See `stability.py` for the bug history that motivated each guard.
- **Multi-Token Prediction (MTP)** — set `model.mtp_tokens: N` (typical 2-4).
  N extra `lm_head`-shaped heads predict tokens at offsets +2, +3, …, +(N+1)
  from the same final hidden state, contributing a `mtp_weight`-scaled CE
  auxiliary to the train loss. Train-only (eval/inference are unchanged),
  ~5-10% sample-efficiency on the DeepSeek-V3 ablation.
- **Zero-init residual projections** — set `model.zero_init_proj: true`. Zeros
  the two residual-write matrices in every block (attention `proj` + MLP/SwiGLU
  `proj`) *after* the GPT-2 `1/sqrt(2N)` rescale, so each block starts as the
  exact identity `x + 0`. muP-like: the early/high-LR phase can't inject
  attention/FFN noise into the residual stream before the norms settle, which
  is what lets you push the warmup LR. Brings midgpt to parity with the same
  knob in `nanogpt-edu` / `distgpt`; MAI-Thinking-1 (§1) uses it at frontier
  scale to keep early attention noise from perturbing MoE routing. Default-off
  (a plain GPT-2 run keeps the rescale-only init, bit-for-bit unchanged).

### Architecture (Llama-style flips)

Three orthogonal flags on `GPTConfig` flip parts of the model from GPT-2
defaults toward Llama. Existing recipes still train pure GPT-2 with the
defaults; flip them for a head-to-head ablation on the same loop.

| Flag | GPT-2 default | Llama-style | What it changes |
|---|---|---|---|
| `pos_kind`  | `learned`   | `rope`     | Drops `pos_emb` table; applies RoPE to Q/K per head |
| `norm_kind` | `layernorm` | `rmsnorm`  | Replaces all block + final norms with RMSNorm (Llama weight-only) |
| `mlp_kind`  | `gelu`      | `swiglu`   | Replaces GELU MLP with gated `proj(silu(w1 x) * w3 x)` |

A fourth knob, `attention_backend` (default `sdpa`), opts into PyTorch
**FlexAttention** (`attention_backend: flex`) for custom-mask / long-context
experiments; SDPA stays the default and picks Flash/mem-efficient kernels
automatically. The flex path is guarded (no dropout, CPU is inference/no-grad
only) and rebuilds its causal block mask per call — fine for experiments, but a
real run should cache masks and `torch.compile` the kernel.

The recipe `configs/gpt2_350m_llamafied_fweb_5060ti.yaml` flips all three on
for a direct A/B against the GPT-2 baseline at the same parameter count
(d_ffn is reduced from 4096 to 2730 ≈ 8/3·d_model to keep SwiGLU's 3-matrix
FFN iso-param with GELU's 2-matrix).

### Eval

- **HuggingFace export** (`export_hf.py`) — writes a trained midgpt
  checkpoint as a `GPT2LMHeadModel`-shaped directory (config.json +
  safetensors + tokenizer). Round-trip verified to bit-identical midgpt
  weights and ~1e-4 fp32 logit-agreement vs `transformers.from_pretrained`.

  ```bash
  python export_hf.py --ckpt out/best.pt --out-dir out/hf_export --verify
  ```

- **`lm-evaluation-harness` runner** (`lm_eval_runner.py`) — wraps the export
  above and hands the dir to EleutherAI's lm-eval. Supports MMLU, HellaSwag,
  LAMBADA, ARC, all standard tasks. Lazy import so lm-eval stays optional.

  ```bash
  python lm_eval_runner.py \
      --ckpt out/best.pt --tasks hellaswag,lambada_openai \
      --device cuda --output results.json
  ```

- **HF export validator** (`tools/validate_hf_export.py`) — dependency-light
  check that an exported directory has a valid `config.json` (required GPT-2
  fields), `generation_config.json`, weights, and tokenizer, then prints the
  vLLM serving command. CI-friendly; `--bench` optionally runs a tiny vLLM
  generation smoke when vLLM is installed.

  ```bash
  python tools/validate_hf_export.py out/hf_export          # fast structural check
  python tools/validate_hf_export.py out/hf_export --bench  # + vLLM generation smoke
  ```

```bash
# Stack-up: Muon + fused-CE + spike-rewind on a single 5060 Ti
python train.py --config configs/gpt2_350m_fweb_5060ti_muon.yaml   # add stability.spike_monitor: true
```


## Worked example: 350M GPT-2 on FineWeb-Edu, 2.5 h on RTX 5060 Ti

A full pretraining run from random init, with a clean loss curve and
real sample completions:
[`examples/5060ti_350m_fineweb.md`](examples/5060ti_350m_fineweb.md).

For a narrative tour of the Tier 6 toolbox (bug fixes, spike-rewind,
HF export, lm-eval, Llama-style flips, MTP) see
[`examples/tier6_toolbox.md`](examples/tier6_toolbox.md).

| | |
|---|---|
| Model         | GPT-2 354M (24L × 1024d × 16H) |
| Dataset       | FineWeb-Edu `sample-10BT`, 1 B-token slice (streamed) |
| Wall-clock    | **2 h 27 min** (4 000 iters, 32 768 tok/step) |
| Throughput    | **14.9 k tok/s** sustained, 99 % GPU util |
| Peak VRAM     | **12.8 GB** / 16 GB |
| Best val      | **ppl 58.2** (loss 4.064) |

![350M training curves](out/gpt2_350m_fweb_5060ti/loss.png)

Textbook loss shape: fast drop in the first ~400 iters as the model
learns the vocab + bigrams, then a long slow descent to ~4.0 as it
actually starts modelling text. Val tracks train to within 0.05 — the
model is undertrained-by-design (0.37× Chinchilla), not overfit.

## Layout

```
midgpt/
├── model.py              # GPT/Llama-style transformer (LayerNorm | RMSNorm,
│                         # learned-posn | RoPE, GELU | SwiGLU, opt. MTP heads)
├── muon.py               # Muon optimizer (Newton-Schulz) + 2D-weight param split
├── stability.py          # SpikeMonitor + RewindController (loss-spike rewind)
├── export_hf.py          # midgpt ⇄ HF GPT2LMHeadModel weight converter
├── lm_eval_runner.py     # EleutherAI lm-evaluation-harness driver
├── data.py               # tiktoken loader, packed sequences, mmap shards
├── prepare.py            # download + tokenize WikiText / OpenWebText / FineWeb-Edu
├── train.py              # DDP loop, AMP, grad-ckpt, grad-accum, resume, spike rewind
├── eval.py               # ppl + HellaSwag zero-shot harness
├── sample.py             # generation (greedy / top-k / top-p)
├── utils/                # logging, schedule, ckpt manager
├── configs/              # YAML per model size & flavor
└── tests/                # 87 tests; CPU smoke + 2-rank gloo distributed
```

## Apple Silicon (MPS) support

`train.py` auto-detects MPS (Metal Performance Shaders) on macOS and runs in
native bf16 with `GradScaler` disabled (Metal doesn't need it for bf16). Tested
on an Apple M1 Pro (24 GB unified, macOS 26.3.1, PyTorch 2.12) — the 124M
config fits at micro-batch 4, sequence length 1024 with grad checkpointing.

A 200-iter smoke run on WikiText-103:

![midgpt MPS smoke run](out/smoke_124m/loss.png)

|              |              |
|--------------|--------------|
| device       | MPS bf16     |
| iters        | 200          |
| wall-clock   | ~1.5 min     |
| step time    | ~485 ms/it (median) |
| throughput   | ~2.1k tok/s  |
| train loss   | 10.96 → 6.99 |
| best val ppl | 1031         |

> ⚠️ Caveat — running a heavier 500-iter config (micro_batch=4, grad_accum=16,
> block_size=512 → 32× more compute per iter than the smoke run) triggered
> what looked like an MPS Metal-driver stall after ~30 iters: the process
> stayed alive but stopped advancing (`STAT=U`, ~8% CPU, 25 s of CPU time
> across 25 min wall-clock). The smoke config above completes cleanly — if
> you hit the same hang, reduce `micro_batch`/`grad_accum` and/or
> `block_size`, or set `grad_checkpoint: false`.

## What's still omitted (see `distgpt/`)

FSDP / tensor parallelism / pipeline parallelism / multi-node orchestration /
DCP (sharded reshardable checkpoints) / MoE / MLA / RLHF — anything that
needs more than one node lives in [`distgpt/`](../distgpt). midgpt is
deliberately single-node so that the full training loop fits on one screen
of mental model.
