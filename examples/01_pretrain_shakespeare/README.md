# 01 — TinyShakespeare pretraining

A full pretrain pipeline on a single CUDA GPU, using only `frontier-platform`
components:

- `platform.tokenizer.bpe.train()` — trains a real ByteLevel BPE (vocab 4096) via the HF `tokenizers` Rust library
- `platform.data.shard.tokenize_and_shard()` — writes uint32 `.bin` shards + `.idx` metadata
- `platform.data.loader.StreamingLoader` — mmap-backed deterministic loader
- `platform.data.mix.MixtureSampler` — even with a single domain, the loader API expects it
- `platform.model.transformer.Transformer` — GQA, RoPE, RMSNorm, SwiGLU
- `platform.training.trainer.Trainer` — cosine LR, grad clip, AMP-free fp32
- `platform.training.checkpoint.CheckpointManager` — keeps last K checkpoints
- `platform.serving.engine.Engine` / `TorchEngine` — inference at end

## Model

```
ModelConfig(vocab_size=4096, n_layer=8, n_head=8, n_kv_head=4,
            d_model=384, d_ffn=1024, max_seq_len=256)
```

~12 M parameters. 3 k steps at micro-batch 16 × seq 256 = ~12 M tokens seen.

## Run

```bash
bash run.sh
```

Produces:
- `out/data/input.txt`, `out/data/train.txt`, `out/data/val.txt`
- `out/tok/tokenizer.json`
- `out/shards/shakespeare/*.bin` and `.idx`
- `out/ckpts/shakespeare-12M/ckpts/step_*/state.pt` (rolling checkpoints from `CheckpointManager`)
- `out/final.pt` — self-contained `{model, model_cfg}` for examples 02/03
- `out/sample.txt` — 500-token generation from `"ROMEO:"`
- `result.md` — final metrics

## Notes

We use fp32 throughout. The GPU has plenty of headroom for a 12 M-param model;
fp16 without `GradScaler` would risk gradient underflow at this batch size.
A fully-fledged `Trainer` upgrade for AMP would slot in here cleanly.
