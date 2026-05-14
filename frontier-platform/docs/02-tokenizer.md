# 02 — Tokenizer

## Choice

- **Algorithm**: byte-level BPE (GPT-2/3/4 family) is the default. Unigram (SentencePiece) is preferred when many low-resource languages must be balanced.
- **Vocab size**: 32k (small models), 100k (frontier), 200k (multilingual frontier). Larger vocab = fewer tokens per doc = faster training, but bigger embedding matrix and worse rare-token coverage.
- **Special tokens**: `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool_call|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, plus 256 reserved.
- **Number handling**: split digits individually (LLaMA-style) for arithmetic.
- **Whitespace**: preserve exactly (no NFKC normalization on code).

## Training

Train on a 50–200 GB representative sample of the final mix — *not* on Common Crawl alone, or code/math will tokenize poorly.

Use HuggingFace `tokenizers` (Rust) or `sentencepiece`. Training a 100k BPE on 100GB takes ~12h on a 96-core box.

## Freeze early

The tokenizer is a hard contract: changing it invalidates every checkpoint and every dataset shard. Freeze before pretraining starts.
