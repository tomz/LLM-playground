"""Tokenize and write fixed-size token shards.

Format: a shard is `<run>/<domain>/<uuid>.bin` containing little-endian uint32
token IDs, plus a sibling `.idx` JSON with {tokens, sha256, doc_offsets}.
"""
from __future__ import annotations
from pathlib import Path

SHARD_TOKENS = 1 << 28  # ~256M tokens ≈ 1 GB at uint32


def tokenize_and_shard(
    docs_iter,
    tokenizer,
    out_dir: str | Path,
    domain: str,
    shard_tokens: int = SHARD_TOKENS,
) -> list[str]:
    """Stream docs, tokenize, append <bos> sep, roll over at shard_tokens.

    Returns list of shard URIs written.
    """
    raise NotImplementedError
