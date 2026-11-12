"""Tokenize and write fixed-size token shards.

Format: a shard is `<out_dir>/<domain>/<uuid>.bin` containing little-endian
uint32 token IDs, plus a sibling `.idx` JSON with metadata.
"""
from __future__ import annotations
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np

SHARD_TOKENS = 1 << 28  # ~256M tokens ≈ 1 GB at uint32
SCHEMA_VERSION = 1


def _bos_id(tokenizer) -> int:
    for attr in ("bos_id", "bos_token_id"):
        v = getattr(tokenizer, attr, None)
        if isinstance(v, int):
            return v
    return 0


def tokenize_and_shard(
    docs_iter,
    tokenizer,
    out_dir: str | Path,
    domain: str,
    shard_tokens: int = SHARD_TOKENS,
) -> list[str]:
    out_root = Path(out_dir) / domain
    out_root.mkdir(parents=True, exist_ok=True)
    bos = _bos_id(tokenizer)
    uris: list[str] = []

    buf: list[int] = []
    doc_offsets: list[int] = []

    def flush() -> None:
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint32)
        name = f"{uuid.uuid4().hex}.bin"
        path = out_root / name
        path.write_bytes(arr.tobytes(order="C"))
        sha = hashlib.sha256(arr.tobytes()).hexdigest()
        idx = {
            "schema_version": SCHEMA_VERSION,
            "tokens": int(arr.size),
            "sha256": sha,
            "doc_offsets": list(doc_offsets),
            "domain": domain,
            "dtype": "uint32",
        }
        path.with_suffix(".idx").write_text(json.dumps(idx))
        uris.append(str(path))
        buf.clear()
        doc_offsets.clear()

    for doc in docs_iter:
        if isinstance(doc, (bytes, bytearray)):
            text = bytes(doc).decode("utf-8", errors="replace")
        elif isinstance(doc, str):
            text = doc
        elif hasattr(doc, "payload"):  # RawDoc-ish
            text = doc.payload.decode("utf-8", errors="replace")
        else:
            text = str(doc)
        ids = list(tokenizer.encode(text))
        doc_offsets.append(len(buf))
        buf.append(bos)
        buf.extend(ids)
        if len(buf) >= shard_tokens:
            flush()
    flush()
    return uris
