"""Stream document-isolated GPT-2 BPE shards with auditable UTF-8 normalization.

Assign each whole document by its text hash BEFORE tokenization/concatenation.
Exact duplicate text therefore stays in one split, regardless of source order.
This is not near-duplicate removal or benchmark decontamination. ``val.bin`` is
adaptive search validation; ``holdout.bin`` is a sealed final-test role, not an
OS-enforced sandbox. Candidate code and local metadata remain trusted.

The token target counts ALL splits and is approximate: finish the last document
rather than cutting it. Memory is bounded by the maximum accepted document plus
one write chunk, never by the corpus token target. The injectable writer needs
no network; only the CLI imports Hugging Face/tiktoken and downloads data.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from importlib.metadata import version
import json
import logging
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Iterable, Protocol

import numpy as np

from data import MANIFEST_SCHEMA, SPLIT_ROLES, sha256_file

LOG = logging.getLogger(__name__)
DEFAULT_MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
ALIASES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "dclm": ("mlfoundations/dclm-baseline-1.0", None),
}


class Tokenizer(Protocol):
    """The small injectable portion of tiktoken used by the streaming writer."""

    n_vocab: int
    eot_token: int

    def encode_ordinary(self, text: str) -> list[int]: ...


def validate_options(*, tokens: int, val_frac: float, holdout_frac: float,
                     seed: int, max_document_bytes: int, chunk_tokens: int) -> None:
    """Reject invalid preparation limits before opening data or downloading."""
    for name, value in (("tokens", tokens), ("max_document_bytes", max_document_bytes),
                        ("chunk_tokens", chunk_tokens)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int:
        raise ValueError("split seed must be an integer")
    for name, value in (("val_frac", val_frac), ("holdout_frac", holdout_frac)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value < 1):
            raise ValueError(f"{name} must be finite and between zero and one")
    if val_frac + holdout_frac >= 1:
        raise ValueError("validation and holdout fractions must leave training documents")


def document_split(document_sha256: str, *, seed: int, val_frac: float,
                   holdout_frac: float) -> str:
    """Stable content-only assignment; fractions apply to documents, not tokens."""
    digest = hashlib.sha256(f"{seed}:{document_sha256}".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big")
    if bucket < int(val_frac * 2**64):
        return "val"
    if bucket < int((val_frac + holdout_frac) * 2**64):
        return "holdout"
    return "train"


def _artifact(path: Path) -> dict:
    return {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _identity(identity: dict, fields: tuple[str, ...], label: str) -> dict:
    """Require explicit immutable/caller-qualified identity, not guessed labels."""
    for field in fields:
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"{label}.{field} must be a nonempty string")
    # Copy nested input and reject non-JSON/nonfinite identity values early.
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def write_document_shards(
    documents: Iterable[dict | str], tokenizer: Tokenizer, *, out_dir: str | Path,
    tokens: int, source: dict, tokenizer_identity: dict, val_frac: float = 0.005,
    holdout_frac: float = 0.005, seed: int = 1337,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES, chunk_tokens: int = 65536,
) -> dict:
    """Write whole documents and publish a complete new dataset directory.

    ``source`` records id/config/revision/split. Injected callers assert those
    identities; the CLI resolves the HF revision to a commit before streaming.
    Per-split document ledgers record source indices, raw UTF-8 SHA-256, token
    offsets/counts, and byte counts, including repeats. EOT is a zero-byte token
    and is included in the denominator. Empty/oversized documents are skipped
    and counted. No document text, corpus-sized token array, or hash set is kept.
    """
    validate_options(tokens=tokens, val_frac=val_frac, holdout_frac=holdout_frac,
                     seed=seed, max_document_bytes=max_document_bytes, chunk_tokens=chunk_tokens)
    source = _identity(source, ("id", "revision", "split"), "source")
    identity = _identity(tokenizer_identity, ("name", "implementation", "version", "sha256"), "tokenizer")
    if len(identity["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in identity["sha256"]):
        raise ValueError("tokenizer.sha256 must be a lowercase SHA-256 digest")
    if type(tokenizer.n_vocab) is not int or not 0 < tokenizer.n_vocab <= 65536:
        raise ValueError("tokenizer vocabulary must fit uint16")
    if type(tokenizer.eot_token) is not int or not 0 <= tokenizer.eot_token < tokenizer.n_vocab:
        raise ValueError("EOT must be a valid token ID")
    identity.update(n_vocab=tokenizer.n_vocab, eot_token=tokenizer.eot_token)
    output = Path(out_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("use a new or empty output directory; do not overwrite a sealed dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {split: {"role": role, "documents_count": 0, "tokens": 0, "utf8_bytes": 0}
              for split, role in SPLIT_ROLES.items()}
    total = seen = empty = oversized = 0
    selected_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        with ExitStack() as stack:
            streams = {split: stack.enter_context((stage / f"{split}.bin").open("wb")) for split in counts}
            ledgers = {split: stack.enter_context((stage / f"{split}.documents.jsonl").open("w", encoding="utf-8"))
                       for split in counts}
            for index, document in enumerate(documents):
                seen += 1
                text = document if isinstance(document, str) else document["text"]
                if not isinstance(text, str):
                    raise ValueError("every document must contain string text")
                if not text:
                    empty += 1
                    continue
                if len(text) > max_document_bytes:
                    oversized += 1
                    continue
                raw = text.encode("utf-8")
                if len(raw) > max_document_bytes:
                    oversized += 1
                    continue
                doc_hash = hashlib.sha256(raw).hexdigest()
                split = document_split(doc_hash, seed=seed, val_frac=val_frac, holdout_frac=holdout_frac)
                ids = tokenizer.encode_ordinary(text)
                if not ids or any(type(token) is not int or not 0 <= token < tokenizer.n_vocab for token in ids):
                    raise ValueError("tokenizer returned empty or invalid token IDs")
                ids.append(tokenizer.eot_token)
                info = counts[split]
                record = {"source_index": index, "sha256": doc_hash, "utf8_bytes": len(raw),
                          "token_offset": info["tokens"], "tokens": len(ids)}
                ledgers[split].write(json.dumps(record, sort_keys=True) + "\n")
                selected_digest.update((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
                for start in range(0, len(ids), chunk_tokens):
                    streams[split].write(np.asarray(ids[start:start + chunk_tokens], dtype="<u2").tobytes())
                info["documents_count"] += 1
                info["tokens"] += len(ids)
                info["utf8_bytes"] += len(raw)
                total += len(ids)
                if total >= tokens:
                    break
        if total == 0:
            raise ValueError("no nonempty documents fit max_document_bytes")
        for split, info in counts.items():
            info["bytes_per_token"] = info["utf8_bytes"] / info["tokens"] if info["tokens"] else None
            info["artifact"] = _artifact(stage / f"{split}.bin")
            info["documents"] = _artifact(stage / f"{split}.documents.jsonl")
        manifest = {
            "schema": MANIFEST_SCHEMA, "dtype": "<u2", "source": source, "tokenizer": identity,
            "split_policy": {
                "algorithm": "sha256(seed_ascii + ':' + utf8_sha256_hex)[:8]_big_endian",
                "seed": seed, "val_frac": val_frac, "holdout_frac": holdout_frac,
                "val_upper_bound": int(val_frac * 2**64),
                "holdout_upper_bound": int((val_frac + holdout_frac) * 2**64),
                "unit": "whole_document", "exact_duplicates": "same_split_not_removed",
            },
            "normalization": {"numerator": "raw_document_utf8_bytes", "denominator": "stored_tokens_including_eot",
                              "eot_utf8_bytes": 0},
            "collection": {"requested_tokens": tokens, "written_tokens": total,
                           "stop_policy": "finish_document_at_total_token_target_or_source_exhaustion",
                           "documents_seen": seen, "skipped_empty": empty, "skipped_oversized": oversized,
                           "max_document_bytes": max_document_bytes,
                           "selected_documents_sha256": selected_digest.hexdigest()},
            "splits": counts,
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        meta = {"vocab_size": tokenizer.n_vocab, "tokenizer": identity["name"],
                "bytes_per_token": {split: info["bytes_per_token"] for split, info in counts.items()},
                "data_manifest_schema": MANIFEST_SCHEMA, "manifest_sha256": sha256_file(manifest_path)}
        with (stage / "meta.pkl").open("wb") as stream:
            pickle.dump(meta, stream)
        os.replace(stage, output)
    return manifest


def tiktoken_identity(encoder: Tokenizer) -> dict:
    """Fingerprint actual GPT-2 ranks, special tokens, pattern and implementation."""
    digest = hashlib.sha256()
    digest.update(encoder._pat_str.encode("utf-8"))
    for token, rank in sorted(encoder._mergeable_ranks.items(), key=lambda item: item[1]):
        digest.update(rank.to_bytes(8, "big") + len(token).to_bytes(8, "big") + token)
    digest.update(json.dumps(encoder._special_tokens, sort_keys=True).encode("utf-8"))
    return {"name": encoder.name, "implementation": "tiktoken", "version": version("tiktoken"),
            "sha256": digest.hexdigest(), "fingerprint": "pattern+ranked_mergeable_bytes+special_tokens/v1"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=100_000_000, help="approximate total tokens across all splits; finish whole documents")
    ap.add_argument("--out-dir", default=str(Path(__file__).with_name("data_fineweb")))
    ap.add_argument("--dataset", default="fineweb-edu", help="fineweb-edu, dclm, or a HF dataset with a text column")
    ap.add_argument("--name", default=None, help="HF config/subset; known aliases have defaults")
    ap.add_argument("--revision", default="main", help="HF revision, resolved to a commit and recorded before streaming")
    ap.add_argument("--val-frac", type=float, default=0.005, help="document-hash fraction for adaptive search validation")
    ap.add_argument("--holdout-frac", type=float, default=0.005, help="document-hash fraction sealed for final evaluation")
    ap.add_argument("--split-seed", type=int, default=1337)
    ap.add_argument("--max-document-bytes", type=int, default=DEFAULT_MAX_DOCUMENT_BYTES)
    args = ap.parse_args()
    validate_options(tokens=args.tokens, val_frac=args.val_frac, holdout_frac=args.holdout_frac,
                     seed=args.split_seed, max_document_bytes=args.max_document_bytes, chunk_tokens=65536)
    output = Path(args.out_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        ap.error("use a new or empty --out-dir; existing datasets are never overwritten")
    # Optional/network dependencies are deliberately below all option validation.
    import tiktoken
    from datasets import load_dataset
    from huggingface_hub import HfApi

    dataset_id, default_name = ALIASES.get(args.dataset, (args.dataset, None))
    name = args.name if args.name is not None else default_name
    revision = HfApi().dataset_info(dataset_id, revision=args.revision).sha
    if not revision:
        raise ValueError("could not resolve an immutable dataset revision")
    encoder = tiktoken.get_encoding("gpt2")
    documents = load_dataset(dataset_id, name=name, revision=revision, split="train", streaming=True)
    manifest = write_document_shards(
        documents, encoder, out_dir=output, tokens=args.tokens,
        source={"id": dataset_id, "config": name, "revision": revision, "split": "train",
                "datasets_version": version("datasets")},
        tokenizer_identity=tiktoken_identity(encoder), val_frac=args.val_frac,
        holdout_frac=args.holdout_frac, seed=args.split_seed, max_document_bytes=args.max_document_bytes,
    )
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOG.info("Wrote %s tokens to %s", manifest["collection"]["written_tokens"], output)
    for split, info in manifest["splits"].items():
        LOG.info("%s: %s documents, %s tokens, bytes/token=%s", split, info["documents_count"], info["tokens"], info["bytes_per_token"])
        if not info["tokens"]:
            LOG.warning("%s is empty: collect more documents before using this split", split)


if __name__ == "__main__":
    main()
