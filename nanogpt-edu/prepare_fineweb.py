"""Download a slice of FineWeb-Edu and write GPT-2-BPE token shards.

The honest fix for nanogpt-edu's "everything overfits 1 MB of Shakespeare"
problem is *more and better tokens*, not a fancier optimizer. FineWeb-Edu is a
high-quality educational-web subset; a few hundred MB of it is enough to make
the `small`/`medium` configs stop overfitting and let bigger models actually
win on val (Chinchilla: scale tokens with params).

Output format is identical to prepare_shakespeare.py — train.bin / val.bin of
uint16 token ids + meta.pkl — so data.py / train.py / sample.py need no change.
The only difference is meta["tokenizer"] == "gpt2", which sample.py uses to
decode with tiktoken instead of a char map.

GPT-2 BPE vocab is 50257 < 65536, so uint16 still fits.

Usage:
    python prepare_fineweb.py                 # ~100M tokens (default)
    python prepare_fineweb.py --tokens 300_000_000
    python prepare_fineweb.py --out-dir data_fineweb

Then point a config's data_dir at the chosen --out-dir and train.
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=100_000_000,
                    help="approx number of training tokens to collect")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data_fineweb"))
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT",
                    help="HF dataset config/subset name (a small sample by default)")
    ap.add_argument("--val-frac", type=float, default=0.005,
                    help="fraction of collected tokens held out for validation")
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256; separates documents

    # Stream so we never materialize the whole sample on disk/RAM.
    ds = load_dataset(args.dataset, name=args.name, split="train", streaming=True)

    buf = np.empty(args.tokens + 1_000_000, dtype=np.uint16)
    n = 0
    docs = 0
    for ex in ds:
        ids = enc.encode_ordinary(ex["text"])
        ids.append(eot)
        if n + len(ids) > len(buf):
            ids = ids[: len(buf) - n]
        buf[n : n + len(ids)] = ids
        n += len(ids)
        docs += 1
        if docs % 1000 == 0:
            print(f"\r  {docs:,} docs  {n:,} tokens", end="", flush=True)
        if n >= args.tokens:
            break
    print(f"\ncollected {n:,} tokens from {docs:,} documents")

    n_val = int(n * args.val_frac)
    val_ids = buf[:n_val]
    train_ids = buf[n_val:n]
    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.out_dir, "val.bin"))
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        # tokenizer="gpt2" tells train.py/sample.py to use tiktoken; vocab_size
        # is the GPT-2 BPE size. No stoi/itos needed for BPE.
        pickle.dump({"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}, f)
    print(f"train={len(train_ids):,} val={len(val_ids):,} -> {args.out_dir}")
    print(f"point a config at data_dir='{os.path.relpath(args.out_dir)}' to train")


if __name__ == "__main__":
    main()
