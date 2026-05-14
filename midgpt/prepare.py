"""Tokenize WikiText-103 or OpenWebText with tiktoken → uint16 shards."""
import argparse, os
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

SHARD_TOKENS = 1 << 26   # ~67M tokens ≈ 134 MB at uint16


def tokenize_doc(enc, text: str, eot: int) -> np.ndarray:
    ids = enc.encode_ordinary(text)
    ids.append(eot)
    arr = np.asarray(ids, dtype=np.int64)
    if (arr >= 2**16).any():
        raise ValueError("vocab > 65535 not supported with uint16; use uint32")
    return arr.astype(np.uint16)


def write_shards(token_iter, out_dir: str, split: str, shard_tokens: int = SHARD_TOKENS):
    os.makedirs(out_dir, exist_ok=True)
    buf = np.empty(shard_tokens, dtype=np.uint16)
    pos, shard_idx = 0, 0
    pbar = tqdm(desc=f"{split}")
    for arr in token_iter:
        n = len(arr)
        i = 0
        while i < n:
            take = min(shard_tokens - pos, n - i)
            buf[pos : pos + take] = arr[i : i + take]
            pos += take; i += take
            if pos == shard_tokens:
                path = os.path.join(out_dir, f"{split}_{shard_idx:06d}.bin")
                buf.tofile(path); shard_idx += 1; pos = 0
        pbar.update(n)
    if pos:
        path = os.path.join(out_dir, f"{split}_{shard_idx:06d}.bin")
        buf[:pos].tofile(path)
    pbar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["wikitext103", "openwebtext"])
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-proc", type=int, default=1)
    args = ap.parse_args()

    enc = tiktoken.get_encoding(args.tokenizer)
    eot = enc.eot_token
    out_dir = args.out or os.path.join("data", args.dataset)

    if args.dataset == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        for split in ("train", "validation"):
            it = (tokenize_doc(enc, ex["text"], eot) for ex in ds[split] if ex["text"].strip())
            write_shards(it, out_dir, "val" if split == "validation" else "train")
    else:
        ds = load_dataset("Skylion007/openwebtext", num_proc=args.num_proc, trust_remote_code=True)
        # 99/1 train/val split
        split = ds["train"].train_test_split(test_size=0.005, seed=2357, shuffle=True)
        for name, sub in (("train", split["train"]), ("val", split["test"])):
            it = (tokenize_doc(enc, ex["text"], eot) for ex in sub)
            write_shards(it, out_dir, name)

    print(f"done. shards in {out_dir}")


if __name__ == "__main__":
    main()
