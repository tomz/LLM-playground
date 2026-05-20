"""Generate dummy uint16 token shards for smoke testing."""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=10_000_000)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--shard-tokens", type=int, default=1 << 22)  # ~4M tokens / shard
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    written, idx = 0, 0
    while written < args.tokens:
        n = min(args.shard_tokens, args.tokens - written)
        arr = rng.integers(0, args.vocab, size=n, dtype=np.int64).astype(np.uint16)
        arr.tofile(os.path.join(args.out, f"shard_{idx:06d}.bin"))
        written += n; idx += 1
    print(f"wrote {idx} shards, {written:,} tokens → {args.out}")


if __name__ == "__main__":
    main()
