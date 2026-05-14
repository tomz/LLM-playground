#!/usr/bin/env python3
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"[serve] ckpt={args.ckpt} tp={args.tp} port={args.port}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
