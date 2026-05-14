#!/usr/bin/env python3
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="fast", choices=["fast", "full", "release"])
    args = ap.parse_args()
    print(f"[eval] ckpt={args.ckpt} suite={args.suite}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
