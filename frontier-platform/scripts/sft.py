#!/usr/bin/env python3
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    print(f"[sft] base={args.base} data={args.data}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
