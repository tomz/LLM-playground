#!/usr/bin/env python3
import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prefs", required=True)
    args = ap.parse_args()
    print(f"[dpo] policy={args.policy} ref={args.ref} prefs={args.prefs}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
