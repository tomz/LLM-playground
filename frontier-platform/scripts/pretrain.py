#!/usr/bin/env python3
"""Entry point: `python scripts/pretrain.py --config configs/model_7b.yaml`."""
import argparse, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="latest")
    args = ap.parse_args()
    print(f"[pretrain] would load {args.config}, resume={args.resume}")
    print("[pretrain] this is a blueprint; wire to platform.training.trainer.Trainer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
