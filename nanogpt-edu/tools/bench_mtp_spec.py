"""Serving benchmark: MTP heads as a self-speculative draft model.

The Multi-Token Prediction heads (`GPTConfig.mtp_tokens`) are trained only as an
auxiliary loss and discarded by the normal `generate()`. But each head predicts a
*future* token from the same final hidden state, so at inference they double as a
Medusa-style speculative-decoding draft — for free. This tool measures the payoff:

    python tools/bench_mtp_spec.py --ckpt out/tiny_mtp/ckpt.pt --tokens 256

It runs three decoders to the same token budget and reports tokens/sec + speedup:
  * baseline greedy (main head only, one trunk pass per token)
  * MTP-speculative greedy (one trunk pass drafts K+1 tokens, one verifies)
and asserts the two produce **identical** output (greedy verification is exact),
so the speedup is a pure latency win with zero quality change.

With no checkpoint it builds a random tiny MTP model so the harness is runnable
anywhere (CI / laptop) — the *mechanism* and acceptance accounting are the point;
absolute speed needs a trained checkpoint where drafts are actually accurate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig  # noqa: E402


def _load_ckpt(path: str, device: str):
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = sd["cfg"]
    mcfg = GPTConfig(
        vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"],
        d_model=cfg["d_model"], d_ffn=cfg["d_ffn"], dropout=0.0,
        rope_base=cfg["rope_base"],
        qk_norm=cfg.get("qk_norm", False),
        zero_init_proj=cfg.get("zero_init_proj", False),
        tie_embeddings=cfg.get("tie_embeddings", True),
        mtp_tokens=cfg.get("mtp_tokens", 0),
    )
    model = GPT(mcfg).to(device).eval()
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)
    return model, sd.get("meta", {})


def _random_model(device: str, mtp_tokens: int):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=256, block_size=256, n_layer=4, n_head=4, n_kv_head=4,
                    d_model=128, d_ffn=256, mtp_tokens=mtp_tokens)
    return GPT(cfg).to(device).eval(), {}


def _encode_prompt(prompt: str, meta: dict, device: str) -> torch.Tensor:
    if meta.get("tokenizer") == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        ids = enc.encode(prompt) or [enc.eot_token]
    elif "stoi" in meta:
        stoi = meta["stoi"]
        ids = [stoi.get(c, 0) for c in prompt] or [0]
    else:
        ids = [1, 2, 3, 4]   # random-model harness: arbitrary seed tokens
    return torch.tensor([ids], dtype=torch.long, device=device)


def _timed(fn, *, sync: bool, device: str):
    if sync and device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync and device.startswith("cuda"):
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint path (omit for a random tiny model)")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=256, help="new tokens to generate")
    ap.add_argument("--warmup", type=int, default=16, help="warmup tokens (excluded from timing)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mtp-tokens", type=int, default=3, help="MTP heads for the random-model fallback")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    if args.ckpt:
        model, meta = _load_ckpt(args.ckpt, device)
        src = args.ckpt
    else:
        model, meta = _random_model(device, args.mtp_tokens)
        src = f"<random tiny model, mtp_tokens={args.mtp_tokens}>"

    K = len(model.mtp_heads)
    if K == 0:
        raise SystemExit(
            "this checkpoint has mtp_tokens=0 — no MTP draft heads to benchmark. "
            "Train with mtp_tokens>=1 (e.g. configs/tiny_mtp.py)."
        )

    ids = _encode_prompt(args.prompt, meta, device)
    print(f"[bench] model={src}")
    print(f"[bench] device={device}  mtp_heads={K}  prompt_len={ids.size(1)}  tokens={args.tokens}")

    # Warmup (kernels / caches), excluded from the timed region.
    if args.warmup:
        model.generate_greedy(ids.clone(), args.warmup)
        model.generate_mtp_speculative(ids.clone(), args.warmup)

    base, base_dt = _timed(lambda: model.generate_greedy(ids.clone(), args.tokens),
                           sync=True, device=device)
    (spec, stats), spec_dt = _timed(
        lambda: model.generate_mtp_speculative(ids.clone(), args.tokens),
        sync=True, device=device)

    identical = torch.equal(base, spec)
    base_tps = args.tokens / base_dt
    spec_tps = args.tokens / spec_dt
    rounds = stats["rounds"]
    total = sum(stats["accepted"])
    mean_acc = total / rounds if rounds else 0.0
    # Acceptance length = tokens emitted per verification pass; >1 means the
    # drafts are paying off. A round emits 1 (true token) + up to K accepted
    # drafts + 1 bonus token when the whole chain verifies, so the ceiling is K+2.
    draft_accept = (mean_acc - 1) / K if K else 0.0

    print("\n  decoder                 tokens/s     wall (s)   trunk passes")
    print(f"  baseline greedy         {base_tps:8.1f}     {base_dt:7.3f}   {args.tokens:>6}")
    print(f"  MTP-speculative         {spec_tps:8.1f}     {spec_dt:7.3f}   {2 * rounds:>6}")
    print(f"\n  identical output:       {identical}")
    print(f"  speedup:                {base_tps and spec_tps / base_tps:5.2f}x")
    print(f"  verification rounds:    {rounds}")
    print(f"  mean tokens/round:      {mean_acc:.2f}  (1 + {draft_accept * 100:.0f}% of {K} drafts, +bonus)")
    print(f"  trunk-pass reduction:   {args.tokens} -> {2 * rounds} "
          f"({(1 - 2 * rounds / args.tokens) * 100:.0f}% fewer)")

    if not identical:
        raise SystemExit("ERROR: speculative output diverged from greedy — verification bug")
    if not args.ckpt:
        print("\n[note] random untrained model: drafts accept ~by chance. Point --ckpt at a "
              "trained mtp_tokens>=1 checkpoint to see real acceptance / wall-clock speedup.")


if __name__ == "__main__":
    main()
