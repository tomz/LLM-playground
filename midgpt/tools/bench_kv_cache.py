"""Benchmark KV-cached vs full-recompute greedy decoding; write a JSON receipt.

Usage: python tools/bench_kv_cache.py --out docs/figures/kv_cache_bench.json
"""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig  # noqa: E402


def _rate(fn, prompt, n, dev):
    fn(prompt, 8, temperature=0)  # warm-up
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(prompt, n, temperature=0)
    if dev == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--prompts", default="64,256,512,768")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = GPTConfig(block_size=1024, pos_kind="rope", norm_kind="rmsnorm", mlp_kind="swiglu", d_ffn=2048)
    m = GPT(cfg).to(dev).eval()
    rows = []
    for p in map(int, a.prompts.split(",")):
        prompt = torch.randint(0, cfg.vocab_size, (1, p), device=dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            full, x = _rate(m.generate, prompt, a.new_tokens, dev)
            cached, y = _rate(m.generate_cached, prompt, a.new_tokens, dev)
        row = {"prompt_tokens": p, "full_tok_s": round(full, 1), "cached_tok_s": round(cached, 1),
               "speedup": round(cached / full, 2), "identical": bool(torch.equal(x, y))}
        if not row["identical"]:
            pos = int((x != y).nonzero()[0, 1])
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits, _ = m(x[:, :pos][:, -cfg.block_size:])
            top2 = logits[0, -1].float().topk(2).values
            row["first_divergent_new_token"] = pos - p
            row["full_path_top2_margin"] = round(float(top2[0] - top2[1]), 5)
        rows.append(row)
        print(rows[-1])
    receipt = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
               "torch": torch.__version__, "params_m": round(m.num_params() / 1e6), "dtype": "bf16 autocast" if dev == "cuda" else "fp32",
               "batch": 1, "new_tokens": a.new_tokens, "decode": "greedy", "rows": rows,
               "note": "Python-loop decode, dynamic cache, no torch.compile/CUDA graphs; one run per point. Under bf16 the cached and full paths use different attention kernels, so a near-tied greedy argmax may flip; divergence rows record the full path's top-2 logit margin. The CPU fp32 test suite requires exact equality."}
    Path(a.out).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
