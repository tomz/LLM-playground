#!/usr/bin/env python3
"""End-to-end frontier-program simulation.

Runs: data → tokenizer → pretrain → SFT+DPO → eval → safety → serving.
Produces console report + events.jsonl + summary.json under out/sim/<name>/.

Usage:
    python scripts/simulate.py --size 7b
    python scripts/simulate.py --size 70b --gpus 2048 --gpu-type H100
    python scripts/simulate.py --size 400b --gpus 8192 --gpu-type B200 --rlhf ppo
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from platform.sim.cluster import Cluster
from platform.sim.alignment_sim import AlignmentSpec
from platform.sim.serving_sim import ServingTier
from platform.sim.orchestrator import ProgramSpec, run_program


PRESETS = {
    # name: (n_params, tokens, seq_len, global_batch_tokens, default_gpus)
    "1b":   (1.2e9,  1.0e12, 4096, 1_000_000,   64),
    "7b":   (6.7e9,  2.0e12, 4096, 4_000_000,  512),
    "70b":  (7.0e10, 5.0e12, 4096, 8_000_000, 4096),
    "400b": (4.0e11, 1.5e13, 4096,16_000_000, 16384),
}


def build(args) -> ProgramSpec:
    n_params, tokens, seq, gbt, default_gpus = PRESETS[args.size]
    gpus = args.gpus or default_gpus
    gpus_per_node = 8
    nodes = max(1, gpus // gpus_per_node)
    pretrain_cluster = Cluster(name=f"{args.size}_pretrain", n_nodes=nodes,
                               gpus_per_node=gpus_per_node, gpu_type=args.gpu_type)
    eval_cluster = Cluster(name=f"{args.size}_eval", n_nodes=8, gpu_type=args.gpu_type)

    alignment = AlignmentSpec(
        sft_examples=args.sft_examples,
        pref_pairs=args.pref_pairs,
        rlhf=args.rlhf,
    )

    serving_tiers = [
        ServingTier("nano", n_params=1.2e9,  quant="int4", gpu="A100", gpus_per_replica=1,
                    target_throughput_tok_s=4000, ttft_ms=50),
        ServingTier("mid",  n_params=6.7e9,  quant="fp8",  gpu=args.gpu_type, gpus_per_replica=1,
                    target_throughput_tok_s=2500, ttft_ms=100),
        ServingTier("pro",  n_params=n_params if n_params <= 7e10 else 7e10,
                    quant="fp8", gpu=args.gpu_type, gpus_per_replica=8,
                    target_throughput_tok_s=1800, ttft_ms=250),
        ServingTier("max",  n_params=n_params, quant="fp8", gpu=args.gpu_type, gpus_per_replica=64,
                    target_throughput_tok_s=900, ttft_ms=600),
    ]
    qps = {"nano": 80.0, "mid": 30.0, "pro": 8.0, "max": 1.5}

    return ProgramSpec(
        name=args.size, n_params=n_params, total_tokens=tokens,
        seq_len=seq, global_batch_tokens=gbt,
        pretrain_cluster=pretrain_cluster, eval_cluster=eval_cluster,
        alignment=alignment, serving_tiers=serving_tiers, serving_qps=qps,
        out_dir=args.out_dir or f"out/sim/{args.size}",
        seed=args.seed,
    )


def report(result: dict, spec: ProgramSpec) -> None:
    print()
    print("═" * 78)
    print(f" FRONTIER PROGRAM SIMULATION  —  {spec.name.upper()}")
    print("═" * 78)
    print(f" model:        {spec.n_params/1e9:>8.1f} B params")
    print(f" tokens:       {spec.total_tokens/1e12:>8.2f} T")
    print(f" cluster:      {spec.pretrain_cluster.total_gpus:>8d} × {spec.pretrain_cluster.gpu_type}")
    print(f" wall-clock:   {result['clock_days']:>8.1f} days  ({result['clock_days']/30:.1f} months)")
    print()
    pre = result["pretrain"]
    print(f" PRETRAIN: {pre['total_steps']:,} steps  final-loss {pre['final_loss']:.3f}  spikes {pre['spikes']}  GPU-failures {pre['failures']}")
    al = result["alignment"]
    print(f" ALIGN:    sft_quality {al['sft_quality']:.3f}  rlhf_quality {al['rlhf_quality']:.3f}")
    ev = result["eval"]
    print(f" EVAL:     MMLU {ev['mmlu']*100:5.1f}%   HumanEval {ev['humaneval']*100:5.1f}%   GSM8K {ev['gsm8k']*100:5.1f}%   ELO {ev['arena_elo']:6.0f}")
    sf = result["safety"]
    print(f" SAFETY:   {sf['verdict']}" + (f"   failed: {','.join(sf['failed'])}" if sf['failed'] else ""))
    for k, v in sf["scores"].items():
        print(f"             {k:12s} {v:.3f}")
    print()
    print(" SERVING (24h projection):")
    print(f"   {'tier':<6s} {'qps':>6s} {'replicas':>9s} {'gpus':>6s} {'$/day':>10s} {'$/Mtok':>9s}")
    for name, s in result["serving"].items():
        print(f"   {name:<6s} {s['qps']:>6.1f} {s['replicas']:>9d} {s['gpus']:>6d} {s['daily_dollars']:>10,.0f} {s['cost_per_mtok']:>9.2f}")
    print()
    print(" COST")
    print(result["cost"].report())
    print("═" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(PRESETS), default="7b")
    ap.add_argument("--gpus", type=int, default=0, help="override default GPU count")
    ap.add_argument("--gpu-type", default="H100", choices=["A100", "H100", "H200", "B200"])
    ap.add_argument("--rlhf", default="dpo", choices=["none", "dpo", "ppo"])
    ap.add_argument("--sft-examples", type=int, default=250_000)
    ap.add_argument("--pref-pairs", type=int, default=200_000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = build(args)
    result = run_program(spec)
    report(result, spec)

    summary = {
        "size": args.size, "gpus": spec.pretrain_cluster.total_gpus,
        "gpu_type": args.gpu_type,
        "clock_days": result["clock_days"],
        "total_dollars": result["cost"].total,
        "cost_by_phase": result["cost"].by_phase,
        "cost_by_resource": result["cost"].by_resource,
        "final_loss": result["pretrain"]["final_loss"],
        "eval": result["eval"], "safety": result["safety"],
        "serving": result["serving"],
    }
    with open(os.path.join(spec.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f" wrote -> {spec.out_dir}/summary.json")
    print(f" wrote -> {spec.out_dir}/events.jsonl")


if __name__ == "__main__":
    main()
