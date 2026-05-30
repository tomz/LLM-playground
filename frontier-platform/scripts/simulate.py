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
from platform.sim.reasoning_rl_sim import ReasoningRLSpec
from platform.sim.serving_sim import ServingTier
from platform.sim.orchestrator import ProgramSpec, run_program
from platform.sim.gpu_probe import (
    have_cuda, probe_all, register_in_gpu_specs, format_probe_report,
)
from platform.sim.real_train import measure_real_throughput, format_real_train_report


PRESETS = {
    # name: (n_params, tokens, seq_len, global_batch_tokens, default_gpus)
    "1b":   (1.2e9,  1.0e12, 4096, 1_000_000,   64),
    "7b":   (6.7e9,  2.0e12, 4096, 4_000_000,  512),
    "70b":  (7.0e10, 5.0e12, 4096, 8_000_000, 4096),
    "400b": (4.0e11, 1.5e13, 4096,16_000_000, 16384),
    # Frontier-class targets we don't have the hardware to actually run.
    # 1T-total sparse MoE (DeepSeek-V3-class active fraction via --moe-experts).
    "1t":   (1.0e12, 2.0e13, 8192,32_000_000, 32768),
    # 2T-total "GPT-5.x-class" envelope; pair with --gpu-type GB200 --precision fp8.
    "2t":   (2.0e12, 3.0e13, 8192,48_000_000, 65536),
}


def build(args, real_calibration=None) -> ProgramSpec:
    n_params, tokens, seq, gbt, default_gpus = PRESETS[args.size]
    gpus = args.gpus or default_gpus
    gpus_per_node = 8
    nodes = max(1, gpus // gpus_per_node)
    gpu_type = args.gpu_type
    # If the user passed --real-gpu and we measured something, optionally
    # pretend the simulated cluster is built from that local SKU.
    if args.use_local_gpu_type and real_calibration is not None:
        gpu_type = real_calibration["gpu_type_key"]
    pretrain_cluster = Cluster(name=f"{args.size}_pretrain", n_nodes=nodes,
                               gpus_per_node=gpus_per_node, gpu_type=gpu_type)
    eval_cluster = Cluster(name=f"{args.size}_eval", n_nodes=8, gpu_type=gpu_type)

    alignment = AlignmentSpec(
        sft_examples=args.sft_examples,
        pref_pairs=args.pref_pairs,
        rlhf=args.rlhf,
    )

    reasoning_rl = ReasoningRLSpec(
        enabled=args.reasoning_rl,
        prompts=args.rl_prompts,
        group_size=args.rl_group_size,
        steps=args.rl_steps,
        avg_response_tokens=args.rl_response_tokens,
    )

    serving_tiers = [
        ServingTier("nano", n_params=1.2e9,  quant="int4", gpu="A100", gpus_per_replica=1,
                    target_throughput_tok_s=4000, ttft_ms=50),
        ServingTier("mid",  n_params=6.7e9,  quant="fp8",  gpu=args.gpu_type, gpus_per_replica=1,
                    target_throughput_tok_s=2500, ttft_ms=100,
                    spec_decode=("mtp" if args.spec_decode else "none"),
                    spec_draft_len=2, spec_accept_rate=0.75),
        # pro/max are the long-context, frontier-scale tiers where MLA's KV
        # compression and speculative decoding pay off the most (docs/14 §3).
        ServingTier("pro",  n_params=n_params if n_params <= 7e10 else 7e10,
                    quant="fp8", gpu=args.gpu_type, gpus_per_replica=8,
                    target_throughput_tok_s=1800, ttft_ms=250,
                    attn_kind=("mla" if args.mla_serving else "gqa"), kv_compression=4.0,
                    spec_decode=("eagle" if args.spec_decode else "none"),
                    spec_draft_len=4, spec_accept_rate=0.8),
        ServingTier("max",  n_params=n_params, quant="fp8", gpu=args.gpu_type, gpus_per_replica=64,
                    target_throughput_tok_s=900, ttft_ms=600,
                    attn_kind=("mla" if args.mla_serving else "gqa"), kv_compression=4.0,
                    spec_decode=("eagle" if args.spec_decode else "none"),
                    spec_draft_len=4, spec_accept_rate=0.8),
    ]
    qps = {"nano": 80.0, "mid": 30.0, "pro": 8.0, "max": 1.5}

    return ProgramSpec(
        name=args.size, n_params=n_params, total_tokens=tokens,
        seq_len=seq, global_batch_tokens=gbt,
        pretrain_cluster=pretrain_cluster, eval_cluster=eval_cluster,
        alignment=alignment, serving_tiers=serving_tiers, serving_qps=qps,
        moe_num_experts=args.moe_experts, moe_top_k=args.moe_top_k,
        precision=args.precision,
        reasoning_rl=reasoning_rl,
        out_dir=args.out_dir or f"out/sim/{args.size}",
        seed=args.seed,
        measured_tflops_per_gpu=(
            real_calibration["per_gpu_tflops"] if real_calibration else None
        ),
        measured_source=(real_calibration["source"] if real_calibration else None),
    )


def report(result: dict, spec: ProgramSpec) -> None:
    print()
    print("═" * 78)
    print(f" FRONTIER PROGRAM SIMULATION  —  {spec.name.upper()}")
    print("═" * 78)
    print(f" model:        {spec.n_params/1e9:>8.1f} B params"
          + (f"  (MoE: {spec.moe_num_experts} experts, top-{spec.moe_top_k} → "
             f"{spec.active_params/1e9:.1f} B active)" if spec.moe_num_experts > 1 else " (dense)"))
    print(f" tokens:       {spec.total_tokens/1e12:>8.2f} T")
    print(f" precision:    {spec.precision:>8s}")
    print(f" cluster:      {spec.pretrain_cluster.total_gpus:>8d} × {spec.pretrain_cluster.gpu_type}")
    print(f" wall-clock:   {result['clock_days']:>8.1f} days  ({result['clock_days']/30:.1f} months)")
    print()
    pre = result["pretrain"]
    print(f" PRETRAIN: {pre['total_steps']:,} steps  final-loss {pre['final_loss']:.3f}  spikes {pre['spikes']}  GPU-failures {pre['failures']}")
    al = result["alignment"]
    print(f" ALIGN:    sft_quality {al['sft_quality']:.3f}  rlhf_quality {al['rlhf_quality']:.3f}")
    rl = result.get("reasoning_rl", {})
    if rl.get("rl_compute_flops", 0) > 0:
        from platform.sim.scaling import compute_flops
        pretrain_flops = compute_flops(spec.active_params, spec.total_tokens)
        pct = rl["rl_compute_flops"] / max(1.0, pretrain_flops) * 100
        print(f" REASON-RL: reasoning_quality {rl['reasoning_quality']:.3f}  "
              f"RL-compute {rl['rl_compute_flops']:.2e} FLOPs ({pct:.2f}% of pretrain)  "
              f"wall-hrs {rl.get('gpu_hours', 0):,.2f}  ${rl.get('compute_dollars', 0):,.0f}")
    ev = result["eval"]
    print(f" EVAL:     MMLU {ev['mmlu']*100:5.1f}%   HumanEval {ev['humaneval']*100:5.1f}%   GSM8K {ev['gsm8k']*100:5.1f}%   ELO {ev['arena_elo']:6.0f}")
    sf = result["safety"]
    print(f" SAFETY:   {sf['verdict']}" + (f"   failed: {','.join(sf['failed'])}" if sf['failed'] else ""))
    for k, v in sf["scores"].items():
        print(f"             {k:12s} {v:.3f}")
    print()
    print(" SERVING (24h projection):")
    print(f"   {'tier':<6s} {'qps':>6s} {'replicas':>9s} {'gpus':>6s} {'$/day':>10s} {'$/Mtok':>9s} {'attn':>5s} {'spec':>6s}")
    for name, s in result["serving"].items():
        spec_tag = (f"{s.get('spec_throughput_mult', 1.0):.2f}x"
                    if s.get("spec_decode", "none") != "none" else "—")
        print(f"   {name:<6s} {s['qps']:>6.1f} {s['replicas']:>9d} {s['gpus']:>6d} "
              f"{s['daily_dollars']:>10,.0f} {s['cost_per_mtok']:>9.2f} "
              f"{s.get('attn_kind', 'gqa'):>5s} {spec_tag:>6s}")
    print()
    print(" COST")
    print(result["cost"].report())
    print("═" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(PRESETS), default="7b")
    ap.add_argument("--gpus", type=int, default=0, help="override default GPU count")
    ap.add_argument("--gpu-type", default="H100",
                    choices=["A100", "H100", "H200", "B200", "GB200", "B300"])
    ap.add_argument("--rlhf", default="dpo", choices=["none", "dpo", "ppo"])
    ap.add_argument("--sft-examples", type=int, default=250_000)
    ap.add_argument("--pref-pairs", type=int, default=200_000)
    # --- sparse MoE ---
    ap.add_argument("--moe-experts", type=int, default=0,
                    help="total experts (0 = dense). Drives active-param FLOPs.")
    ap.add_argument("--moe-top-k", type=int, default=2)
    # --- training precision ---
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "nvfp4"],
                    help="training numeric format; fp8/nvfp4 speed up pretrain")
    # --- serving economics ---
    ap.add_argument("--mla-serving", action="store_true",
                    help="use Multi-head Latent Attention KV compression on the "
                         "pro/max long-context tiers (fewer GPUs, lower $/Mtok)")
    ap.add_argument("--spec-decode", action="store_true",
                    help="enable speculative decoding (MTP/EAGLE draft heads) on "
                         "the mid/pro/max tiers — lossless decode throughput uplift")
    # --- reasoning RL (RLVR/GRPO) ---
    ap.add_argument("--reasoning-rl", action="store_true",
                    help="add an RLVR/GRPO reasoning post-training phase")
    ap.add_argument("--rl-prompts", type=int, default=100_000)
    ap.add_argument("--rl-group-size", type=int, default=8)
    ap.add_argument("--rl-steps", type=int, default=1_000)
    ap.add_argument("--rl-response-tokens", type=int, default=4_000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-gpu", action="store_true",
                    help="probe local CUDA GPUs and calibrate the simulator "
                         "from a few real training steps")
    ap.add_argument("--real-gpu-index", type=int, default=0,
                    help="which local CUDA device to use for the calibration run")
    ap.add_argument("--use-local-gpu-type", action="store_true",
                    help="replace --gpu-type with the locally-measured SKU for pricing")
    ap.add_argument("--real-steps", type=int, default=6,
                    help="number of timed steps in the calibration run")
    args = ap.parse_args()

    real_calibration = None
    probes = []
    real_result = None
    if args.real_gpu:
        if not have_cuda():
            print(" --real-gpu requested but no CUDA device is visible; "
                  "falling back to pure simulation.")
        else:
            probes = probe_all()
            register_in_gpu_specs(probes)
            real_result = measure_real_throughput(
                device_index=args.real_gpu_index,
                measure_steps=args.real_steps,
            )
            if real_result is not None:
                # Per-GPU achieved TFLOP/s is the calibration knob (roughly
                # model-size invariant, unlike raw tok/s).
                per_gpu_tflops = real_result.achieved_tflops
                # find the synthetic GPU_SPECS key for the device we used
                probe = probes[args.real_gpu_index] if args.real_gpu_index < len(probes) else None
                gpu_type_key = probe.synthetic_gpu_key if probe else args.gpu_type
                real_calibration = {
                    "per_gpu_tflops": per_gpu_tflops,
                    "source": f"{real_result.device} {real_result.dtype}",
                    "gpu_type_key": gpu_type_key,
                }

    spec = build(args, real_calibration=real_calibration)
    result = run_program(spec)
    report(result, spec)

    if probes:
        print(format_probe_report(probes))
        print()
    if real_result is not None:
        print(format_real_train_report(real_result))
        print()

    summary = {
        "size": args.size, "gpus": spec.pretrain_cluster.total_gpus,
        "gpu_type": spec.pretrain_cluster.gpu_type,
        "n_params": spec.n_params, "active_params": spec.active_params,
        "moe_experts": spec.moe_num_experts, "precision": spec.precision,
        "clock_days": result["clock_days"],
        "total_dollars": result["cost"].total,
        "cost_by_phase": result["cost"].by_phase,
        "cost_by_resource": result["cost"].by_resource,
        "final_loss": result["pretrain"]["final_loss"],
        "reasoning_rl": result.get("reasoning_rl"),
        "eval": result["eval"], "safety": result["safety"],
        "serving": result["serving"],
        "real_gpu": {
            "probes": [p.as_dict() for p in probes],
            "calibration": (real_result.as_dict() if real_result else None),
        },
    }
    with open(os.path.join(spec.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f" wrote -> {spec.out_dir}/summary.json")
    print(f" wrote -> {spec.out_dir}/events.jsonl")


if __name__ == "__main__":
    main()
