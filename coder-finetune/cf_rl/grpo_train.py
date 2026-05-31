"""GRPO / RLVR post-training entry point (verifiable code rewards).

Optimizes a (LoRA/QLoRA/full) code model against unit-test rewards using TRL's
GRPOTrainer — the DeepSeek-R1 recipe at consumer-GPU scale. Reuses the model
loader and PEFT plumbing from `train.py`, and the HumanEval subprocess sandbox
from `eval/run_humaneval.py` as the verifier.

Usage:
    python -m cf_rl.grpo_train --config configs/grpo_3050.yaml

The config mirrors `train.py`'s schema with an extra ``grpo:`` block. GRPO samples
``num_generations`` completions per prompt, scores each with the reward
function(s), standardizes within the group, and takes a clipped policy step.

SECURITY: GRPO executes model-generated code every step (it *is* the reward).
Run untrusted models inside Docker/gVisor — the subprocess guard is a floor.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_rl import prompts as grpo_data  # noqa: E402
from cf_rl.reward import (  # noqa: E402
    code_unit_test_reward,
    format_reward,
    soft_length_penalty,
)
from train import build_model_and_tokenizer, build_model_and_tokenizer_unsloth  # noqa: E402


def build_reward_funcs(cfg: dict, tokenizer):
    """Assemble the GRPO reward function list from the config.

    The unit-test verifier is the primary, correctness signal; format and length
    shaping are optional and default-off (parity with the frontier CompositeReward
    components). TRL sums the per-function rewards (optionally weighted).
    """
    g = cfg.get("grpo", {})
    funcs = [code_unit_test_reward]
    if g.get("format_reward", False):
        funcs.append(format_reward)
    if g.get("length_penalty", False):
        # Bind the tokenizer so the length penalty counts real tokens.
        def _len(prompts=None, completions=None, **kw):
            return soft_length_penalty(
                prompts, completions, tokenizer=tokenizer,
                target_tokens=g.get("length_target_tokens", 384),
                max_tokens=g.get("length_max_tokens", 1024),
                coef=g.get("length_coef", 0.1), **kw,
            )
        _len.__name__ = "soft_length_penalty"
        funcs.append(_len)
    return funcs


def grpo_extra_kwargs(cfg: dict) -> dict:
    """Optional GRPOConfig kwargs that are gated behind config flags (default
    off). Factored out as a pure function so it can be unit-tested without
    importing TRL or loading a model.

    ``grpo.use_vllm: true`` delegates rollout generation to vLLM via TRL's
    ``GRPOConfig(use_vllm=...)``. GRPO is generation-heavy (G samples/prompt
    every step), and vLLM's paged-attention + continuous batching makes the
    rollout phase ~3-8× faster on the same GPU — usually the single biggest
    GRPO speedup. It costs an extra ``vllm`` dependency and a one-time engine
    warm-up, so it stays opt-in. We pass it through only when truthy so older
    TRL builds that lack the kwarg aren't handed an unexpected ``False``.
    """
    g = cfg.get("grpo", {})
    extra: dict = {}
    if g.get("use_vllm", False):
        extra["use_vllm"] = True
        # Only thread the GPU memory fraction through when explicitly set —
        # leaving it unset lets TRL/vLLM pick its own default.
        if g.get("vllm_gpu_memory_utilization") is not None:
            extra["vllm_gpu_memory_utilization"] = float(
                g["vllm_gpu_memory_utilization"]
            )
    return extra


def build_grpo_trainer(model, tok, train_ds, reward_funcs, cfg: dict):
    from trl import GRPOConfig, GRPOTrainer

    t = cfg["train"]
    g = cfg.get("grpo", {})
    method = cfg["method"]
    grad_ckpt = bool(t.get("gradient_checkpointing", method != "full"))

    # GRPO divisibility check — fail fast with an actionable message.
    # TRL's GRPOTrainer requires the *effective* batch (per_device_batch_size
    # × grad_accum × world_size) to be divisible by num_generations, because
    # each step processes one group of G samples per prompt and groups can't
    # straddle accumulation boundaries. Without this guard the failure mode is
    # an opaque tensor-shape error deep inside trainer.train(), often after
    # several minutes of model loading and dataset prep.
    G = int(g.get("num_generations", 8))
    per_dev = int(t["batch_size"])
    accum = int(t["grad_accum"])
    world = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    effective = per_dev * accum * world
    if effective % G != 0:
        raise SystemExit(
            f"[grpo] config error: effective batch ({per_dev} × {accum} × world={world}"
            f" = {effective}) is not divisible by num_generations (G={G}). "
            f"Fix by adjusting one of: train.batch_size, train.grad_accum, "
            f"grpo.num_generations so they line up (e.g. G={G} -> effective batch "
            f"in {{{G}, {2*G}, {3*G}, ...}})."
        )

    args = GRPOConfig(
        output_dir=cfg["out_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["grad_clip"],
        logging_steps=t["log_every"],
        save_steps=t["save_every"],
        save_total_limit=2,
        bf16=(cfg["model"]["dtype"] == "bfloat16"),
        fp16=(cfg["model"]["dtype"] == "float16"),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        report_to="none",
        seed=cfg["seed"],
        # --- GRPO-specific ---
        num_generations=g.get("num_generations", 8),     # G samples per prompt
        max_prompt_length=g.get("max_prompt_length", 512),
        max_completion_length=g.get("max_completion_length", 512),
        temperature=g.get("temperature", 1.0),
        beta=g.get("beta", 0.04),                          # KL-to-reference coef
        num_iterations=g.get("num_iterations", 1),         # inner PPO epochs
        # Opt-in speed knobs (vLLM rollouts) threaded in only when configured,
        # so the default path is unchanged and old TRL builds aren't handed an
        # unknown kwarg.
        **grpo_extra_kwargs(cfg),
    )
    return GRPOTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=train_ds,
        reward_funcs=reward_funcs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])

    print(f"[grpo] dataset: {cfg['dataset']['source']}")
    train_ds = grpo_data.load(cfg["dataset"])
    if cfg["dataset"].get("max_examples"):
        train_ds = train_ds.select(range(min(cfg["dataset"]["max_examples"], len(train_ds))))
    print(f"[grpo] {len(train_ds)} prompts (each scored by unit tests)")

    print(f"[grpo] model: {cfg['model']['name']} method={cfg['method']}")
    if cfg["model"].get("use_unsloth", False):
        model, tok = build_model_and_tokenizer_unsloth(cfg)
    else:
        model, tok = build_model_and_tokenizer(cfg)

    reward_funcs = build_reward_funcs(cfg, tok)
    print(f"[grpo] rewards: {[f.__name__ for f in reward_funcs]}")

    trainer = build_grpo_trainer(model, tok, train_ds, reward_funcs, cfg)
    print("[grpo] starting RLVR")
    trainer.train()

    save_path = os.path.join(cfg["out_dir"], "final")
    trainer.save_model(save_path)
    tok.save_pretrained(save_path)
    print(f"[grpo] saved -> {save_path}")
    print(f"[grpo] to evaluate: python eval/run_humaneval.py --model {save_path}")

    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() // (1024 * 1024)
        resv = torch.cuda.max_memory_reserved() // (1024 * 1024)
        print(f"[vram] peak_alloc={alloc} MiB  peak_reserved={resv} MiB")


if __name__ == "__main__":
    main()
