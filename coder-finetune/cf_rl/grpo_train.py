"""GRPO / RLVR with an explicitly trusted local Docker code verifier.

Run: python -m cf_rl.grpo_train --config configs/grpo_3050.yaml
The execution backend is preflighted before datasets or model weights load.
Unassessed candidates abort training, never becoming fabricated zero rewards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_dist import dist_env, rank0_print
from cf_provenance import check_tested_stack, digest, strict_config, trainer_evidence, write_json
from cf_rl import prompts as grpo_data
from cf_rl.reward import code_unit_test_reward, format_reward, overlong_reward_shaping, soft_length_penalty
from eval.python_jail import DockerPythonExecutor, executor_from_config
from train import build_model_and_tokenizer, model_references, training_kwargs


def build_reward_funcs(cfg: dict, tokenizer, *, executor: DockerPythonExecutor | None = None,
                       verdict_sink=None) -> list:
    """Bind the operator-owned executor, not a backend supplied by dataset rows."""
    executor = executor if executor is not None else executor_from_config(cfg.get("execution"))
    executor.require_available()
    g = cfg.get("grpo", {})

    def unit_tests(prompts=None, completions=None, **kw):
        if {"executor", "verdict_sink"} & kw.keys():
            raise ValueError("dataset/reward arguments cannot override the execution policy")
        return code_unit_test_reward(prompts, completions, executor=executor, verdict_sink=verdict_sink, **kw)
    unit_tests.__name__ = "code_unit_test_reward"
    funcs = [unit_tests]
    if g.get("format_reward", False):
        funcs.append(format_reward)
    if g.get("length_penalty", False):
        def length(prompts=None, completions=None, **kw):
            return soft_length_penalty(
                prompts, completions, tokenizer=tokenizer,
                target_tokens=g.get("length_target_tokens", 384),
                max_tokens=g.get("length_max_tokens", 1024), coef=g.get("length_coef", 0.1), **kw)
        length.__name__ = "soft_length_penalty"
        funcs.append(length)
    if g.get("overlong_shaping", False):
        def overlong(prompts=None, completions=None, **kw):
            return overlong_reward_shaping(
                prompts, completions, tokenizer=tokenizer,
                max_tokens=g.get("overlong_max_tokens", g.get("max_completion_length", 512)),
                soft_window=g.get("overlong_soft_window", 128), coef=g.get("overlong_coef", 0.2), **kw)
        overlong.__name__ = "overlong_reward_shaping"
        funcs.append(overlong)
    return funcs


def grpo_extra_kwargs(cfg: dict) -> dict:
    """Explicit optional knobs; incompatible fields are rejected by strict_config."""
    g = cfg.get("grpo", {})
    extra = {}
    if g.get("use_vllm", False):
        extra["use_vllm"] = True
        if g.get("vllm_gpu_memory_utilization") is not None:
            extra["vllm_gpu_memory_utilization"] = float(g["vllm_gpu_memory_utilization"])
    elif g.get("vllm_gpu_memory_utilization") is not None:
        raise ValueError("vllm_gpu_memory_utilization requires use_vllm")
    if g.get("epsilon_high") is not None:
        extra["epsilon_high"] = float(g["epsilon_high"])
    return extra


def build_grpo_config(cfg: dict):
    """Fail on incompatible objectives or removed prompt-truncation settings."""
    from trl import GRPOConfig
    g, t = cfg.get("grpo", {}), cfg["train"]
    allowed = {"num_generations", "max_completion_length", "temperature", "beta", "num_iterations",
               "loss_type", "scale_rewards", "use_vllm", "vllm_gpu_memory_utilization", "epsilon_high",
               "format_reward", "length_penalty", "length_target_tokens", "length_max_tokens", "length_coef",
               "overlong_shaping", "overlong_max_tokens", "overlong_soft_window", "overlong_coef"}
    if set(g) - allowed:
        raise ValueError(f"unsupported grpo settings: {sorted(set(g) - allowed)}; prompt truncation is not implicit")
    generations = g.get("num_generations", 8)
    if type(generations) is not int or generations < 2:
        raise ValueError("num_generations must be an integer >= 2")
    per_dev, accum, world = t["batch_size"], t["grad_accum"], dist_env().world_size
    effective = per_dev * accum * world
    if effective % generations:
        raise SystemExit(
            f"[grpo] effective batch ({per_dev} × {accum} × world={world} = {effective}) "
            f"is not divisible by num_generations (G={generations}); adjust batch_size/grad_accum; world is WORLD_SIZE")
    kwargs = training_kwargs(cfg)
    kwargs.update(num_generations=generations, max_completion_length=g.get("max_completion_length", 512),
                  temperature=g.get("temperature", 1.0), beta=g.get("beta", 0.04),
                  num_iterations=g.get("num_iterations", 1), loss_type=g.get("loss_type", "grpo"),
                  scale_rewards=g.get("scale_rewards", "group"), **grpo_extra_kwargs(cfg))
    return strict_config(GRPOConfig, kwargs)


def build_grpo_trainer(model, tok, train_ds, reward_funcs, cfg: dict):
    """Use TRL's actual GRPO objective, not its evolving default loss type."""
    from trl import GRPOTrainer
    return GRPOTrainer(model=model, processing_class=tok, args=build_grpo_config(cfg),
                       train_dataset=train_ds, reward_funcs=reward_funcs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    # This MUST precede all model/dataset loading, including optional backends.
    executor = executor_from_config(cfg.get("execution"))
    executor.require_available()
    check_tested_stack()
    build_grpo_config(cfg)
    model_references(cfg)
    torch.manual_seed(cfg["seed"])
    train_ds = grpo_data.load(cfg["dataset"])
    model, tok = build_model_and_tokenizer(cfg)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # Each rank writes its own complete attempt stream; never concurrent writes
    # to rank zero's file, and never an artifact containing only successes.
    with (out / f"reward_attempts.rank-{dist_env().rank}.jsonl").open("w", encoding="utf-8") as stream:
        attempt = 0

        def save_verdict(record: dict) -> None:
            nonlocal attempt
            stream.write(json.dumps({"attempt": attempt, **record}, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            attempt += 1

        reward_funcs = build_reward_funcs(cfg, tok, executor=executor, verdict_sink=save_verdict)
        trainer = build_grpo_trainer(model, tok, train_ds, reward_funcs, cfg)
        evidence = trainer_evidence(trainer, cfg, objective="grpo", data={
            "rows_sha256": digest(list(train_ds)), "n": len(train_ds)})
        evidence["evaluator"] = executor.identity()
        if dist_env().is_main:
            write_json(out / "run_manifest.json", evidence)
        trainer.train()
        final = out / "final"
        trainer.save_model(str(final))
        if dist_env().is_main:
            tok.save_pretrained(final)
            write_json(final / "run_manifest.json", evidence)
        rank0_print(f"[grpo] saved -> {final}; evaluation requires --python-image")


if __name__ == "__main__":
    main()
