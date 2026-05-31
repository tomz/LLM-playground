"""DPO / ORPO offline preference optimization entry point.

The cheap, stable preference-alignment step that sits between SFT and online RL:

    python -m cf_pref.dpo_train --config configs/dpo_3050.yaml      # DPO (default)
    python -m cf_pref.dpo_train --config configs/orpo_3050.yaml     # ORPO (objective: orpo)

DPO (Rafailov et al., 2023) optimizes the policy to prefer ``chosen`` over
``rejected`` as a classification loss against a frozen reference copy of the
model — no reward model, no sampling. ORPO (Hong et al., 2024) folds the same
preference signal into SFT via an odds-ratio penalty and needs *no* reference
model. Both reuse the LoRA/QLoRA loader and PEFT plumbing from ``train.py``; with
a PEFT adapter, DPO's reference is the frozen base (adapters disabled), so no
second model copy is loaded.

NOTE on TRL versions: TRL 1.x removed the standalone ``ORPOTrainer``. We import
it lazily and raise a clear, actionable error if the objective is ``orpo`` but
the installed TRL doesn't ship it — DPO remains available everywhere.
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

from cf_pref import pairs as pref_data  # noqa: E402
from train import build_model_and_tokenizer, build_model_and_tokenizer_unsloth  # noqa: E402


def build_pref_trainer(model, tok, train_ds, cfg: dict):
    """Construct a DPO or ORPO trainer from the shared config schema.

    The ``pref.objective`` key selects the algorithm:
      * ``dpo``  (default) — DPOTrainer with a frozen reference (implicit for PEFT).
      * ``orpo`` — ORPOTrainer (reference-free); requires a TRL that ships it.

    ``model`` is already PEFT-wrapped by ``build_model_and_tokenizer`` for
    lora/qlora (same as the GRPO path), so we do NOT pass ``peft_config`` again
    — DPO detects the PEFT model and uses adapter-disabling as the reference, so
    no second full-size model copy is loaded.
    """
    p = cfg.get("pref", {})
    objective = p.get("objective", "dpo").lower()
    t = cfg["train"]
    method = cfg["method"]
    grad_ckpt = bool(t.get("gradient_checkpointing", method != "full"))

    # Fields shared by DPOConfig and ORPOConfig (both subclass TrainingArguments).
    common = dict(
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
        # bf16/fp16 are only valid with a capable GPU. transformers'
        # TrainingArguments.__post_init__ raises ValueError("...doesn't support
        # bf16/gpu") if you request bf16 on a CPU-only host — which broke CI
        # (GPU-less runner) and would equally break any real CPU smoke run.
        # Gate the request on actual hardware support so the config is valid
        # everywhere; on a real GPU run the dtype still selects bf16/fp16.
        bf16=(cfg["model"]["dtype"] == "bfloat16" and _bf16_supported()),
        fp16=(cfg["model"]["dtype"] == "float16" and torch.cuda.is_available()),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        report_to="none",
        seed=cfg["seed"],
        max_length=t.get("max_seq_len", 1024),
        max_prompt_length=p.get("max_prompt_length", 512),
        beta=p.get("beta", 0.1),
    )

    if objective == "dpo":
        from trl import DPOConfig, DPOTrainer
        # DPO supports a family of pairwise losses; sigmoid is the original DPO.
        common["loss_type"] = p.get("loss_type", "sigmoid")
        args = _make_config(DPOConfig, common)
        return DPOTrainer(
            model=model, ref_model=None, args=args,
            train_dataset=train_ds, processing_class=tok,
        )
    if objective == "orpo":
        try:
            from trl import ORPOConfig, ORPOTrainer
        except ImportError as e:
            raise SystemExit(
                "pref.objective='orpo' but this TRL build does not ship "
                "ORPOTrainer (removed in TRL 1.x). Pin `trl<0.12` to use ORPO, "
                "or use pref.objective='dpo' (recommended; available everywhere)."
            ) from e
        # ORPO is reference-free: beta acts as the odds-ratio (lambda) weight.
        common.pop("max_prompt_length", None)
        args = _make_config(ORPOConfig, common)
        return ORPOTrainer(
            model=model, args=args,
            train_dataset=train_ds, processing_class=tok,
        )
    raise ValueError(f"unknown pref.objective: {objective!r} (use 'dpo' or 'orpo')")


def _bf16_supported() -> bool:
    """True only if a CUDA GPU that actually supports bf16 is present.

    bf16 training requires Ampere+ (cc>=8.0). transformers validates this in
    TrainingArguments.__post_init__ and raises on a CPU-only or pre-Ampere
    host, so we must not request bf16 unless it's really available."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _make_config(config_cls, kwargs: dict):
    """Instantiate a TRL config, dropping keys this TRL version doesn't accept.

    TRL renames/removes config fields between minors (e.g. max_prompt_length);
    filtering to the dataclass's real fields keeps the entry point version-robust
    instead of crashing on an unknown kwarg.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(config_cls)}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return config_cls(**filtered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])

    objective = cfg.get("pref", {}).get("objective", "dpo").lower()
    print(f"[pref] objective={objective}  dataset: {cfg['dataset']['source']}")
    train_ds = pref_data.load(cfg["dataset"])
    if cfg["dataset"].get("max_examples"):
        train_ds = train_ds.select(range(min(cfg["dataset"]["max_examples"], len(train_ds))))
    print(f"[pref] {len(train_ds)} preference pairs (chosen vs rejected)")

    print(f"[pref] model: {cfg['model']['name']} method={cfg['method']}")
    if cfg["model"].get("use_unsloth", False):
        model, tok = build_model_and_tokenizer_unsloth(cfg)
    else:
        model, tok = build_model_and_tokenizer(cfg)

    trainer = build_pref_trainer(model, tok, train_ds, cfg)
    print(f"[pref] starting {objective.upper()}")
    trainer.train()

    save_path = os.path.join(cfg["out_dir"], "final")
    trainer.save_model(save_path)
    tok.save_pretrained(save_path)
    print(f"[pref] saved -> {save_path}")
    print(f"[pref] to evaluate: python eval/run_humaneval.py --model {save_path}")

    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() // (1024 * 1024)
        resv = torch.cuda.max_memory_reserved() // (1024 * 1024)
        print(f"[vram] peak_alloc={alloc} MiB  peak_reserved={resv} MiB")


if __name__ == "__main__":
    main()
