"""SFT / LoRA / QLoRA fine-tuning entry point.

Wraps HuggingFace TRL SFTTrainer; no custom training loop. The point is to
demonstrate the *plumbing*, not to reinvent it. Eval is split into a
separate CLI (`python eval/run_humaneval.py`) to keep this file pure-train.

Usage:
    python train.py --config configs/tiny.yaml
    python train.py --config configs/qlora.yaml
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import yaml
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_model_and_tokenizer(cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = cfg["model"]["name"]
    dtype = DTYPES[cfg["model"]["dtype"]]
    method = cfg["method"]

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg["model"]["trust_remote_code"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=cfg["model"]["attn_impl"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    if method == "qlora":
        from transformers import BitsAndBytesConfig
        q = cfg["qlora"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=DTYPES[q["bnb_4bit_compute_dtype"]],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        )

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if method == "qlora":
            # `prepare_model_for_kbit_training` already enables gradient
            # checkpointing internally — don't double-set it via SFTConfig.
            model = prepare_model_for_kbit_training(model)
        lcfg = cfg["lora"]
        peft_cfg = LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            bias=lcfg["bias"], target_modules=lcfg["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        # Plain LoRA + gradient-checkpointing gotcha: the input embedding
        # output has requires_grad=False, which breaks the autograd graph at
        # checkpoint boundaries. Calling enable_input_require_grads() turns on
        # a forward hook that flips that bit. prepare_model_for_kbit_training
        # already does this for QLoRA, so only do it for plain LoRA.
        if method == "lora" and cfg.get("train", {}).get("gradient_checkpointing", True):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        # use_cache=True is incompatible with gradient checkpointing.
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.print_trainable_parameters()

    return model, tok


def build_trainer(model, tok, train_ds, cfg: dict):
    from trl import SFTTrainer, SFTConfig
    t = cfg["train"]
    method = cfg["method"]
    # Only force gradient checkpointing for the largest recipes; QLoRA already
    # gets it from prepare_model_for_kbit_training, full FT of 0.5B doesn't
    # need it on a 3050.
    grad_ckpt = bool(t.get("gradient_checkpointing", method != "full"))
    args = SFTConfig(
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
        max_length=t["max_seq_len"],
        packing=t["packing"],
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        report_to="none",
        seed=cfg["seed"],
    )
    return SFTTrainer(
        model=model,
        processing_class=tok,
        args=args,
        train_dataset=train_ds,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])

    print(f"[load] dataset: {cfg['dataset']['source']}")
    # Import via the explicit subpackage name to avoid shadowing if someone
    # runs from a different cwd.
    import cf_data as data  # local package; see coder-finetune/cf_data/__init__.py
    train_ds = data.load(cfg["dataset"])
    if cfg["dataset"].get("max_examples"):
        train_ds = train_ds.select(range(min(cfg["dataset"]["max_examples"], len(train_ds))))
    print(f"[load] {len(train_ds)} training examples")

    print(f"[load] model: {cfg['model']['name']} method={cfg['method']}")
    model, tok = build_model_and_tokenizer(cfg)

    trainer = build_trainer(model, tok, train_ds, cfg)
    print("[train] starting")
    trainer.train()

    save_path = os.path.join(cfg["out_dir"], "final")
    trainer.save_model(save_path)
    tok.save_pretrained(save_path)
    print(f"[train] saved -> {save_path}")
    print(f"[train] to evaluate: python eval/run_humaneval.py --model {save_path}")


if __name__ == "__main__":
    main()
