"""SFT / LoRA / QLoRA with pinned inputs and an explicit supervision contract.

Run: python train.py --config configs/tiny.yaml
Evaluation is separate: eval/run_humaneval.py requires a trusted local image.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_dist import dist_env, placement_device_map
from cf_provenance import (
    check_tested_stack, hf_load_kwargs, reference_identity, strict_config,
    tokenizer_identity, trainer_evidence, write_json,
)

LOGGER = logging.getLogger(__name__)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def model_references(cfg: dict) -> tuple[dict, dict]:
    """Validate immutable references before any from_pretrained call."""
    m = cfg["model"]
    if m.get("trust_remote_code", False):
        raise ValueError("trust_remote_code is unsupported in the tested training profile")
    if m.get("use_unsloth", False):
        raise ValueError("Unsloth is outside the tested pinned HF profile; use the standard loader")
    if cfg["method"] not in {"full", "lora", "qlora"}:
        raise ValueError("method must be full, lora or qlora")
    model_ref = reference_identity(m["name"], m.get("revision"))
    tok_name = m.get("tokenizer_name", m["name"])
    tok_revision = m.get("tokenizer_revision", m.get("revision") if tok_name == m["name"] else None)
    tok_ref = reference_identity(tok_name, tok_revision, kind="tokenizer")
    return model_ref, tok_ref


def build_tokenizer(cfg: dict):
    """Load only the pinned tokenizer, enabling template checks before weights."""
    from transformers import AutoTokenizer
    _, reference = model_references(cfg)
    tok = AutoTokenizer.from_pretrained(reference["name"], **hf_load_kwargs(
        reference, local_files_only=cfg["model"].get("local_files_only", True)))
    if tok.eos_token_id is None:
        raise ValueError("the tokenizer needs an explicit EOS token")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model_and_tokenizer_unsloth(cfg: dict):
    """Reject an untested alternative instead of silently changing the backend."""
    raise ValueError("Unsloth is outside the tested pinned HF profile; use the standard loader")


def build_model_and_tokenizer(cfg: dict, *, tokenizer=None):
    """Load safe weights and preserve the base revision in saved PEFT adapters."""
    from transformers import AutoModelForCausalLM
    model_ref, tok_ref = model_references(cfg)
    m, method = cfg["model"], cfg["method"]
    tok = tokenizer if tokenizer is not None else build_tokenizer(cfg)
    kwargs = dict(dtype=DTYPES[m["dtype"]], attn_implementation=m["attn_impl"],
                  use_safetensors=True,
                  **hf_load_kwargs(model_ref, local_files_only=m.get("local_files_only", True)))
    if method == "qlora":
        if not torch.cuda.is_available():
            raise ValueError("QLoRA requires the optional bitsandbytes GPU profile")
        from transformers import BitsAndBytesConfig
        q = cfg["qlora"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=DTYPES[q["bnb_4bit_compute_dtype"]],
            bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"])
        dmap = placement_device_map()
        if dmap is not None:
            kwargs["device_map"] = dmap
    model = AutoModelForCausalLM.from_pretrained(m["name"], **kwargs)
    if method in {"lora", "qlora"}:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if method == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True))
        lcfg = cfg["lora"]
        peft_cfg = LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            bias=lcfg["bias"], target_modules=lcfg["target_modules"],
            use_dora=bool(lcfg.get("use_dora", False)), use_rslora=bool(lcfg.get("use_rslora", False)),
            task_type="CAUSAL_LM", revision=m.get("revision"))
        model = get_peft_model(model, peft_cfg)
        if method == "lora" and cfg["train"].get("gradient_checkpointing", True):
            model.enable_input_require_grads()
        model.config.use_cache = False
    model._cf_provenance = {"reference": model_ref, "tokenizer": tokenizer_identity(tok, tok_ref),
                            "method": method, "dtype": m["dtype"], "attn_impl": m["attn_impl"]}
    return model, tok


def training_kwargs(cfg: dict) -> dict:
    """Shared effective arguments; precision is never silently downgraded."""
    t = cfg["train"]
    dtype = cfg["model"]["dtype"]
    if dtype not in DTYPES:
        raise ValueError(f"unsupported dtype: {dtype}")
    if dtype == "float16" and not torch.cuda.is_available():
        raise ValueError("float16 training requires CUDA; explicitly choose float32 for CPU")
    grad_ckpt = bool(t.get("gradient_checkpointing", cfg["method"] != "full"))
    return dict(
        output_dir=cfg["out_dir"], num_train_epochs=t["epochs"], max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["batch_size"], per_device_eval_batch_size=t.get("eval_batch_size", t["batch_size"]),
        gradient_accumulation_steps=t["grad_accum"], learning_rate=t["lr"],
        warmup_ratio=t["warmup_ratio"], weight_decay=t["weight_decay"], max_grad_norm=t["grad_clip"],
        logging_steps=t["log_every"], save_steps=t["save_every"], save_total_limit=2,
        bf16=(dtype == "bfloat16"), fp16=(dtype == "float16"),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        report_to="none", seed=cfg["seed"], data_seed=cfg["seed"],
        use_cpu=t.get("use_cpu", False), dataloader_pin_memory=torch.cuda.is_available(),
    )


def build_sft_config(cfg: dict):
    """Validate the objective independently of model loading."""
    from trl import SFTConfig
    t = cfg["train"]
    if (cfg.get("eval") or {}).get("enable", False):
        raise ValueError("legacy in-process code evaluation is disabled; use the isolated evaluation CLI")
    if t.get("loss_scope") not in {"all_tokens", "assistant_only"}:
        raise ValueError("train.loss_scope must explicitly be all_tokens or assistant_only")
    if t.get("packing", False):
        # BFD packing in TRL is padding-free and requires boundary-aware flash
        # attention. Do not silently use cross-example causal SDPA attention.
        if cfg["model"].get("attn_impl") not in {"flash_attention_2", "flash_attention_3"}:
            raise ValueError("packing requires boundary-aware flash attention; disable packing for SDPA/CPU")
    kwargs = training_kwargs(cfg)
    kwargs.update(max_length=t["max_seq_len"], packing=t.get("packing", False),
                  assistant_only_loss=t["loss_scope"] == "assistant_only", completion_only_loss=False,
                  eval_strategy="steps", eval_steps=t.get("eval_every", t["log_every"]),
                  use_liger_kernel=t.get("use_liger_kernel", False),
                  neftune_noise_alpha=t.get("neftune_noise_alpha"))
    return strict_config(SFTConfig, kwargs)


def validate_supervision(tok, datasets: tuple, cfg: dict) -> None:
    """Fail early if a template cannot yield the requested, nonempty loss masks."""
    scope = cfg["train"].get("loss_scope")
    if scope not in {"all_tokens", "assistant_only"}:
        raise ValueError("train.loss_scope must explicitly be all_tokens or assistant_only")
    template = tok.get_chat_template()
    if scope == "assistant_only" and not re.search(r"\{%[-+]?\s*generation\s*[-+]?%\}", template):
        raise ValueError("assistant_only requires a chat template with {% generation %} markers")
    for ds in datasets:
        if ds is None or len(ds) == 0:
            raise ValueError("SFT requires nonempty, separate train and validation datasets")
        for row in ds:
            encoded = tok.apply_chat_template(
                row["messages"], tokenize=True, return_dict=True,
                return_assistant_tokens_mask=(scope == "assistant_only"),
                truncation=True, max_length=cfg["train"]["max_seq_len"], add_generation_prompt=False)
            if len(encoded["input_ids"]) < 2:
                raise ValueError("SFT example has no next-token targets after truncation")
            if scope == "assistant_only" and not any(encoded.get("assistant_masks", [])[1:]):
                raise ValueError("assistant_only example has no assistant targets after truncation")


def build_trainer(model, tok, train_ds, cfg: dict, *, eval_ds=None):
    """Construct real TRL SFT with an explicit objective and held-out validation."""
    from trl import SFTTrainer
    args = build_sft_config(cfg)
    validate_supervision(tok, (train_ds, eval_ds), cfg)
    # Metadata is retained in the dataset/split manifest, not passed to a model
    # or a collator that could accidentally interpret an old 'labels' column.
    return SFTTrainer(model=model, processing_class=tok, args=args,
                      train_dataset=train_ds.select_columns(["messages"]),
                      eval_dataset=eval_ds.select_columns(["messages"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    check_tested_stack()
    build_sft_config(cfg)
    model_references(cfg)
    torch.manual_seed(cfg["seed"])
    import cf_data as data
    ds = data.load(cfg["dataset"])
    train_ds, eval_ds, manifest = data.split_dataset(
        ds, validation_fraction=cfg["dataset"].get("validation_fraction", 0.1),
        seed=cfg["dataset"].get("split_seed", cfg["seed"]), group_by=cfg["dataset"].get("group_by"))
    tok = build_tokenizer(cfg)
    validate_supervision(tok, (train_ds, eval_ds), cfg)
    model, tok = build_model_and_tokenizer(cfg, tokenizer=tok)
    trainer = build_trainer(model, tok, train_ds, cfg, eval_ds=eval_ds)
    if dist_env().is_main:
        write_json(Path(cfg["out_dir"]) / "split_manifest.json", manifest)
        write_json(Path(cfg["out_dir"]) / "run_manifest.json",
                   trainer_evidence(trainer, cfg, objective="sft", data=manifest))
    trainer.train()
    save_path = Path(cfg["out_dir"]) / "final"
    trainer.save_model(str(save_path))
    if dist_env().is_main:
        tok.save_pretrained(save_path)
        write_json(save_path / "run_manifest.json", trainer_evidence(trainer, cfg, objective="sft", data=manifest))
        LOGGER.info("Saved %s; evaluate separately with an explicit --python-image", save_path)


if __name__ == "__main__":
    main()
