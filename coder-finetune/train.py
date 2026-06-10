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

from cf_dist import dist_env, placement_device_map, rank0_print  # noqa: E402

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_model_and_tokenizer_unsloth(cfg: dict):
    """Optional fast-path loader using Unsloth's custom autograd kernels.

    Gated behind `model.use_unsloth: true` in the YAML. Unsloth claims ~2x
    faster / ~70% less memory for single-GPU LoRA/QLoRA on exactly our model
    family (Qwen2.5-Coder), via hand-written fused kernels. It REPLACES the
    transformers loader + peft wrapping (it returns an already-PEFT'd model),
    so we keep it on a separate code path rather than threading flags through
    the standard one. The returned model/tokenizer plug straight into TRL's
    SFTTrainer just like the vanilla path.

    Trade-off vs the default path: heavier dependency, and it diverges from
    the repo's "just the standard HF plumbing" teaching goal — hence opt-in.
    Recommended mainly for the 7B QLoRA recipe where the wall-clock matters.
    """
    from unsloth import FastLanguageModel
    name = cfg["model"]["name"]
    dtype = DTYPES[cfg["model"]["dtype"]]
    method = cfg["method"]
    t = cfg.get("train", {})

    model, tok = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=t.get("max_seq_len", 1024),
        dtype=dtype,
        load_in_4bit=(method == "qlora"),
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if method in ("lora", "qlora"):
        lcfg = cfg["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            bias=lcfg["bias"], target_modules=lcfg["target_modules"],
            use_dora=bool(lcfg.get("use_dora", False)),
            use_rslora=bool(lcfg.get("use_rslora", False)),
            # Unsloth's own checkpointing variant; "unsloth" is the memory-
            # optimal setting, True/False also accepted.
            use_gradient_checkpointing=t.get("gradient_checkpointing", "unsloth") or False,
            random_state=cfg["seed"],
        )
    return model, tok


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
        # Under DDP, bitsandbytes places the 4-bit weights on a concrete device
        # at load time and accelerate cannot relocate them afterwards. Pin this
        # rank's quantized copy to its own local GPU; on a single process this
        # is None and Trainer/accelerate keeps doing the placement (unchanged).
        dmap = placement_device_map()
        if dmap is not None:
            kwargs["device_map"] = dmap

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if method == "qlora":
            # `prepare_model_for_kbit_training` already enables gradient
            # checkpointing internally — don't double-set it via SFTConfig.
            model = prepare_model_for_kbit_training(model)
        lcfg = cfg["lora"]
        # DoRA (weight-decomposed LoRA) and rsLoRA (rank-stabilized scaling)
        # are both pure-PEFT quality knobs that cost ~nothing to flip on:
        #   * use_dora=True   — decomposes each update into magnitude+direction;
        #     consistently beats plain LoRA at low rank (our r=16) for a small
        #     (~10-20%) step-time overhead. Not compatible with 4-bit QLoRA in
        #     older peft, so we gate it off for qlora unless explicitly forced.
        #   * use_rslora=True — scales adapters by alpha/sqrt(r) instead of
        #     alpha/r, which stops higher ranks from being effectively down-
        #     weighted into uselessness. Free; safe to leave on.
        use_dora = bool(lcfg.get("use_dora", False))
        use_rslora = bool(lcfg.get("use_rslora", False))
        if use_dora and method == "qlora":
            print("[lora] WARNING: DoRA + 4-bit QLoRA needs peft>=0.12 and is "
                  "slower; proceed only if your stack supports it.")
        peft_cfg = LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            bias=lcfg["bias"], target_modules=lcfg["target_modules"],
            use_dora=use_dora, use_rslora=use_rslora,
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

    # --- Throughput / quality knobs (all optional, default off for parity) ---
    # Liger Kernel: fused Triton RMSNorm/RoPE/SwiGLU + FusedLinearCrossEntropy.
    # ~20% faster, up to ~60% less memory, *exact* (not an approximation). The
    # fused-linear-CE alone is a big deal for Qwen2.5-Coder's ~150K vocab — it
    # avoids materializing the full [batch*seq, vocab] logits tensor, which is
    # the single largest activation in the forward pass. Buys longer context /
    # bigger batch on the same card. Requires `pip install liger-kernel` and a
    # Triton-capable GPU (no-op/raises on CPU, so we only flip it when asked).
    use_liger = bool(t.get("use_liger_kernel", False))
    # NEFTune: add uniform noise to embedding outputs during training only.
    # Consistently improves instruction-following with zero inference cost.
    # alpha ~5 is the published sweet spot; 0/None disables.
    neftune_alpha = t.get("neftune_noise_alpha", None)

    sft_kwargs = dict(
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
    if use_liger:
        sft_kwargs["use_liger_kernel"] = True
        print("[train] Liger Kernel enabled (fused RMSNorm/RoPE/SwiGLU/CE)")
    if neftune_alpha:
        sft_kwargs["neftune_noise_alpha"] = float(neftune_alpha)
        print(f"[train] NEFTune enabled (noise_alpha={neftune_alpha})")
    args = SFTConfig(**sft_kwargs)
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
    rank0_print(f"[load] {len(train_ds)} training examples")

    rank0_print(f"[load] model: {cfg['model']['name']} method={cfg['method']}")
    if cfg["model"].get("use_unsloth", False):
        rank0_print("[load] using Unsloth fast-path loader")
        model, tok = build_model_and_tokenizer_unsloth(cfg)
    else:
        model, tok = build_model_and_tokenizer(cfg)

    trainer = build_trainer(model, tok, train_ds, cfg)
    rank0_print("[train] starting")
    trainer.train()

    save_path = os.path.join(cfg["out_dir"], "final")
    trainer.save_model(save_path)            # TRL guards this to the main process
    if dist_env().is_main:                   # tokenizer save is not guarded — do it once
        tok.save_pretrained(save_path)
    rank0_print(f"[train] saved -> {save_path}")
    rank0_print(f"[train] to evaluate: python eval/run_humaneval.py --model {save_path}")

    if torch.cuda.is_available():
        # Report what the run actually cost. Useful for the worked-example
        # docs and for capacity planning on the next-bigger model. Each rank
        # tracks its own peak; rank 0's is the representative figure.
        alloc = torch.cuda.max_memory_allocated() // (1024 * 1024)
        resv = torch.cuda.max_memory_reserved() // (1024 * 1024)
        rank0_print(f"[vram] peak_alloc={alloc} MiB  peak_reserved={resv} MiB")


if __name__ == "__main__":
    main()
