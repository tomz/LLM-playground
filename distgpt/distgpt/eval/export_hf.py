"""Export distgpt checkpoints to HuggingFace format for downstream eval.

Why HuggingFace format
----------------------
`lm-evaluation-harness` and most academic benchmarks (MMLU, HellaSwag,
ARC, etc.) consume models via HuggingFace's `from_pretrained(...)`. Rather
than maintain a parallel eval-loading codebase, we serialise distgpt
weights into the on-disk layout `transformers.LlamaForCausalLM` expects
and let HF's standard loaders do the rest.

Architecture mapping
--------------------
distgpt's GPT model is structurally a Llama: pre-norm, RMSNorm, RoPE,
SwiGLU MLP (w1=gate, w3=up, w2=down), GQA. The weight-key rename table
is therefore mostly mechanical:

  distgpt key                 →  HF (LlamaForCausalLM) key
  -------                        ---
  tok_emb.weight              →  model.embed_tokens.weight
  layers.{i}.attn_norm.weight →  model.layers.{i}.input_layernorm.weight
  layers.{i}.attn.q_proj.W    →  model.layers.{i}.self_attn.q_proj.weight
  layers.{i}.attn.k_proj.W    →  model.layers.{i}.self_attn.k_proj.weight
  layers.{i}.attn.v_proj.W    →  model.layers.{i}.self_attn.v_proj.weight
  layers.{i}.attn.o_proj.W    →  model.layers.{i}.self_attn.o_proj.weight
  layers.{i}.ffn_norm.weight  →  model.layers.{i}.post_attention_layernorm.weight
  layers.{i}.ffn.w1.weight    →  model.layers.{i}.mlp.gate_proj.weight
  layers.{i}.ffn.w3.weight    →  model.layers.{i}.mlp.up_proj.weight
  layers.{i}.ffn.w2.weight    →  model.layers.{i}.mlp.down_proj.weight
  final_norm.weight           →  model.norm.weight
  lm_head.weight              →  lm_head.weight

The RoPE convention is identical (no rotation-order rewrite needed).
Tied embeddings are preserved through `config.tie_word_embeddings=True`.

What we deliberately do NOT support
-----------------------------------
* QK-Norm (cfg.qk_norm=True). Stock LlamaForCausalLM has no QK-norm
  layer, so the exported model wouldn't match the trained one. We raise
  a clear error rather than silently strip the norms. Future: export
  as `Qwen2ForCausalLM` which DOES have q_norm/k_norm.
* MoE / MLA / MTP — none of those are implemented in distgpt's GPT yet,
  so no mapping needed (this is a non-issue until they land).
"""
from __future__ import annotations
import json
from pathlib import Path

import torch


def _build_hf_config_dict(cfg) -> dict:
    """Build the HuggingFace `config.json` dict for a LlamaForCausalLM that
    matches the given distgpt ModelConfig.

    This is a static mapping — no introspection of the actual model
    object, so it round-trips correctly even when the model lives on a
    different rank during DCP loading.
    """
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_ffn,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "num_key_value_heads": cfg.n_kv_head,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_base,
        "rms_norm_eps": cfg.rms_eps,
        "tie_word_embeddings": cfg.tie_embeddings,
        "hidden_act": "silu",  # SwiGLU's activation
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.0",  # informational
    }


def _rename_state_dict(sd: dict[str, torch.Tensor],
                         n_layer: int) -> dict[str, torch.Tensor]:
    """Apply the distgpt→HF key-rename table to a flat state_dict.

    Iterates by layer index because the rename is per-layer (the table is
    too long to express as a single string substitution). Raises
    KeyError if an expected source key is missing — much better than a
    silent partial export that quietly drops weights.
    """
    out: dict[str, torch.Tensor] = {}

    def take(src: str, dst: str) -> None:
        if src not in sd:
            raise KeyError(
                f"missing distgpt weight {src!r}; can't build HF state dict"
            )
        out[dst] = sd[src]

    take("tok_emb.weight", "model.embed_tokens.weight")
    take("final_norm.weight", "model.norm.weight")
    # lm_head: in tied-embedding mode the lm_head weight equals tok_emb;
    # HF's tie_word_embeddings setting handles that, but we still write
    # the duplicate tensor so load_state_dict doesn't complain.
    take("lm_head.weight", "lm_head.weight")

    for i in range(n_layer):
        take(f"layers.{i}.attn_norm.weight",
              f"model.layers.{i}.input_layernorm.weight")
        take(f"layers.{i}.ffn_norm.weight",
              f"model.layers.{i}.post_attention_layernorm.weight")
        take(f"layers.{i}.attn.q_proj.weight",
              f"model.layers.{i}.self_attn.q_proj.weight")
        take(f"layers.{i}.attn.k_proj.weight",
              f"model.layers.{i}.self_attn.k_proj.weight")
        take(f"layers.{i}.attn.v_proj.weight",
              f"model.layers.{i}.self_attn.v_proj.weight")
        take(f"layers.{i}.attn.o_proj.weight",
              f"model.layers.{i}.self_attn.o_proj.weight")
        take(f"layers.{i}.ffn.w1.weight",
              f"model.layers.{i}.mlp.gate_proj.weight")
        take(f"layers.{i}.ffn.w3.weight",
              f"model.layers.{i}.mlp.up_proj.weight")
        take(f"layers.{i}.ffn.w2.weight",
              f"model.layers.{i}.mlp.down_proj.weight")
    return out


def export_to_hf(model, cfg, out_dir: str | Path,
                   *, tokenizer_files: dict[str, str] | None = None) -> Path:
    """Export `model` (a distgpt GPT instance) to a HuggingFace-format dir.

    Args:
      model: the trained distgpt.model.transformer.GPT instance, on any
        device. Weights are read via `state_dict()` and dtype-preserved.
      cfg: the ModelConfig used to build the model.
      out_dir: target directory; will be created. Existing files are
        overwritten without prompting (export should be re-runnable).
      tokenizer_files: optional `{filename: source_path}` map. If provided,
        each file is copied into out_dir alongside the weights so the HF
        loader picks up the tokenizer. The expected names are
        `tokenizer.json`, `tokenizer_config.json`, and optionally
        `special_tokens_map.json`.

    Returns:
      `Path(out_dir)` — handy for chaining into a subsequent
      `lm_eval` call.

    Raises:
      ValueError if the model has features stock LlamaForCausalLM can't
      represent (currently: qk_norm).
      KeyError if the model's state_dict is missing an expected key.
    """
    if getattr(cfg, "qk_norm", False):
        raise ValueError(
            "export_to_hf does not support qk_norm=True yet — stock "
            "LlamaForCausalLM has no QK-norm layer. Use the Qwen2 export "
            "path (TODO) or retrain with qk_norm=False for HF eval."
        )
    if getattr(cfg, "moe_enabled", False):
        # Stock LlamaForCausalLM is dense; an MoE export needs the Mixtral
        # or DeepSeek-V3 HF class with its own key layout and a router
        # state. Rather than silently strip the experts (which would
        # produce a tiny "first-expert-only" model with nonsense output)
        # we raise. The exact-error contract is pinned in tests/test_moe.py.
        raise NotImplementedError(
            "export_to_hf does not support MoE (moe_num_experts > 1) yet — "
            "the Mixtral / DeepSeek HF key layout is a separate exporter. "
            "Run dense (moe_num_experts=0) for HF eval until that lands."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    with open(out_dir / "config.json", "w") as f:
        json.dump(_build_hf_config_dict(cfg), f, indent=2)

    # 2. state dict. We save as pytorch_model.bin for portability; safetensors
    # is preferred if the user has the package, but importing it lazily so
    # `export_to_hf` doesn't pull a hard dep.
    sd_distgpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    sd_hf = _rename_state_dict(sd_distgpt, cfg.n_layer)
    # Tied embeddings: distgpt stores one tensor under both `tok_emb.weight`
    # and `lm_head.weight`, and our rename table promotes the duplicate into
    # `lm_head.weight` in the HF dict. safetensors refuses to write shared
    # storage twice (would be silently de-duped to one tensor on load),
    # which masks bugs. Drop the lm_head copy and rely on HF's
    # `tie_word_embeddings=True` flag (set in config.json) to recreate the
    # tie at load time — this is how `transformers` itself handles it.
    if cfg.tie_embeddings and (
        sd_hf["lm_head.weight"].data_ptr()
        == sd_hf["model.embed_tokens.weight"].data_ptr()
    ):
        del sd_hf["lm_head.weight"]
    try:
        from safetensors.torch import save_file
        save_file(sd_hf, str(out_dir / "model.safetensors"))
    except ImportError:
        torch.save(sd_hf, str(out_dir / "pytorch_model.bin"))

    # 3. tokenizer files (optional).
    if tokenizer_files:
        import shutil
        for name, src in tokenizer_files.items():
            shutil.copy(src, out_dir / name)

    # 4. generation_config.json — small, helpful when sampling via HF.
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump({
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
            "do_sample": True, "temperature": 0.8, "top_p": 0.95,
            "transformers_version": "4.45.0",
        }, f, indent=2)

    return out_dir


def load_hf_state_into_distgpt(model, hf_dir: str | Path) -> None:
    """Inverse of `export_to_hf`: load a HuggingFace LlamaForCausalLM
    state_dict (from `hf_dir`) into a distgpt GPT model in-place.

    Useful for continued pre-training of a base model published in HF
    format. We invert the rename table and call `load_state_dict(strict=True)`
    so the user notices missing or extra weights immediately.
    """
    hf_dir = Path(hf_dir)
    # Prefer safetensors when available — it's mmap-friendly and the
    # default for new HF checkpoints.
    safe = hf_dir / "model.safetensors"
    bin_ = hf_dir / "pytorch_model.bin"
    if safe.exists():
        from safetensors.torch import load_file
        sd_hf = load_file(str(safe))
    elif bin_.exists():
        sd_hf = torch.load(str(bin_), map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(
            f"no model.safetensors or pytorch_model.bin in {hf_dir}"
        )

    # Invert the rename table.
    n_layer = model.cfg.n_layer
    sd_distgpt: dict[str, torch.Tensor] = {}

    def take(src: str, dst: str) -> None:
        if src not in sd_hf:
            raise KeyError(f"missing HF weight {src!r}; can't import to distgpt")
        sd_distgpt[dst] = sd_hf[src]

    take("model.embed_tokens.weight", "tok_emb.weight")
    take("model.norm.weight", "final_norm.weight")
    # lm_head may be absent on disk when the source was tied-embedding (we
    # dropped the duplicate during export, see `export_to_hf`). Recreate
    # the tie locally rather than failing the load: copy from the embed
    # tensor so the in-memory state_dict has both keys for `strict=True`.
    if "lm_head.weight" in sd_hf:
        take("lm_head.weight", "lm_head.weight")
    else:
        sd_distgpt["lm_head.weight"] = sd_distgpt["tok_emb.weight"]
    for i in range(n_layer):
        take(f"model.layers.{i}.input_layernorm.weight",
              f"layers.{i}.attn_norm.weight")
        take(f"model.layers.{i}.post_attention_layernorm.weight",
              f"layers.{i}.ffn_norm.weight")
        take(f"model.layers.{i}.self_attn.q_proj.weight",
              f"layers.{i}.attn.q_proj.weight")
        take(f"model.layers.{i}.self_attn.k_proj.weight",
              f"layers.{i}.attn.k_proj.weight")
        take(f"model.layers.{i}.self_attn.v_proj.weight",
              f"layers.{i}.attn.v_proj.weight")
        take(f"model.layers.{i}.self_attn.o_proj.weight",
              f"layers.{i}.attn.o_proj.weight")
        take(f"model.layers.{i}.mlp.gate_proj.weight",
              f"layers.{i}.ffn.w1.weight")
        take(f"model.layers.{i}.mlp.up_proj.weight",
              f"layers.{i}.ffn.w3.weight")
        take(f"model.layers.{i}.mlp.down_proj.weight",
              f"layers.{i}.ffn.w2.weight")
    model.load_state_dict(sd_distgpt, strict=True)
