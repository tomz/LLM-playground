"""Export midgpt checkpoints to HuggingFace GPT-2 format for downstream eval.

Why HuggingFace format
----------------------
`lm-evaluation-harness` and most academic benchmarks (MMLU, HellaSwag, ARC,
LAMBADA, etc.) consume models via HuggingFace's ``from_pretrained(...)``.
Rather than maintain a parallel eval-loading codebase, we serialise midgpt
weights into the on-disk layout ``transformers.GPT2LMHeadModel`` expects
and let HF's standard loaders do the rest. The exported directory is then
directly usable with::

    python lm_eval_runner.py --model out/hf_export --tasks hellaswag,lambada

Architecture mapping
--------------------
midgpt's GPT is bit-for-bit GPT-2: LayerNorm (pre-norm), learned position
embeddings, fused QKV projection, full causal self-attention, GELU MLP,
optional weight-tied embeddings.

Important wrinkles:

* HF GPT-2 uses ``Conv1D`` for ``c_attn``, ``c_proj``, ``c_fc`` — a layer
  that is mathematically identical to ``nn.Linear`` but **stores the weight
  transposed** (``[in, out]`` instead of ``[out, in]``). We transpose every
  linear weight on the way out (and the same on the way back in).
* HF GPT-2 always has biases. midgpt defaults to ``bias=False``. We write
  zero biases when the source model has none; reloading then preserves the
  no-bias forward pass exactly (zero added to anything = anything).
* Tied embeddings: GPT-2 ties via ``config.tie_word_embeddings=True``; the
  ``lm_head.weight`` is not stored separately in HF format. We mirror that.

What we do NOT support
----------------------
* ``qk_norm=True`` — stock GPT-2 has no QK-norm layer; raises.
* Liger fused-CE (``fused_ce=True``) — that's purely a training detail and
  doesn't affect weights; we ignore the flag.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch


def _hf_config(cfg) -> dict:
    """Build the ``config.json`` for ``GPT2LMHeadModel`` matching a GPTConfig."""
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": cfg.vocab_size,
        "n_positions": cfg.block_size,
        "n_ctx": cfg.block_size,
        "n_embd": cfg.d_model,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_inner": cfg.d_ffn,
        "activation_function": "gelu_new",   # tanh-approx GELU, matches midgpt
        "resid_pdrop": cfg.dropout,
        "embd_pdrop": cfg.dropout,
        "attn_pdrop": cfg.dropout,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
        "tie_word_embeddings": cfg.tie_embeddings,
        "bos_token_id": 50256,                # GPT-2 BPE: <|endoftext|>
        "eos_token_id": 50256,
        "torch_dtype": "float32",
        "transformers_version": "4.45.0",
    }


def _rename_state_dict(sd: dict[str, torch.Tensor], cfg) -> dict[str, torch.Tensor]:
    """Apply the midgpt → HF GPT-2 key rename + weight transposition.

    midgpt linear layers store ``[out_features, in_features]`` (PyTorch
    ``nn.Linear`` standard); HF GPT-2 uses ``Conv1D`` which stores
    ``[in_features, out_features]``. Every projection weight gets ``.t()``
    on the way out.

    When ``cfg.bias=False`` the source has no bias parameters but HF GPT-2
    always expects them; we write zero biases of the right shape and dtype.
    Reloading a zero-bias projection gives byte-identical numerics (the bias
    add is `x + 0` for every position).
    """
    out: dict[str, torch.Tensor] = {}

    def take(src: str, dst: str, *, transpose: bool = False) -> None:
        if src not in sd:
            raise KeyError(f"missing midgpt weight {src!r}; can't build HF state dict")
        t = sd[src]
        if transpose:
            t = t.t().contiguous()
        out[dst] = t

    def zero_like(dst: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        out[dst] = torch.zeros(shape, dtype=dtype)

    # Embeddings.
    take("tok_emb.weight", "transformer.wte.weight")
    take("pos_emb.weight", "transformer.wpe.weight")
    take("ln_f.weight",    "transformer.ln_f.weight")
    if cfg.bias:
        take("ln_f.bias",  "transformer.ln_f.bias")
    else:
        zero_like("transformer.ln_f.bias", sd["ln_f.weight"].shape,
                  sd["ln_f.weight"].dtype)

    for i in range(cfg.n_layer):
        # ln_1 / ln_2 are LayerNorm (weight + bias). bias may not exist.
        take(f"blocks.{i}.ln1.weight", f"transformer.h.{i}.ln_1.weight")
        take(f"blocks.{i}.ln2.weight", f"transformer.h.{i}.ln_2.weight")
        ln_dtype = sd[f"blocks.{i}.ln1.weight"].dtype
        ln_shape = sd[f"blocks.{i}.ln1.weight"].shape
        if cfg.bias:
            take(f"blocks.{i}.ln1.bias", f"transformer.h.{i}.ln_1.bias")
            take(f"blocks.{i}.ln2.bias", f"transformer.h.{i}.ln_2.bias")
        else:
            zero_like(f"transformer.h.{i}.ln_1.bias", ln_shape, ln_dtype)
            zero_like(f"transformer.h.{i}.ln_2.bias", ln_shape, ln_dtype)

        # Attention: fused QKV → c_attn (transposed to Conv1D layout).
        take(f"blocks.{i}.attn.qkv.weight",
             f"transformer.h.{i}.attn.c_attn.weight", transpose=True)
        take(f"blocks.{i}.attn.proj.weight",
             f"transformer.h.{i}.attn.c_proj.weight", transpose=True)
        c_attn_out = sd[f"blocks.{i}.attn.qkv.weight"].shape[0]   # 3 * d_model
        c_proj_out = sd[f"blocks.{i}.attn.proj.weight"].shape[0]  # d_model
        attn_dtype = sd[f"blocks.{i}.attn.qkv.weight"].dtype
        if cfg.bias:
            take(f"blocks.{i}.attn.qkv.bias",  f"transformer.h.{i}.attn.c_attn.bias")
            take(f"blocks.{i}.attn.proj.bias", f"transformer.h.{i}.attn.c_proj.bias")
        else:
            zero_like(f"transformer.h.{i}.attn.c_attn.bias",
                      (c_attn_out,), attn_dtype)
            zero_like(f"transformer.h.{i}.attn.c_proj.bias",
                      (c_proj_out,), attn_dtype)

        # MLP: c_fc + c_proj (both transposed to Conv1D layout).
        take(f"blocks.{i}.mlp.fc.weight",
             f"transformer.h.{i}.mlp.c_fc.weight", transpose=True)
        take(f"blocks.{i}.mlp.proj.weight",
             f"transformer.h.{i}.mlp.c_proj.weight", transpose=True)
        fc_out = sd[f"blocks.{i}.mlp.fc.weight"].shape[0]    # d_ffn
        pj_out = sd[f"blocks.{i}.mlp.proj.weight"].shape[0]  # d_model
        mlp_dtype = sd[f"blocks.{i}.mlp.fc.weight"].dtype
        if cfg.bias:
            take(f"blocks.{i}.mlp.fc.bias",   f"transformer.h.{i}.mlp.c_fc.bias")
            take(f"blocks.{i}.mlp.proj.bias", f"transformer.h.{i}.mlp.c_proj.bias")
        else:
            zero_like(f"transformer.h.{i}.mlp.c_fc.bias",   (fc_out,), mlp_dtype)
            zero_like(f"transformer.h.{i}.mlp.c_proj.bias", (pj_out,), mlp_dtype)

    # lm_head: ONLY emit a separate tensor when not tied. HF's
    # tie_word_embeddings=True will recreate the tie at load time.
    if not cfg.tie_embeddings:
        take("lm_head.weight", "lm_head.weight")
    return out


def export_to_hf(model, cfg, out_dir: str | Path,
                 *, tokenizer_name: str | None = "gpt2") -> Path:
    """Export ``model`` (a midgpt GPT instance) to a HuggingFace-format dir.

    Args:
      model: trained ``model.GPT`` instance on any device.
      cfg: the ``GPTConfig`` used to build it.
      out_dir: target directory; created if missing, overwritten if not.
      tokenizer_name: ``tiktoken`` name to fetch HF tokenizer files for. The
        default ``"gpt2"`` matches midgpt's only tokenizer. Pass None to skip
        (you'll need to copy tokenizer files in manually for HF eval to work).

    Returns:
      ``Path(out_dir)`` for convenient chaining into eval CLIs.

    Raises:
      ValueError if the model has features stock ``GPT2LMHeadModel`` can't
      represent (currently: ``qk_norm=True``).
      KeyError if the source state_dict is missing an expected key.
    """
    if getattr(cfg, "qk_norm", False):
        raise ValueError(
            "export_to_hf does not support qk_norm=True — stock "
            "GPT2LMHeadModel has no QK-norm layer. Retrain with "
            "qk_norm=False if you need HF eval."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    with open(out_dir / "config.json", "w") as f:
        json.dump(_hf_config(cfg), f, indent=2)

    # Weights. .detach().cpu() so we don't accidentally serialize CUDA tensors.
    sd_src = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    sd_hf = _rename_state_dict(sd_src, cfg)
    try:
        from safetensors.torch import save_file
        # safetensors refuses shared storage — but with tie_embeddings we've
        # already dropped the duplicate lm_head key.
        save_file(sd_hf, str(out_dir / "model.safetensors"))
    except ImportError:
        torch.save(sd_hf, str(out_dir / "pytorch_model.bin"))

    # Tokenizer (optional but convenient). HF GPT-2 tokenizer is in the hub.
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            tok.save_pretrained(str(out_dir))
        except Exception as e:                                              # noqa: BLE001
            # Network-down or offline-only env: skip silently, the export
            # without tokenizer is still loadable for weight-only checks.
            print(f"[export_to_hf] could not save tokenizer ({e!r}); skipping")

    # generation_config.json — sensible defaults for HF .generate().
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump({
            "bos_token_id": 50256, "eos_token_id": 50256,
            "do_sample": True, "temperature": 0.8, "top_p": 0.95,
            "transformers_version": "4.45.0",
        }, f, indent=2)

    return out_dir


def load_hf_state_into_midgpt(model, hf_dir: str | Path) -> None:
    """Inverse of :func:`export_to_hf`: load an HF GPT-2 ``state_dict`` into
    a midgpt ``GPT`` model in-place. Useful for continued pretraining.

    Inverts the rename table and re-transposes the Conv1D weights back to
    ``nn.Linear`` layout. With ``cfg.bias=False`` any non-zero HF bias is
    silently discarded (the model has no place to put it); a warning is
    printed so a user notices that's happening.
    """
    hf_dir = Path(hf_dir)
    safe = hf_dir / "model.safetensors"
    bin_ = hf_dir / "pytorch_model.bin"
    if safe.exists():
        from safetensors.torch import load_file
        sd_hf = load_file(str(safe))
    elif bin_.exists():
        sd_hf = torch.load(str(bin_), map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin in {hf_dir}")

    cfg = model.cfg
    n_layer = cfg.n_layer
    sd: dict[str, torch.Tensor] = {}

    def take(src: str, dst: str, *, transpose: bool = False) -> None:
        if src not in sd_hf:
            raise KeyError(f"missing HF weight {src!r}; can't import to midgpt")
        t = sd_hf[src]
        if transpose:
            t = t.t().contiguous()
        sd[dst] = t

    take("transformer.wte.weight", "tok_emb.weight")
    take("transformer.wpe.weight", "pos_emb.weight")
    take("transformer.ln_f.weight", "ln_f.weight")
    if cfg.bias:
        take("transformer.ln_f.bias", "ln_f.bias")

    nonzero_bias_dropped = 0
    for i in range(n_layer):
        take(f"transformer.h.{i}.ln_1.weight", f"blocks.{i}.ln1.weight")
        take(f"transformer.h.{i}.ln_2.weight", f"blocks.{i}.ln2.weight")
        if cfg.bias:
            take(f"transformer.h.{i}.ln_1.bias", f"blocks.{i}.ln1.bias")
            take(f"transformer.h.{i}.ln_2.bias", f"blocks.{i}.ln2.bias")
        else:
            for k in (f"transformer.h.{i}.ln_1.bias",
                      f"transformer.h.{i}.ln_2.bias"):
                if k in sd_hf and sd_hf[k].abs().sum() > 0:
                    nonzero_bias_dropped += 1
        take(f"transformer.h.{i}.attn.c_attn.weight",
             f"blocks.{i}.attn.qkv.weight", transpose=True)
        take(f"transformer.h.{i}.attn.c_proj.weight",
             f"blocks.{i}.attn.proj.weight", transpose=True)
        take(f"transformer.h.{i}.mlp.c_fc.weight",
             f"blocks.{i}.mlp.fc.weight", transpose=True)
        take(f"transformer.h.{i}.mlp.c_proj.weight",
             f"blocks.{i}.mlp.proj.weight", transpose=True)
        if cfg.bias:
            take(f"transformer.h.{i}.attn.c_attn.bias",
                 f"blocks.{i}.attn.qkv.bias")
            take(f"transformer.h.{i}.attn.c_proj.bias",
                 f"blocks.{i}.attn.proj.bias")
            take(f"transformer.h.{i}.mlp.c_fc.bias",
                 f"blocks.{i}.mlp.fc.bias")
            take(f"transformer.h.{i}.mlp.c_proj.bias",
                 f"blocks.{i}.mlp.proj.bias")
        else:
            for k in (f"transformer.h.{i}.attn.c_attn.bias",
                      f"transformer.h.{i}.attn.c_proj.bias",
                      f"transformer.h.{i}.mlp.c_fc.bias",
                      f"transformer.h.{i}.mlp.c_proj.bias"):
                if k in sd_hf and sd_hf[k].abs().sum() > 0:
                    nonzero_bias_dropped += 1

    # lm_head: tied means HF's file has no separate copy; recreate the tie
    # locally so strict=True doesn't complain.
    if "lm_head.weight" in sd_hf:
        sd["lm_head.weight"] = sd_hf["lm_head.weight"]
    else:
        sd["lm_head.weight"] = sd["tok_emb.weight"]

    if nonzero_bias_dropped:
        print(f"[load_hf_state_into_midgpt] WARNING: dropped "
              f"{nonzero_bias_dropped} non-zero HF bias tensor(s) because "
              f"cfg.bias=False. Forward pass will differ from the HF model.")

    model.load_state_dict(sd, strict=True)


def main():
    """CLI: midgpt ckpt → HF dir, plus optional smoke-roundtrip check."""
    import argparse
    from model import GPT, GPTConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="midgpt ckpt path")
    ap.add_argument("--out-dir", required=True, help="target HF directory")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer name to save alongside (or 'none')")
    ap.add_argument("--verify", action="store_true",
                    help="reload via transformers and assert logits match midgpt's")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**sd["cfg"]["model"])
    model = GPT(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in sd["model"].items()}
    model.load_state_dict(state)
    out = export_to_hf(model, cfg, args.out_dir,
                       tokenizer_name=None if args.tokenizer == "none" else args.tokenizer)
    print(f"[export_to_hf] wrote {out}")

    if args.verify:
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(str(out)).eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            mid_logits, _ = model(idx, return_full_logits=True)
            hf_logits = hf_model(idx).logits
        # Bigger tolerance because Conv1D transposition + fused QKV vs split
        # may accumulate in different float order, but for fp32 they should
        # match to ~1e-4 absolute.
        delta = (mid_logits - hf_logits).abs().max().item()
        print(f"[verify] max |Δlogits| = {delta:.3e}  (expect < 1e-3 fp32)")
        assert delta < 1e-3, f"export verify failed: {delta}"


if __name__ == "__main__":
    main()
