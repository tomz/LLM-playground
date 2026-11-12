"""Emit an offline vLLM configuration and hash-bound provenance, NOT a converter.

Example: python infer/export_vllm.py --model out/merged --out out/serving-config
The source must be a local full/merged HF model with unquantized safetensors and
its tokenizer. Weights remain at that source path: this output is not a portable
model bundle. No model is loaded, no weights are cast or quantized, and no vLLM
runtime/GPU is exercised. Requested serving settings are NOT runtime validation.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_provenance import (  # noqa: E402
    environment_versions, hf_load_kwargs, reference_identity, tokenizer_identity, write_json,
)
from infer.merge_lora import (  # noqa: E402
    DTYPES, NOT_VLLM_VALIDATION, check_saved_tokenizer, new_output, output_identity,
    require_safe_model, source_manifests, staged_output,
)

LOGGER = logging.getLogger(__name__)


def export_config(model: str | Path, out: str | Path, *, dtype: str = "auto",
                  tensor_parallel_size: int = 1, max_model_len: int | None = None) -> dict:
    """Emit a config only, verifying any upstream merge manifest against its files."""
    model_path = Path(model).expanduser().resolve()
    output = new_output(out, model_path)
    if dtype not in (*DTYPES, "auto"):
        raise ValueError(f"dtype must be auto or one of {DTYPES}")
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be a positive integer")
    if max_model_len is not None and (type(max_model_len) is not int or max_model_len < 1):
        raise ValueError("max_model_len must be a positive integer")
    if not model_path.is_dir():
        raise ValueError("config export requires a local full/merged model directory")
    if (model_path / "adapter_config.json").exists():
        raise ValueError("merge the adapter first with infer/merge_lora.py")
    model_config = require_safe_model(model_path)
    if not (model_path / "tokenizer_config.json").is_file():
        raise ValueError("model must include its saved tokenizer; no remote fallback")
    model_ref = reference_identity(str(model_path))
    tok_ref = reference_identity(str(model_path), kind="tokenizer")
    manifests = source_manifests(model_path)
    upstream = manifests.get("export_manifest.json", {}).get("content")
    if upstream is not None:
        if upstream.get("schema") != "coder-finetune.merge/v1":
            raise ValueError("unsupported upstream export manifest schema")
        recorded = upstream.get("output", {})
        if (recorded.get("revision") != model_ref["revision"]
                or recorded.get("files") != model_ref["files"]):
            raise ValueError("source model hashes differ from the merge manifest")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path), **hf_load_kwargs(tok_ref))
    tok_identity = tokenizer_identity(tok, tok_ref)
    check_saved_tokenizer(tok_identity, manifests)
    effective_config = {
        "model": str(model_path), "tokenizer": str(model_path), "dtype": dtype,
        "trust_remote_code": False, "load_format": "safetensors",
        "tensor_parallel_size": tensor_parallel_size,
    }
    if max_model_len is not None:
        effective_config["max_model_len"] = max_model_len
    training = manifests.get("run_manifest.json", {}).get("content", {})
    base_ref = ((training.get("model") or {}).get("reference") or model_ref)
    if upstream is not None:
        base_ref = upstream["base_model"]
    with staged_output(output) as staging:
        write_json(staging / "vllm_config.json", effective_config)
        if reference_identity(str(model_path))["revision"] != model_ref["revision"]:
            raise ValueError("source model changed during config export")
        if source_manifests(model_path) != manifests:
            raise ValueError("source manifests changed during config export")
        evidence = {
            "schema": "coder-finetune.vllm-config/v1", "operation": "config-only",
            "scope": "Configuration emitter only; weights remain at source_model.name. "
                     "No copying, dtype conversion, quantization, or runtime support validation.",
            "source_model": model_ref, "base_model": base_ref,
            "adapter": upstream.get("adapter") if upstream is not None else None,
            "tokenizer": tok_identity, "dtype": dtype,
            "source_dtype_from_config": model_config.get("dtype", model_config.get("torch_dtype")),
            "effective_config": effective_config,
            "versions": environment_versions(), "source_manifests": manifests,
            "output": output_identity(staging, output),
            "validation": {"vllm": dict(NOT_VLLM_VALIDATION), "quantization_performed": False},
        }
        write_json(staging / "export_manifest.json", evidence)
    return evidence


def main(argv: list[str] | None = None) -> None:
    """Expose config-only settings without implying conversion or vLLM support."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local full/merged model and tokenizer directory")
    parser.add_argument("--out", required=True, help="new configuration directory; existing paths are rejected")
    parser.add_argument("--dtype", choices=("auto", *DTYPES), default="auto",
                        help="requested serving dtype only; does not convert weights")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="requested serving parallelism, not a validated runtime capability")
    parser.add_argument("--max-model-len", type=int,
                        help="requested serving context length; support is not validated")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        evidence = export_config(args.model, args.out, dtype=args.dtype,
                                 tensor_parallel_size=args.tensor_parallel_size,
                                 max_model_len=args.max_model_len)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    LOGGER.info("Config only -> %s; NOT real vLLM validation", evidence["output"]["name"])


if __name__ == "__main__":
    main()
