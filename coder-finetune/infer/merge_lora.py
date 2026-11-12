"""Merge a local LoRA adapter into pinned, safetensors-only HF weights on CPU.

Example: python infer/merge_lora.py --adapter out/lora/final --out out/merged
Legacy remote bases require --revision <40-character commit>; --base can select
an explicitly pinned replacement or a local directory identified by its bytes.
Inputs are offline-only. No remote Python code, quantization, or vLLM validation
is performed. Existing outputs (including empty directories) are never reused.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import errno
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_provenance import (  # noqa: E402
    environment_versions, file_sha256, hf_load_kwargs, reference_identity,
    require_revision, tokenizer_identity, write_json,
)

LOGGER = logging.getLogger(__name__)
DTYPES = ("float32", "bfloat16", "float16")
TOKENIZER_FIELDS = ("vocab_sha256", "vocab_size", "chat_template_sha256",
                    "special_tokens", "eos_token_id", "pad_token_id")
NOT_VLLM_VALIDATION = {
    "validated": False,
    "statement": "NOT real vLLM validation: no vLLM runtime was loaded or exercised.",
}


def read_json(path: Path) -> dict:
    """Read an object without accepting an ambiguous manifest/config shape."""
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def source_manifests(directory: Path) -> dict:
    """Bind upstream evidence itself, which artifact hashes deliberately exclude."""
    return {name: {"sha256": file_sha256(directory / name),
                   "content": read_json(directory / name)}
            for name in ("run_manifest.json", "export_manifest.json")
            if (directory / name).is_file()}


def check_tokenizer(actual: dict, expected: dict | None) -> None:
    """Compare semantic identity, not a loader class name or relocated path."""
    if expected is not None:
        changed = [key for key in TOKENIZER_FIELDS
                   if key in expected and actual[key] != expected[key]]
        if changed:
            raise ValueError(f"tokenizer identity differs from saved evidence: {changed}")


def check_saved_tokenizer(actual: dict, manifests: dict) -> None:
    """Check both the training-time and export-time tokenizer fingerprints."""
    for record in manifests.values():
        evidence = record["content"]
        check_tokenizer(actual, evidence.get("tokenizer"))
        check_tokenizer(actual, (evidence.get("model") or {}).get("tokenizer"))


def require_safe_model(directory: Path) -> dict:
    """Require local full-model safetensors; never accept pickle or quantization."""
    config = read_json(directory / "config.json")
    if config.get("quantization_config"):
        raise ValueError("quantized bases/exports are unsupported; use unquantized safetensors")
    if (directory / "model.safetensors").is_file():
        return config
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        shards = set(read_json(index).get("weight_map", {}).values())
        if shards and all(isinstance(name, str) and name.endswith(".safetensors")
                          and Path(name).name == name and (directory / name).is_file()
                          for name in shards):
            return config
    raise ValueError(f"model requires complete safetensors weights: {directory}")


def new_output(out: str | Path, *inputs: Path) -> Path:
    """Reject existing entries, broken symlinks, and outputs inside source trees."""
    output = Path(out).expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing preexisting output: {output}")
    output = output.resolve()
    if any(output.is_relative_to(path.resolve()) for path in inputs):
        raise ValueError("output must not be inside an input artifact directory")
    return output


def publish_directory(staging: Path, output: Path) -> None:
    """Atomically publish without replacing even a concurrently created empty dir.

    Linux renameat2(RENAME_NOREPLACE) and Windows rename provide this contract.
    Other platforms fail closed rather than use a racy exists-then-replace pair.
    """
    if os.name == "nt":
        os.rename(staging, output)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory publication is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


@contextmanager
def staged_output(output: Path) -> Iterator[Path]:
    """Keep incomplete weights/evidence private and clean up on every failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        yield staging
        publish_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def output_identity(staging: Path, output: Path) -> dict:
    """Hash published payload files, excluding the self-referential manifest."""
    identity = reference_identity(str(staging), kind="export-output")
    identity["name"] = str(output)
    return identity


def adapter_references(adapter: Path, *, base: str | None,
                       revision: str | None, manifests: dict) -> tuple[dict, dict, dict]:
    """Resolve the stored base identity before any HF loader can run."""
    config = read_json(adapter / "adapter_config.json")
    if config.get("peft_type") != "LORA":
        raise ValueError("only LORA adapters are supported")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError("adapter requires adapter_model.safetensors; pickle is not allowed")
    if not (adapter / "tokenizer_config.json").is_file():
        raise ValueError("adapter must include its saved tokenizer; no base-tokenizer fallback")
    stored_name = config.get("base_model_name_or_path")
    name = base if base is not None else stored_name
    run = manifests.get("run_manifest.json", {}).get("content", {})
    saved_ref = (run.get("model") or {}).get("reference")
    selected_revision = revision
    if revision is not None:
        require_revision(revision, label="base override")
    elif base is None or base == stored_name:
        selected_revision = config.get("revision")
        if selected_revision is None and saved_ref and saved_ref.get("name") == name:
            if saved_ref.get("source") == "huggingface":
                selected_revision = saved_ref.get("revision")
    base_ref = reference_identity(name, selected_revision, kind="base-model")
    if base_ref["source"] == "local":
        base_ref["name"] = str(Path(name).resolve())
        require_safe_model(Path(base_ref["name"]))
    if saved_ref and base is None and revision is None:
        if (saved_ref.get("revision") != base_ref["revision"]
                or saved_ref.get("source") != base_ref["source"]
                or (base_ref["source"] == "huggingface" and saved_ref.get("name") != name)):
            raise ValueError("base identity differs from run manifest; an explicit pinned override is required")
    return base_ref, reference_identity(str(adapter), kind="adapter"), config


def merge_adapter(adapter: str | Path, out: str | Path, *, base: str | None = None,
                  revision: str | None = None, dtype: str = "float32") -> dict:
    """Merge on CPU, preserving tokenizer semantics and the entire identity chain."""
    adapter_path = Path(adapter).expanduser().resolve()
    output = new_output(out, adapter_path)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if not adapter_path.is_dir():
        raise ValueError("adapter must be a local directory")
    manifests = source_manifests(adapter_path)
    base_ref, adapter_ref, adapter_config = adapter_references(
        adapter_path, base=base, revision=revision, manifests=manifests)
    if base_ref["source"] == "local":
        new_output(output, Path(base_ref["name"]))
    tok_ref = reference_identity(str(adapter_path), kind="tokenizer")

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_ref["name"], **hf_load_kwargs(tok_ref))
    tok_identity = tokenizer_identity(tok, tok_ref)
    check_saved_tokenizer(tok_identity, manifests)
    base_kwargs = hf_load_kwargs(base_ref)
    config = AutoConfig.from_pretrained(base_ref["name"], **base_kwargs)
    if getattr(config, "quantization_config", None):
        raise ValueError("quantized bases are unsupported; use unquantized safetensors")
    model = AutoModelForCausalLM.from_pretrained(
        base_ref["name"], config=config, dtype=getattr(torch, dtype),
        use_safetensors=True, device_map={"": "cpu"}, **base_kwargs)
    peft_kwargs = {"local_files_only": True, "trust_remote_code": False,
                   "torch_device": "cpu", "autocast_adapter_dtype": False,
                   "is_trainable": False}
    model = PeftModel.from_pretrained(model, str(adapter_path), **peft_kwargs)
    merged = model.merge_and_unload(safe_merge=True).eval()

    with staged_output(output) as staging:
        merged.save_pretrained(staging, safe_serialization=True)
        tok.save_pretrained(staging)
        require_safe_model(staging)
        # Serialization must not silently change a template or special-token ID.
        saved_tok = AutoTokenizer.from_pretrained(str(staging), local_files_only=True,
                                                  trust_remote_code=False)
        check_tokenizer(tokenizer_identity(saved_tok), tok_identity)
        # Detect input edits during loading/merging instead of signing stale hashes.
        for identity in (base_ref, adapter_ref):
            current = reference_identity(identity["name"], identity["revision"], kind=identity["kind"])
            if current["revision"] != identity["revision"]:
                raise ValueError("input artifacts changed during merge")
        if source_manifests(adapter_path) != manifests:
            raise ValueError("source manifests changed during merge")
        evidence = {
            "schema": "coder-finetune.merge/v1", "operation": "cpu-lora-merge",
            "base_model": base_ref, "adapter": adapter_ref, "tokenizer": tok_identity,
            "dtype": dtype, "versions": environment_versions(), "source_manifests": manifests,
            "effective_config": {
                "base_override": base, "revision_override": revision,
                "base_load": {**base_kwargs, "dtype": dtype, "use_safetensors": True,
                              "device_map": {"": "cpu"}},
                "adapter_load": peft_kwargs, "adapter_config": adapter_config,
                "model_config": merged.config.to_dict(), "safe_merge": True,
                "safe_serialization": True,
            },
            "output": output_identity(staging, output),
            "validation": {"vllm": dict(NOT_VLLM_VALIDATION)},
        }
        write_json(staging / "export_manifest.json", evidence)
        evidence = read_json(staging / "export_manifest.json")
    return evidence


def main(argv: list[str] | None = None) -> None:
    """Expose the complete offline merge contract through argparse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="local adapter with saved tokenizer and safetensors")
    parser.add_argument("--out", required=True, help="new directory; existing paths are rejected")
    parser.add_argument("--base", help="explicit replacement base (local path or pinned HF repo)")
    parser.add_argument("--revision", help="immutable 40-character base commit; required for legacy remote bases")
    parser.add_argument("--dtype", choices=DTYPES, default="float32", help="CPU merge/output dtype (default: float32)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        evidence = merge_adapter(args.adapter, args.out, base=args.base,
                                 revision=args.revision, dtype=args.dtype)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    LOGGER.info("Merged safetensors -> %s; NOT real vLLM validation", evidence["output"]["name"])


if __name__ == "__main__":
    main()
