"""Pinned model/data identities and JSON evidence for the tested HF profile."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import tempfile
import os


TESTED_VERSIONS = {"transformers": "5.8.0", "peft": "0.19.1", "trl": "1.3.0",
                   "datasets": "4.8.5", "accelerate": "1.13.0"}


def canonical_json(value: object) -> str:
    """Canonical JSON used by content and split identities."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    """Hash a JSON value, with no implicit whitespace or order differences."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Stream file hashing so large safetensors files do not fill memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def require_revision(revision: str | None, *, label: str) -> str:
    """Remote identities must be immutable HF commit SHAs, never main/tags."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{label} requires an immutable 40-character commit revision (not main/a tag)")
    return revision


def reference_identity(name: str, revision: str | None = None, *, kind: str = "model") -> dict:
    """Identify a remote pinned snapshot or the exact relevant local file bytes."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} name/path is required")
    path = Path(name)
    if path.is_dir():
        extensions = {".json", ".safetensors", ".model", ".txt", ".tiktoken", ".jinja", ".bin"}
        hashes = {str(item.relative_to(path)): file_sha256(item)
                  for item in sorted(path.rglob("*"))
                  if item.is_file() and item.suffix in extensions and not any(p.startswith(".") for p in item.relative_to(path).parts)
                  and item.name not in {"run_manifest.json", "export_manifest.json"}}
        if not hashes:
            raise ValueError(f"local {kind} directory contains no model/tokenizer artifacts: {name}")
        return {"name": name, "kind": kind, "source": "local", "revision": digest(hashes),
                "revision_type": "sha256-files", "files": hashes}
    if path.exists():
        raise ValueError(f"local {kind} must be a directory")
    return {"name": name, "kind": kind, "source": "huggingface", "revision_type": "git-commit",
            "revision": require_revision(revision, label=kind)}


def hf_load_kwargs(identity: dict, *, local_files_only: bool = True) -> dict:
    """Never enable pickle weights or remote Python code as a compatibility fix."""
    kwargs = {"trust_remote_code": False, "local_files_only": local_files_only}
    if identity["source"] == "huggingface":
        kwargs["revision"] = identity["revision"]
    return kwargs


def tokenizer_identity(tokenizer, reference: dict | None = None) -> dict:
    """Record vocabulary, special tokens, and the actual chat template used."""
    return {"reference": reference, "class": type(tokenizer).__name__,
            "vocab_sha256": digest(tokenizer.get_vocab()), "vocab_size": len(tokenizer),
            "chat_template_sha256": digest(tokenizer.chat_template),
            "special_tokens": tokenizer.special_tokens_map,
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}


def environment_versions() -> dict:
    """Record library/runtime versions only; never dump environment variables."""
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "tokenizers", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def check_tested_stack() -> None:
    """Reject untested trainer versions rather than promise open-ended support."""
    installed = environment_versions()
    mismatch = {name: {"installed": installed[name], "tested": version}
                for name, version in TESTED_VERSIONS.items() if installed[name] != version}
    if mismatch:
        raise RuntimeError(f"unsupported HF training profile: {mismatch}; use requirements.txt/constraints-ci.txt")


def strict_config(config_cls, kwargs: dict):
    """Reject every unsupported field; mandatory objective settings cannot vanish."""
    if not is_dataclass(config_cls):
        raise TypeError("trainer config must be a dataclass")
    supported = {field.name for field in fields(config_cls) if field.init}
    unsupported = set(kwargs) - supported
    if unsupported:
        raise ValueError(f"{config_cls.__name__} does not support requested fields: {sorted(unsupported)}")
    return config_cls(**kwargs)


def write_json(path: str | Path, value: object) -> None:
    """Atomically publish JSON evidence, rejecting nonfinite numeric values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def trainer_evidence(trainer, cfg: dict, *, objective: str, data: dict | None = None) -> dict:
    """Capture effective HF arguments alongside model/tokenizer and data identity."""
    evidence = {"schema": "coder-finetune.training/v1", "objective": objective,
                "versions": environment_versions(), "requested_config": cfg,
                "effective_trainer_config": trainer.args.to_dict(), "data": data,
                "model": getattr(trainer.model, "_cf_provenance", None),
                "tokenizer": tokenizer_identity(trainer.processing_class)}
    # Configs are operator-owned, but avoid the common accidental secret fields.
    def redact(value):
        if isinstance(value, dict):
            return {key: "<redacted>" if key.lower() in {"hub_token", "hf_token", "password", "api_key", "token"}
                    else redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value
    return redact(evidence)
