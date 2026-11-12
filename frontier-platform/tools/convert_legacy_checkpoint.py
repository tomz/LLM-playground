"""Convert common legacy ModelConfig checkpoints without importing `platform.*`.

Only PyTorch weights-only loading with an explicit ModelConfig alias is allowed.
Other legacy Python objects are rejected; there is no unrestricted-pickle fallback.
Use trusted local artifacts and size resources for the checkpoint being converted.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_platform.model.config import ModelConfig


def file_hash(path: Path) -> str:
    """Stream an artifact digest without retaining the file bytes in memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def convert_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Preserve tensor/config data and publish a new file without overwriting."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError("source must be a regular, non-symlink local checkpoint")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite {destination}")
    before = file_hash(source)
    allowed = [ModelConfig, (ModelConfig, "platform.model.config.ModelConfig")]
    with torch.serialization.safe_globals(allowed):
        blob = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not isinstance(blob.get("model_cfg"), ModelConfig) or "model" not in blob:
        raise ValueError("supported format requires model state and a ModelConfig model_cfg")
    if "namespace_conversion" in blob:
        raise ValueError("checkpoint already carries a conversion receipt")
    blob["namespace_conversion"] = {"schema": 1, "source_sha256": before,
                                    "from": "platform.model.config.ModelConfig",
                                    "to": "frontier_platform.model.config.ModelConfig"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            torch.save(blob, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if file_hash(source) != before:
            raise ValueError("source changed during conversion")
        # A file hard link is an atomic no-clobber publication in the same directory.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = convert_checkpoint(args.source, args.destination)
    logging.info("Converted local checkpoint to %s; source preserved", path)


if __name__ == "__main__":
    main()
