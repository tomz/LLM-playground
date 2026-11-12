"""Atomic local-filesystem metadata and JSON-safe process RNG state."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


def fsync_dir(path: Path) -> None:
    """Persist a directory entry on the supported POSIX checkpoint filesystem."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, payload: Any) -> None:
    """Replace one JSON file atomically, rejecting NaN and infinity."""
    data = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    atomic_text(path, data)


def atomic_text(path: Path, data: str) -> None:
    """Write, fsync, rename, and fsync the containing directory."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    """Hash files incrementally rather than loading shards into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_rng(state: torch.Tensor) -> str:
    return base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii")


def _decode_rng(state: str) -> torch.Tensor:
    return torch.from_numpy(np.frombuffer(base64.b64decode(state, validate=True), dtype=np.uint8).copy())


def capture_rng_state() -> dict:
    """Capture Python, NumPy, CPU Torch, and the active CUDA device only."""
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch": _encode_rng(torch.get_rng_state()),
        "cuda": _encode_rng(torch.cuda.get_rng_state()) if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restore a rank's stream; CPU/CUDA migrations require a weights-only load."""
    if (state["cuda"] is not None) != torch.cuda.is_available():
        raise ValueError("RNG device changed; use a weights-only warm start, not full resume")
    python = state["python"]
    numpy = state["numpy"]
    random.setstate((python[0], tuple(python[1]), python[2]))
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(_decode_rng(state["torch"]))
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(_decode_rng(state["cuda"]))
