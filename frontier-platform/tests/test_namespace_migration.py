"""Installed namespace and restricted legacy checkpoint conversion contracts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import platform
import sys
import zipfile

import pytest
import torch

from frontier_platform.model.config import ModelConfig

PATH = Path(__file__).resolve().parents[1] / "tools" / "convert_legacy_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("legacy_checkpoint_conversion", PATH)
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


def legacy_fixture(tmp_path):
    cfg = ModelConfig(vocab_size=16, n_layer=1, n_head=2, n_kv_head=2,
                      d_model=8, d_ffn=16, max_seq_len=8)
    original = tmp_path / "new.pt"
    weights = {"weight": torch.arange(16).reshape(4, 4).float()}
    torch.save({"model": weights, "model_cfg": cfg, "step": 7}, original)
    legacy = tmp_path / "old.pt"
    with zipfile.ZipFile(original) as src, zipfile.ZipFile(legacy, "w") as dst:
        changed = False
        for entry in src.infolist():
            value = src.read(entry.filename)
            if entry.filename.endswith("data.pkl"):
                before = b"frontier_platform.model.config\n"
                assert before in value
                value = value.replace(before, b"platform.model.config\n")
                changed = True
            dst.writestr(entry, value)
    assert changed
    return legacy, weights, cfg


def test_legacy_config_conversion_preserves_state_and_stdlib(tmp_path):
    legacy, weights, cfg = legacy_fixture(tmp_path)
    before = CONVERT.file_hash(legacy)
    stdlib = sys.modules["platform"]
    destination = CONVERT.convert_checkpoint(legacy, tmp_path / "converted.pt")
    assert sys.modules["platform"] is stdlib is platform
    assert not hasattr(platform, "__path__")
    assert "platform.model.config" not in sys.modules
    with torch.serialization.safe_globals([ModelConfig]):
        result = torch.load(destination, weights_only=True)
    assert result["model_cfg"] == cfg and result["step"] == 7
    torch.testing.assert_close(result["model"]["weight"], weights["weight"])
    assert result["namespace_conversion"]["source_sha256"] == before
    assert CONVERT.file_hash(legacy) == before
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_conversion_refuses_overwrite_before_loading(tmp_path):
    legacy, _, _ = legacy_fixture(tmp_path)
    out = tmp_path / "keep.pt"
    out.write_bytes(b"operator data")
    with pytest.raises(FileExistsError):
        CONVERT.convert_checkpoint(legacy, out)
    assert out.read_bytes() == b"operator data"


def test_conversion_does_not_guess_other_checkpoint_formats(tmp_path):
    source = tmp_path / "unsupported.pt"
    torch.save({"model": {"weight": torch.ones(2)}}, source)
    with pytest.raises(ValueError, match="supported format"):
        CONVERT.convert_checkpoint(source, tmp_path / "out.pt")
    assert not (tmp_path / "out.pt").exists()
