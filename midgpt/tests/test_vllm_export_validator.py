import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from validate_hf_export import validate_export  # noqa: E402


def test_validate_export_accepts_minimal_hf_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "gpt2",
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "n_positions": 16,
    }))
    (tmp_path / "generation_config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    report = validate_export(tmp_path)
    assert report["model_type"] == "gpt2"
    assert report["params_hint"]["layers"] == 2
    assert "vllm.entrypoints" in report["vllm_command"]


def test_validate_export_rejects_missing_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "gpt2",
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "n_positions": 16,
    }))
    (tmp_path / "generation_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="model"):
        validate_export(tmp_path)
