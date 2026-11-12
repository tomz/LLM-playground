"""Offline export contracts plus a real, local CPU PEFT merge parity test."""
from __future__ import annotations

import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import peft
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from cf_provenance import file_sha256, reference_identity, tokenizer_identity, write_json
from infer import export_vllm, merge_lora
from test_real_sft_repair import local_fixture

COMMIT = "a" * 40
OVERRIDE = "b" * 40


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression must fail, not reach a network service or use a GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in export tests")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)


@pytest.fixture
def tiny(tmp_path: Path) -> tuple:
    """Reuse only the local test fixture, never another subproject's code."""
    model, tok, *_ = local_fixture(tmp_path, "all_tokens")
    return model, tok


def adapter_files(path: Path, tok, *, base: str = "example/base",
                  revision: str | None = COMMIT) -> Path:
    """Make metadata for mocked loaders; the real test saves actual PEFT weights."""
    path.mkdir()
    write_json(path / "adapter_config.json", {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2, "lora_alpha": 4,
        "base_model_name_or_path": base, "revision": revision,
    })
    (path / "adapter_model.safetensors").write_bytes(b"mock-only adapter weights")
    tok.save_pretrained(path)
    return path


def full_model(path: Path, tiny: tuple) -> Path:
    """Save local safe weights/tokenizer, without a hub or remote model loader."""
    model, tok = tiny
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    return path


@pytest.fixture
def loaders(monkeypatch: pytest.MonkeyPatch, tiny: tuple) -> SimpleNamespace:
    """Mock weight/config loading; use the real local tokenizer serializer."""
    model, _ = tiny
    config = Mock(return_value=model.config)
    weights = Mock(return_value=model)
    tokenizer = Mock(side_effect=AutoTokenizer.from_pretrained)
    wrapped = Mock()
    wrapped.merge_and_unload.return_value = model
    adapter = Mock(return_value=wrapped)
    monkeypatch.setattr(AutoConfig, "from_pretrained", config)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", weights)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter)
    return SimpleNamespace(config=config, weights=weights, tokenizer=tokenizer,
                           adapter=adapter, wrapped=wrapped, model=model)


def assert_no_loaders(loaders: SimpleNamespace) -> None:
    """Invalid metadata/output must fail before even a tokenizer/config lookup."""
    for loader in (loaders.config, loaders.weights, loaders.tokenizer, loaders.adapter):
        loader.assert_not_called()


def assert_output_hashes(path: Path, evidence: dict) -> None:
    """The manifest binds every payload artifact without hashing itself."""
    actual = reference_identity(str(path), kind="export-output")
    assert evidence["output"] == actual
    assert "export_manifest.json" not in actual["files"]
    assert json.loads((path / "export_manifest.json").read_text()) == evidence
    assert evidence["validation"]["vllm"]["validated"] is False
    assert "NOT real vLLM validation" in evidence["validation"]["vllm"]["statement"]


@pytest.mark.parametrize(("stored", "override", "expected"), [
    (COMMIT, None, COMMIT), (None, OVERRIDE, OVERRIDE),
    (COMMIT, OVERRIDE, OVERRIDE), ("main", OVERRIDE, OVERRIDE),
])
def test_pinned_revision_propagates_to_safe_cpu_loaders(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
    stored: str | None, override: str | None, expected: str,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1], revision=stored)
    out = tmp_path / "merged"
    evidence = merge_lora.merge_adapter(adapter, out, revision=override)
    base_calls = [call for call in loaders.config.call_args_list if call.args == ("example/base",)]
    assert len(base_calls) == 1
    assert base_calls[0].kwargs == {
        "revision": expected, "trust_remote_code": False, "local_files_only": True,
    }
    for call in loaders.config.call_args_list:
        assert call.kwargs["local_files_only"] is True
        assert call.kwargs["trust_remote_code"] is False
        assert call.args == ("example/base",) or str(tmp_path) in call.args[0]
    kwargs = loaders.weights.call_args.kwargs
    assert kwargs["revision"] == expected
    assert kwargs["use_safetensors"] is True
    assert kwargs["trust_remote_code"] is False and kwargs["local_files_only"] is True
    assert kwargs["dtype"] is torch.float32 and kwargs["device_map"] == {"": "cpu"}
    assert loaders.adapter.call_args.args[1] == str(adapter)
    assert loaders.adapter.call_args.kwargs == {
        "local_files_only": True, "trust_remote_code": False, "torch_device": "cpu",
        "autocast_adapter_dtype": False, "is_trainable": False,
    }
    loaders.wrapped.merge_and_unload.assert_called_once_with(safe_merge=True)
    assert loaders.tokenizer.call_args_list[0].args == (str(adapter),)
    for call in loaders.tokenizer.call_args_list:
        assert call.kwargs == {"local_files_only": True, "trust_remote_code": False}
    assert evidence["base_model"]["revision"] == expected
    assert evidence["base_model"]["revision_type"] == "git-commit"
    assert evidence["adapter"]["files"]["adapter_model.safetensors"] == file_sha256(
        adapter / "adapter_model.safetensors")
    assert evidence["tokenizer"]["chat_template_sha256"] == tokenizer_identity(tiny[1])["chat_template_sha256"]
    assert evidence["effective_config"]["revision_override"] == override
    assert evidence["dtype"] == "float32"
    assert_output_hashes(out, evidence)


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "a" * 39, "a" * 64])
def test_unpinned_legacy_remote_base_fails_before_loaders(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace, revision: str | None,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1], revision=revision)
    with pytest.raises(ValueError, match="immutable.*revision"):
        merge_lora.merge_adapter(adapter, tmp_path / "out")
    assert_no_loaders(loaders)
    assert not (tmp_path / "out").exists()


def test_replacement_repo_does_not_inherit_original_revision(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1])
    with pytest.raises(ValueError, match="immutable"):
        merge_lora.merge_adapter(adapter, tmp_path / "out", base="another/base")
    assert_no_loaders(loaders)
    evidence = merge_lora.merge_adapter(adapter, tmp_path / "out", base="another/base", revision=OVERRIDE)
    assert loaders.weights.call_args.args == ("another/base",)
    assert loaders.weights.call_args.kwargs["revision"] == OVERRIDE
    assert evidence["effective_config"]["base_override"] == "another/base"


def test_run_manifest_can_supply_stored_base_revision(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1], revision=None)
    manifest = {"model": {"reference": reference_identity("example/base", COMMIT),
                           "tokenizer": tokenizer_identity(tiny[1])}}
    write_json(adapter / "run_manifest.json", manifest)
    evidence = merge_lora.merge_adapter(adapter, tmp_path / "out")
    assert loaders.weights.call_args.kwargs["revision"] == COMMIT
    assert evidence["source_manifests"]["run_manifest.json"]["content"] == manifest
    assert evidence["source_manifests"]["run_manifest.json"]["sha256"] == file_sha256(
        adapter / "run_manifest.json")


def test_conflicting_training_base_identity_fails_before_loaders(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1])
    write_json(adapter / "run_manifest.json", {
        "model": {"reference": reference_identity("example/base", OVERRIDE)},
    })
    with pytest.raises(ValueError, match="base identity differs"):
        merge_lora.merge_adapter(adapter, tmp_path / "out")
    assert_no_loaders(loaders)


def test_local_base_hash_needs_no_remote_revision(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    base = full_model(tmp_path / "base", tiny)
    adapter = adapter_files(tmp_path / "adapter", tiny[1], base=str(base), revision=None)
    evidence = merge_lora.merge_adapter(adapter, tmp_path / "out")
    assert "revision" not in loaders.weights.call_args.kwargs
    assert evidence["base_model"] == reference_identity(str(base), kind="base-model")
    assert evidence["base_model"]["files"]["model.safetensors"] == file_sha256(base / "model.safetensors")


@pytest.mark.parametrize("defect", ["pickle", "tokenizer", "peft-type", "dtype"])
def test_unsafe_adapter_or_options_fail_before_loaders(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace, defect: str,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1])
    kwargs = {}
    if defect == "pickle":
        (adapter / "adapter_model.safetensors").rename(adapter / "adapter_model.bin")
    elif defect == "tokenizer":
        (adapter / "tokenizer_config.json").unlink()
    elif defect == "peft-type":
        config = merge_lora.read_json(adapter / "adapter_config.json")
        config["peft_type"] = "PROMPT_TUNING"
        write_json(adapter / "adapter_config.json", config)
    else:
        kwargs["dtype"] = "int8"
    with pytest.raises(ValueError):
        merge_lora.merge_adapter(adapter, tmp_path / "out", **kwargs)
    assert_no_loaders(loaders)


@pytest.mark.parametrize("entry", ["file", "empty-directory", "nonempty-directory", "symlink", "broken-symlink"])
@pytest.mark.parametrize("operation", [merge_lora.merge_adapter, export_vllm.export_config])
def test_preexisting_outputs_are_never_reused(
    tmp_path: Path, loaders: SimpleNamespace, entry: str, operation,
) -> None:
    out = tmp_path / "out"
    if entry == "file":
        out.write_text("keep")
    elif entry.endswith("directory"):
        out.mkdir()
        if entry == "nonempty-directory":
            (out / "keep").write_text("keep")
    else:
        target = tmp_path / "target"
        if entry == "symlink":
            target.mkdir()
        out.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="preexisting output"):
        operation(tmp_path / "unused-source", out)
    assert_no_loaders(loaders)
    if entry == "file":
        assert out.read_text() == "keep"
    elif entry == "nonempty-directory":
        assert (out / "keep").read_text() == "keep"
    elif "symlink" in entry:
        assert out.is_symlink()
    else:
        assert out.is_dir() and not list(out.iterdir())


def test_output_cannot_be_nested_inside_base_or_adapter(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    base = full_model(tmp_path / "base", tiny)
    adapter = adapter_files(tmp_path / "adapter", tiny[1], base=str(base), revision=None)
    for parent in (adapter, base):
        with pytest.raises(ValueError, match="inside an input"):
            merge_lora.merge_adapter(adapter, parent / "out")
    assert_no_loaders(loaders)


def test_tokenizer_template_mismatch_stops_before_weight_loading(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1])
    expected = tokenizer_identity(tiny[1])
    expected["chat_template_sha256"] = "changed"
    write_json(adapter / "run_manifest.json", {"tokenizer": expected})
    with pytest.raises(ValueError, match="tokenizer identity.*chat_template"):
        merge_lora.merge_adapter(adapter, tmp_path / "out")
    loaders.weights.assert_not_called()
    for call in loaders.config.call_args_list:
        assert call.args == (str(adapter),)
        assert call.kwargs["local_files_only"] is True
    loaders.adapter.assert_not_called()
    assert not (tmp_path / "out").exists()


def test_failed_weight_save_removes_staging(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1])
    before = reference_identity(str(adapter))

    def fail_save(path: Path, **kwargs: object) -> None:
        (path / "partial.safetensors").write_bytes(b"partial")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(loaders.model, "save_pretrained", fail_save)
    with pytest.raises(OSError, match="simulated disk failure"):
        merge_lora.merge_adapter(adapter, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))
    assert reference_identity(str(adapter)) == before


@pytest.mark.parametrize("nonempty", [False, True])
def test_atomic_publication_does_not_clobber_a_racing_directory(tmp_path: Path, nonempty: bool) -> None:
    out = tmp_path / "out"
    with pytest.raises(FileExistsError):
        with merge_lora.staged_output(out) as staging:
            (staging / "payload.json").write_text("{}")
            out.mkdir()  # Another writer wins after our preflight.
            if nonempty:
                (out / "keep").write_text("keep")
    assert out.is_dir()
    assert not (out / "payload.json").exists()
    assert not list(tmp_path.glob(".out-*"))
    if nonempty:
        assert (out / "keep").read_text() == "keep"


def test_export_is_config_only_and_does_not_cast_or_load_weights(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    source = full_model(tmp_path / "model", tiny)
    before = reference_identity(str(source))
    out = tmp_path / "config"
    evidence = export_vllm.export_config(source, out, dtype="float16", tensor_parallel_size=2, max_model_len=32)
    loaders.weights.assert_not_called()
    loaders.adapter.assert_not_called()
    for call in loaders.config.call_args_list:
        assert call.args == (str(source),)
        assert call.kwargs["local_files_only"] is True
    assert reference_identity(str(source)) == before
    config = json.loads((out / "vllm_config.json").read_text())
    assert config == {
        "model": str(source), "tokenizer": str(source), "dtype": "float16",
        "trust_remote_code": False, "load_format": "safetensors",
        "tensor_parallel_size": 2, "max_model_len": 32,
    }
    assert evidence["effective_config"] == config
    assert evidence["operation"] == "config-only"
    assert evidence["adapter"] is None
    assert evidence["source_model"]["revision"] == before["revision"]
    assert evidence["source_dtype_from_config"] == "float32"
    assert evidence["validation"]["quantization_performed"] is False
    assert not list(out.glob("*.safetensors"))
    assert_output_hashes(out, evidence)


def test_export_rejects_source_tampering_before_loaders(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace,
) -> None:
    adapter = adapter_files(tmp_path / "adapter", tiny[1])
    merged = tmp_path / "merged"
    merge_lora.merge_adapter(adapter, merged)
    for loader in (loaders.config, loaders.weights, loaders.tokenizer, loaders.adapter):
        loader.reset_mock()
    with (merged / "model.safetensors").open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="hashes differ"):
        export_vllm.export_config(merged, tmp_path / "config")
    assert_no_loaders(loaders)
    assert not (tmp_path / "config").exists()


@pytest.mark.parametrize("defect", ["remote", "adapter", "pickle", "quantized", "tokenizer"])
def test_invalid_export_source_fails_before_loaders(
    tmp_path: Path, tiny: tuple, loaders: SimpleNamespace, defect: str,
) -> None:
    source = full_model(tmp_path / "model", tiny)
    if defect == "remote":
        source = "example/not-downloaded"
    elif defect == "adapter":
        write_json(source / "adapter_config.json", {})
    elif defect == "pickle":
        (source / "model.safetensors").rename(source / "pytorch_model.bin")
    elif defect == "quantized":
        config = merge_lora.read_json(source / "config.json")
        config["quantization_config"] = {"quant_method": "awq"}
        write_json(source / "config.json", config)
    else:
        (source / "tokenizer_config.json").unlink()
    with pytest.raises(ValueError):
        export_vllm.export_config(source, tmp_path / "out")
    assert_no_loaders(loaders)


@pytest.mark.parametrize("kwargs", [
    {"dtype": "int4"}, {"tensor_parallel_size": 0}, {"tensor_parallel_size": True},
    {"max_model_len": -1}, {"max_model_len": 1.5},
])
def test_invalid_serving_settings_fail_before_loaders(
    tmp_path: Path, loaders: SimpleNamespace, kwargs: dict,
) -> None:
    with pytest.raises(ValueError):
        export_vllm.export_config(tmp_path / "model", tmp_path / "out", **kwargs)
    assert_no_loaders(loaders)


def test_export_evidence_write_failure_removes_staging(
    tmp_path: Path, tiny: tuple, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = full_model(tmp_path / "model", tiny)

    def fail_manifest(path: Path, value: object) -> None:
        if path.name == "export_manifest.json":
            raise OSError("simulated manifest failure")
        write_json(path, value)

    monkeypatch.setattr(export_vllm, "write_json", fail_manifest)
    with pytest.raises(OSError, match="simulated manifest failure"):
        export_vllm.export_config(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


def test_argparse_wires_all_supported_options(monkeypatch: pytest.MonkeyPatch) -> None:
    merge = Mock(return_value={"output": {"name": "merged"}})
    export = Mock(return_value={"output": {"name": "config"}})
    monkeypatch.setattr(merge_lora, "merge_adapter", merge)
    monkeypatch.setattr(export_vllm, "export_config", export)
    merge_lora.main(["--adapter", "adapter", "--base", "repo/base", "--revision", COMMIT,
                     "--out", "merged", "--dtype", "bfloat16"])
    merge.assert_called_once_with("adapter", "merged", base="repo/base", revision=COMMIT, dtype="bfloat16")
    export_vllm.main(["--model", "merged", "--out", "config", "--dtype", "float32",
                      "--tensor-parallel-size", "2", "--max-model-len", "64"])
    export.assert_called_once_with("merged", "config", dtype="float32", tensor_parallel_size=2, max_model_len=64)
    with pytest.raises(SystemExit) as exc:
        export_vllm.main(["--model", "merged", "--out", "config", "--quantization", "awq"])
    assert exc.value.code == 2


def test_real_cpu_peft_merge_logit_parity_and_export_identity_chain(tmp_path: Path, tiny: tuple) -> None:
    """Real nonzero LoRA deltas must survive save, merge, reload, and config export."""
    model, tok = tiny
    base = full_model(tmp_path / "base", tiny)
    base_ref = reference_identity(str(base))
    model = AutoModelForCausalLM.from_pretrained(
        str(base), dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, use_safetensors=True).cpu().eval()
    inputs = tok("user question red [EOS] assistant yes [EOS]", return_tensors="pt")
    with torch.no_grad():
        base_logits = model(**inputs).logits.clone()
    adapted = peft.get_peft_model(model, peft.LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0., target_modules=["c_attn"],
        fan_in_fan_out=True, task_type="CAUSAL_LM", bias="none",
    ))
    torch.manual_seed(19)
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_" in name:
                parameter.normal_(mean=0., std=0.15)
    adapted.eval()
    with torch.no_grad():
        expected = adapted(**inputs).logits.clone()
    assert (expected - base_logits).abs().max().item() > 1e-4
    assert all(parameter.device.type == "cpu" for parameter in adapted.parameters())
    adapter = tmp_path / "adapter"
    adapted.save_pretrained(adapter, safe_serialization=True)
    tok.save_pretrained(adapter)
    training = {"schema": "coder-finetune.training/v1",
                "model": {"reference": base_ref, "tokenizer": tokenizer_identity(tok, base_ref)},
                "tokenizer": tokenizer_identity(tok)}
    write_json(adapter / "run_manifest.json", training)
    out = tmp_path / "merged"
    merged_evidence = merge_lora.merge_adapter(adapter, out, dtype="float32")
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(out), dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, use_safetensors=True).cpu().eval()
    with torch.no_grad():
        actual = reloaded(**inputs).logits
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert not any("lora_" in name for name, _ in reloaded.named_parameters())
    saved_tok = AutoTokenizer.from_pretrained(str(out), local_files_only=True, trust_remote_code=False)
    for field in merge_lora.TOKENIZER_FIELDS:
        assert tokenizer_identity(saved_tok)[field] == tokenizer_identity(tok)[field]
    messages = [{"role": "user", "content": "question red"}, {"role": "assistant", "content": "yes"}]
    assert saved_tok.apply_chat_template(messages, tokenize=False) == tok.apply_chat_template(messages, tokenize=False)
    assert merged_evidence["base_model"]["revision"] == base_ref["revision"]
    assert merged_evidence["adapter"]["files"]["adapter_model.safetensors"] == file_sha256(
        adapter / "adapter_model.safetensors")
    assert merged_evidence["source_manifests"]["run_manifest.json"]["content"] == training
    assert_output_hashes(out, merged_evidence)
    config_dir = tmp_path / "vllm-config"
    config_evidence = export_vllm.export_config(out, config_dir, dtype="float32")
    assert config_evidence["base_model"] == merged_evidence["base_model"]
    assert config_evidence["adapter"] == merged_evidence["adapter"]
    assert config_evidence["source_model"]["revision"] == merged_evidence["output"]["revision"]
    assert config_evidence["source_manifests"]["export_manifest.json"]["sha256"] == file_sha256(
        out / "export_manifest.json")
    assert_output_hashes(config_dir, config_evidence)
