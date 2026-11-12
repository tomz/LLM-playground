"""Offline regressions for exact budgets, document splits and failed work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import harness
import loop
from provenance import AttemptRecord, append_attempt, cost_summary, read_attempts
from prepare_fineweb import document_split, write_document_shards
from data import load_meta

KNOBS = {"block_size": 4, "batch_size": 2, "grad_accum": 1,
         "n_layer": 1, "n_head": 2, "n_kv_head": 2, "d_model": 8,
         "d_ffn": 16, "dropout": 0.0, "dtype": "float32", "lr": 0.001,
         "min_lr": 0.0001, "betas": (0.9, 0.95), "weight_decay": 0.01,
         "max_gen_gap": 100.0}


class ByteTokenizer:
    n_vocab = 257
    eot_token = 256

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def prepare(path: Path, documents: list[str] | None = None, **overrides) -> dict:
    args = dict(out_dir=path, tokens=10000,
                source={"id": "fixture", "revision": "a" * 40, "split": "train"},
                tokenizer_identity={"name": "bytes", "implementation": "fixture",
                                    "version": "1", "sha256": "0" * 64},
                val_frac=0.2, holdout_frac=0.2, seed=1337)
    args.update(overrides)
    docs = documents if documents is not None else [f"doc-{i}: λ" for i in range(60)]
    return write_document_shards(iter(docs), ByteTokenizer(), **args)


@pytest.mark.parametrize("budget", [0, -1, True, 1, 10, 8.5])
def test_invalid_exact_budgets_fail_before_loading_data(tmp_path, budget):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("KNOBS = " + repr(KNOBS) + "\n")
    with pytest.raises(ValueError):
        harness.run_candidate(str(candidate), data_dir=str(tmp_path / "missing"),
                              token_budget=budget, minute_budget=None, seed=7,
                              eval_iters=1, bytes_per_token=1.0, device="cpu")


def test_exact_budget_has_no_rounding():
    assert harness.validate_budget(KNOBS, token_budget=16, minute_budget=None,
                                   eval_iters=1, bytes_per_token=1.0) == (8, 2)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_time_and_normalization_fail_early(value):
    with pytest.raises(ValueError):
        harness.validate_budget(KNOBS, token_budget=None, minute_budget=value,
                                eval_iters=1, bytes_per_token=1.0)
    with pytest.raises(ValueError):
        harness.validate_budget(KNOBS, token_budget=8, minute_budget=None,
                                eval_iters=1, bytes_per_token=value)


def test_whole_document_splits_duplicates_and_exact_artifact_hashes(tmp_path):
    documents = [f"doc-{i}: λ" for i in range(60)] + ["repeat é", "repeat é"]
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare(a, documents)
    second = prepare(b, documents)
    assert first == second
    expected = {split: [] for split in ("train", "val", "holdout")}
    groups = {split: set() for split in expected}
    for text in documents:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        split = document_split(digest, seed=1337, val_frac=0.2, holdout_frac=0.2)
        expected[split].extend(ByteTokenizer().encode_ordinary(text) + [256])
        groups[split].add(digest)
    assert all(not groups[x] & groups[y] for x in groups for y in groups if x != y)
    for split, info in first["splits"].items():
        assert info["documents_count"] > 0
        assert np.fromfile(a / f"{split}.bin", dtype="<u2").tolist() == expected[split]
        assert info["bytes_per_token"] == pytest.approx(info["utf8_bytes"] / info["tokens"])
        for kind in ("artifact", "documents"):
            record = info[kind]
            assert hashlib.sha256((a / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    meta = load_meta(str(a))
    assert meta["bytes_per_token"]["val"] == first["splits"]["val"]["bytes_per_token"]
    with pytest.raises((ValueError, FileExistsError)):
        prepare(a, documents)


def test_legacy_bpe_requires_a_measured_or_explicit_byte_factor():
    with pytest.raises(ValueError):
        harness.resolve_byte_normalization({"vocab_size": 16}, "val", None)
    factor, origin = harness.resolve_byte_normalization({"vocab_size": 16}, "val", 3.2)
    assert factor == 3.2 and "explicit" in origin


def test_actual_cpu_holdout_run_uses_sealed_split_and_exact_budget(tmp_path):
    data = tmp_path / "data"
    prepare(data)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("KNOBS = " + repr(KNOBS) + "\n")
    result = harness.run_candidate(str(candidate), data_dir=str(data), token_budget=8,
                                   minute_budget=None, seed=7, eval_iters=1,
                                   bytes_per_token=None, device="cpu", final_holdout=True)
    assert result.tokens == 8
    assert result.evaluation_role == "final_holdout"
    assert result.evaluation_split == "holdout"
    assert result.data_manifest_sha256
    assert result.bytes_per_token > 0


@pytest.mark.parametrize("mode", ["crash", "empty", "malformed", "timeout", "launch", "holdout", "success"])
def test_loop_retains_failure_costs_cleans_files_and_rejects_holdout(monkeypatch, mode):
    paths = []

    def child(cmd, **kwargs):
        path = Path(cmd[cmd.index("--json-out") + 1]); paths.append(path)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, 0.01)
        if mode == "launch":
            raise OSError("cannot launch")
        if mode == "malformed":
            path.write_text("{")
        elif mode in {"success", "holdout"}:
            path.write_text(json.dumps({
                "ok": True, "reason": "ok", "tokens": 8, "wall_s": 0.1,
                "verifier_cost_s": 0.01, "val_bpb": 1.5,
                "evaluation_role": "final_holdout" if mode == "holdout" else "search_validation",
                "evaluation_split": "holdout" if mode == "holdout" else "val",
            }))
        return SimpleNamespace(returncode=7 if mode == "crash" else 0, stderr="fixture")

    monkeypatch.setattr(loop.subprocess, "run", child)
    result = loop.run_once(tokens=8, minutes=None, seed=7, keep_git=False, timeout_s=1)
    assert paths and all(not path.exists() for path in paths)
    assert result["wall_s"] >= 0
    assert result["ok"] is (mode == "success")
    if mode not in {"success", "holdout"}:
        assert result["tokens"] is None and result["verifier_cost_s"] is None
    if mode == "holdout":
        assert "holdout" in result["reason"] and result["tokens"] == 8
    json.dumps(result, allow_nan=False)


def test_unknown_costs_are_not_zero_and_schema2_remains_readable(tmp_path):
    common = dict(experiment=1, candidate_sha256="fixture", git_revision="test",
                  description="test", seed=7, budget_kind="tokens", budget_value=8,
                  status="keep", gate_ok=True, gate_reason="ok")
    known = AttemptRecord(**common, tokens=8, wall_s=1, schema_version=2)
    unknown = AttemptRecord(**(common | {"experiment": 2, "status": "crash", "gate_ok": False}),
                            tokens=None, wall_s=2, verifier_cost_s=None)
    path = tmp_path / "attempts.jsonl"
    append_attempt(path, known); append_attempt(path, unknown)
    rows = list(read_attempts(path))
    assert [row.schema_version for row in rows] == [2, 3]
    summary = cost_summary(rows)
    assert summary["totals"]["wall_s"] == 3
    assert summary["totals"]["tokens"] is None
    assert summary["per_accepted_result"]["tokens"] is None
    assert summary["per_accepted_result"]["verifier_cost_s"] is None
    assert summary["measured_attempts"]["tokens"] == 1
    json.dumps(summary, allow_nan=False)
