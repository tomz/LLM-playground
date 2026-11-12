"""Version compatibility and failed-branch cost accounting contracts."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from provenance import AttemptRecord, TOKEN_FIELDS, USD_FIELDS, append_attempt, cost_summary, read_attempts, total_cost


def record(**changes: object) -> AttemptRecord:
    """Construct an ordinary successful v2 record with optional overrides."""
    values = dict(experiment=1, candidate_sha256="a" * 64, git_revision="abc", description="test",
                  seed=7, budget_kind="tokens", budget_value=100, status="keep", gate_ok=True, gate_reason="ok")
    return AttemptRecord(**(values | changes))


def test_v1_missing_version_and_mixed_roundtrip(tmp_path: Path) -> None:
    legacy_names = {"experiment", "candidate_sha256", "git_revision", "description", "seed", "budget_kind",
                    "budget_value", "status", "gate_ok", "gate_reason", "tokens", "wall_s", "val_bpb",
                    "verifier_cost_s", "human_interventions", "metadata"}
    legacy = {key: value for key, value in record(tokens=100, wall_s=2).to_dict().items() if key in legacy_names}
    path = tmp_path / "attempts.jsonl"
    path.write_text(json.dumps(legacy) + "\n\n" + json.dumps(legacy | {"schema_version": 1}) + "\n")
    current = record(domain="code", model_id="local@sha", policy_id="policy@3", grader_id="pytest@9",
                     effort="small", attempt_id="retry", parent_attempt_id="first", retry_index=1,
                     input_tokens=20, output_tokens=10, cached_input_tokens=5, reasoning_tokens=4,
                     grader_tokens=3, model_cost_usd=0.1, grader_cost_usd=0.02, other_cost_usd=0.0)
    append_attempt(path, current)
    rows = list(read_attempts(path))
    assert [row.schema_version for row in rows] == [1, 1, 3]
    assert rows[-1] == current
    assert rows[0].model_id is None and rows[0].input_tokens is None
    assert total_cost(rows) == {"attempts": 3.0, "tokens": 200.0, "wall_s": 4.0, "verifier_cost_s": 0.0}


@pytest.mark.parametrize("version", [0, 4, 1.5, "1", True, None])
def test_reject_invalid_schema(tmp_path: Path, version: object) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record().to_dict() | {"schema_version": version}))
    with pytest.raises(ValueError, match="schema"):
        list(read_attempts(path))


@pytest.mark.parametrize("name", ["budget_value", "wall_s", "verifier_cost_s", "tokens", "retry_index",
                                 *TOKEN_FIELDS, *USD_FIELDS])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, "3"])
def test_reject_invalid_costs(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        record(**{name: value})


@pytest.mark.parametrize("name", ["tokens", "retry_index", *TOKEN_FIELDS])
def test_token_and_retry_counts_are_integral(name: str) -> None:
    with pytest.raises(ValueError):
        record(**{name: 0.5})


@pytest.mark.parametrize("changes", [
    {"budget_value": 0}, {"budget_value": 1.5}, {"experiment": 0}, {"experiment": 1.1},
    {"wall_s": None}, {"verifier_cost_s": None, "schema_version": 2}, {"gate_ok": "false"},
    {"cached_input_tokens": 1}, {"input_tokens": 2, "cached_input_tokens": 3},
    {"output_tokens": 2, "reasoning_tokens": 3}, {"retry_index": 1},
    {"attempt_id": "same", "parent_attempt_id": "same"}, {"domain": " "}, {"model_id": 9},
])
def test_inconsistent_records_fail(changes: dict) -> None:
    with pytest.raises(ValueError):
        record(**changes)


def test_full_campaign_cost_counts_failures_and_not_subsets_twice() -> None:
    base = record(tokens=100, wall_s=2, verifier_cost_s=0.5, input_tokens=20, output_tokens=10,
                  cached_input_tokens=5, reasoning_tokens=3, grader_tokens=4,
                  model_cost_usd=0.2, grader_cost_usd=0.1, other_cost_usd=0.4)
    rows = [dataclasses.replace(base, attempt_id=str(i), status=status, gate_ok=passed)
            for i, (status, passed) in enumerate([("keep", True), ("crash", False), ("discard", True),
                                                 ("keep", False), ("keep", True)])]
    summary = cost_summary(iter(rows))
    assert summary["accepted_results"] == 2
    totals = summary["totals"]
    assert totals["tokens"] == 500
    assert totals["inference_tokens"] == 170  # 5 * (20 + 10 + 4), no cache/reasoning double count
    assert totals["cost_usd"] == pytest.approx(3.5)
    assert summary["per_accepted_result"]["cost_usd"] == pytest.approx(1.75)
    assert summary["per_accepted_result"]["wall_s"] == 5
    assert summary["measured_attempts"]["model_cost_usd"] == 5


def test_unknown_costs_are_not_free_and_zero_acceptance_is_null() -> None:
    summary = cost_summary([record(status="discard")])
    assert summary["totals"]["cost_usd"] is None
    assert summary["totals"]["inference_tokens"] is None
    assert summary["measured_attempts"]["model_cost_usd"] == 0
    assert all(value is None for value in summary["per_accepted_result"].values())
    mixed = cost_summary([record(), record(model_cost_usd=1, grader_cost_usd=0, other_cost_usd=0)])
    assert mixed["totals"]["model_cost_usd"] is None
    assert mixed["measured_attempts"]["model_cost_usd"] == 1
    assert mixed["per_accepted_result"]["cost_usd"] is None
    empty = cost_summary([])
    assert empty["totals"]["cost_usd"] == 0
    assert empty["accepted_results"] == 0
    json.dumps([summary, mixed, empty], allow_nan=False)


def test_parent_lineage_does_not_double_count_and_duplicate_ids_rejected() -> None:
    first = record(attempt_id="first", status="crash", gate_ok=False, tokens=50)
    retry = record(attempt_id="retry", parent_attempt_id="first", retry_index=1, tokens=75)
    assert cost_summary([first, retry])["per_accepted_result"]["tokens"] == 125
    with pytest.raises(ValueError, match="duplicate"):
        cost_summary([first, first])
