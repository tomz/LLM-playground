"""Tests for SWE task-contract auditing and structured proof outputs."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.structured_proof import evaluate_submission  # noqa: E402
from eval.task_contract import Issue, TaskContract, adjudicate, audit_task  # noqa: E402


def test_audit_flags_hidden_exact_literal_and_missing_behavior():
    report = audit_task(TaskContract(
        "task-1",
        prompt="Render the title as markdown.",
        tests="assert render('Chapter') == '  | Chapter |'  # preserve spacing",
        required_behaviors=["preserve spacing", "escape pipes"],
    ))
    issues = {finding.issue for finding in report.findings}
    assert Issue.OVERLY_STRICT in issues
    assert Issue.UNDERSPECIFIED in issues
    assert Issue.LOW_COVERAGE in issues
    assert report.requires_human_review


def test_clean_contract_has_no_findings():
    report = audit_task(TaskContract(
        "task-2",
        prompt="Sort ascending and handle empty input.",
        tests="assert sort_ascending([]) == []  # handle empty input; ascending",
        required_behaviors=["ascending", "empty input"],
    ))
    assert not report.broken


def test_adjudication_keeps_human_boundary():
    report = audit_task(TaskContract(
        "task-3", "Values must not be included", "assert include(value)",
    ))
    assert adjudicate(report, lambda finding: finding.issue == Issue.MISLEADING) == {
        Issue.MISLEADING,
    }


def test_structured_proof_requires_exact_theorem_and_novelty_review():
    payload = json.dumps({
        "theorem": "forall n, n + 0 = n",
        "certificate": "by simp",
        "informal_argument": "zero is additive identity",
        "citations": ["Mathlib"],
    })
    checker = lambda theorem, cert: (cert == "by simp", "lean accepted")
    result = evaluate_submission(payload, "forall n, n + 0 = n", checker)
    assert result.certificate_valid and not result.releasable
    reviewed = evaluate_submission(
        payload, "forall n, n + 0 = n", checker, novelty_reviewed=True,
    )
    assert reviewed.releasable


def test_structured_proof_rejects_wrong_target_before_checker():
    called = False

    def checker(theorem, cert):
        nonlocal called
        called = True
        return True, "ok"

    payload = json.dumps({
        "theorem": "False", "certificate": "by contradiction",
        "informal_argument": "", "citations": [],
    })
    result = evaluate_submission(payload, "True", checker)
    assert not result.theorem_matches and not called


@pytest.mark.parametrize("payload", ["[]", "null", "1", '"proof"', "true"])
def test_structured_proof_rejects_non_object_json(payload):
    def checker(theorem, cert):
        raise AssertionError("invalid schemas must not reach the checker")

    result = evaluate_submission(payload, "True", checker)
    assert not result.schema_valid and not result.releasable
    assert "JSON object" in result.message
