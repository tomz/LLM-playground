"""Conservative contract audit for SWE-style benchmark tasks.

The auditor produces evidence-backed flags for human adjudication.  It never
silently removes a task: benchmark versioning and exclusion remain reviewer
responsibilities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Issue(str, Enum):
    OVERLY_STRICT = "overly_strict_tests"
    UNDERSPECIFIED = "underspecified_prompt"
    LOW_COVERAGE = "low_coverage_tests"
    MISLEADING = "misleading_prompt"


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    prompt: str
    tests: str
    gold_patch: str = ""
    required_behaviors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuditFinding:
    issue: Issue
    evidence: str
    confidence: float


@dataclass(frozen=True)
class AuditReport:
    task_id: str
    findings: list[AuditFinding]
    requires_human_review: bool = True

    @property
    def broken(self) -> bool:
        return bool(self.findings)


def _quoted_literals(text: str) -> set[str]:
    return {match[1] for match in re.findall(r"(['\"])(.*?)(?:\1)", text)}


def audit_task(contract: TaskContract) -> AuditReport:
    """Flag observable prompt/test/patch inconsistencies.

    ``required_behaviors`` should be supplied by dataset construction or a
    reviewer.  String heuristics are intentionally conservative and each flag
    carries inspectable evidence rather than an automatic exclusion decision.
    """
    findings: list[AuditFinding] = []
    prompt_lower = contract.prompt.lower()
    tests_lower = contract.tests.lower()

    hidden_literals = sorted(_quoted_literals(contract.tests) - _quoted_literals(contract.prompt))
    exact_assertions = re.findall(r"assert\s+.+?\s*==\s*(['\"].*?['\"])", contract.tests)
    if hidden_literals and exact_assertions:
        findings.append(AuditFinding(
            Issue.OVERLY_STRICT,
            f"tests enforce prompt-absent literals: {hidden_literals[:5]}",
            0.75,
        ))

    missing = [behavior for behavior in contract.required_behaviors
               if behavior.lower() not in prompt_lower and behavior.lower() in tests_lower]
    if missing:
        findings.append(AuditFinding(
            Issue.UNDERSPECIFIED,
            f"tested behaviors absent from prompt: {missing}",
            0.9,
        ))

    uncovered = [behavior for behavior in contract.required_behaviors
                 if behavior.lower() not in tests_lower]
    if uncovered:
        findings.append(AuditFinding(
            Issue.LOW_COVERAGE,
            f"required behaviors absent from tests: {uncovered}",
            0.85,
        ))

    negations = (("must not be included", "include"),
                 ("must not include", "include"),
                 ("exclude", "include"),
                 ("ascending", "descending"),
                 ("before", "after"))
    contradictions = [f"prompt={left!r}, tests={right!r}"
                      for left, right in negations
                      if left in prompt_lower and right in tests_lower]
    if contradictions:
        findings.append(AuditFinding(
            Issue.MISLEADING,
            "; ".join(contradictions),
            0.7,
        ))
    return AuditReport(contract.task_id, findings)


def adjudicate(report: AuditReport, reviewer: Callable[[AuditFinding], bool]) -> set[Issue]:
    """Return reviewer-confirmed issues; retain the human decision boundary."""
    return {finding.issue for finding in report.findings if reviewer(finding)}
