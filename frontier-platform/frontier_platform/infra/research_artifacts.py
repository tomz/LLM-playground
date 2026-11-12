"""Auditable artifacts and release gates for automated research.

Formal validity, novelty, and significance are separate decisions.  The schema
records failed attempts and human edits so marginal inference cost cannot be
mistaken for total research-system cost.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class ResearchCost:
    generation_tokens: int = 0
    verifier_seconds: float = 0.0
    reviewer_seconds: float = 0.0
    preparation_seconds: float = 0.0

    def __add__(self, other: "ResearchCost") -> "ResearchCost":
        return ResearchCost(
            self.generation_tokens + other.generation_tokens,
            self.verifier_seconds + other.verifier_seconds,
            self.reviewer_seconds + other.reviewer_seconds,
            self.preparation_seconds + other.preparation_seconds,
        )


@dataclass(frozen=True)
class ResearchArtifact:
    artifact_id: str
    claim: str
    model: str
    model_revision: str
    prompt_sha256: str
    parent_attempt_ids: list[str]
    status: str
    cost: ResearchCost
    certificate_uri: str | None = None
    certificate_valid: bool = False
    theorem_matches_claim: bool = False
    prior_art_reviewed: bool = False
    significance_reviewed: bool = False
    reviewer_ids: list[str] = field(default_factory=list)
    human_edits: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.status not in {"candidate", "rejected", "verified", "released"}:
            raise ValueError("invalid research artifact status")
        if not self.artifact_id or not self.claim or not self.model_revision:
            raise ValueError("artifact identity, claim, and model revision are required")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ResearchGateResult:
    passed: bool
    missing: list[str]


def release_gate(artifact: ResearchArtifact, *, min_reviewers: int = 2) -> ResearchGateResult:
    checks = {
        "certificate_uri": bool(artifact.certificate_uri),
        "certificate_valid": artifact.certificate_valid,
        "theorem_matches_claim": artifact.theorem_matches_claim,
        "prior_art_reviewed": artifact.prior_art_reviewed,
        "significance_reviewed": artifact.significance_reviewed,
        "independent_reviewers": len(set(artifact.reviewer_ids)) >= min_reviewers,
    }
    return ResearchGateResult(all(checks.values()), [key for key, ok in checks.items() if not ok])


def write_artifacts(path: str | Path, artifacts: Iterable[ResearchArtifact]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for artifact in artifacts:
            stream.write(json.dumps(artifact.to_dict(), sort_keys=True) + "\n")
    return destination


def read_artifacts(path: str | Path) -> Iterator[ResearchArtifact]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("schema_version", 1) != 1:
                raise ValueError("unsupported research artifact schema")
            data["cost"] = ResearchCost(**data["cost"])
            yield ResearchArtifact(**data)


def total_research_cost(artifacts: Iterable[ResearchArtifact]) -> ResearchCost:
    total = ResearchCost()
    for artifact in artifacts:
        total += artifact.cost
    return total
