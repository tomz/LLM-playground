"""Structured proof-output evaluation with an injectable certificate checker.

A real Lean invocation belongs outside the Python process in the project's
sandbox/jailer.  This module validates the model contract and records checker
results without pretending that a passing certificate establishes novelty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

CertificateChecker = Callable[[str, str], tuple[bool, str]]


@dataclass(frozen=True)
class ProofSubmission:
    theorem: str
    certificate: str
    informal_argument: str
    citations: list[str]


@dataclass(frozen=True)
class ProofEvaluation:
    schema_valid: bool
    certificate_valid: bool
    theorem_matches: bool
    novelty_reviewed: bool
    message: str

    @property
    def releasable(self) -> bool:
        return (self.schema_valid and self.certificate_valid and
                self.theorem_matches and self.novelty_reviewed)


def parse_submission(text: str) -> ProofSubmission:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("proof submission must be a JSON object")
    required = {"theorem", "certificate", "informal_argument", "citations"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"missing proof fields: {missing}")
    if not all(isinstance(data[key], str)
               for key in ("theorem", "certificate", "informal_argument")):
        raise ValueError("proof text fields must be strings")
    if not isinstance(data["citations"], list) or not all(
            isinstance(item, str) for item in data["citations"]):
        raise ValueError("citations must be a list of strings")
    return ProofSubmission(**{key: data[key] for key in required})


def evaluate_submission(
    text: str,
    expected_theorem: str,
    checker: CertificateChecker,
    *,
    novelty_reviewed: bool = False,
) -> ProofEvaluation:
    try:
        submission = parse_submission(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return ProofEvaluation(False, False, False, novelty_reviewed, str(exc))
    theorem_matches = " ".join(submission.theorem.split()) == " ".join(expected_theorem.split())
    if not theorem_matches:
        return ProofEvaluation(True, False, False, novelty_reviewed,
                               "certificate targets a different theorem")
    valid, message = checker(submission.theorem, submission.certificate)
    return ProofEvaluation(True, bool(valid), True, novelty_reviewed, message)


def main() -> None:
    """Check a submission against a separate trusted host contract, offline."""
    import argparse
    from dataclasses import asdict
    from pathlib import Path

    from .lean_jail import DockerLeanChecker, LeanContract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--image", required=True, help="locally installed sha256 image ID")
    parser.add_argument("--lean-path", default="/opt/lean/bin/lean")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    data = json.loads(args.contract.read_text())
    contract = LeanContract(data["theorem"], tuple(data.get("premises", [])))
    checker = DockerLeanChecker(contract, args.image, timeout=args.timeout, lean_path=args.lean_path)
    result = evaluate_submission(args.submission.read_text(), contract.theorem, checker)
    # CLI never grants novelty review. Persist stdout for an audit artifact.
    print(json.dumps(asdict(result), indent=2))
    raise SystemExit(0 if result.certificate_valid else 1)


if __name__ == "__main__":
    main()
