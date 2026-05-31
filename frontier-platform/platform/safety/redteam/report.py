"""Build the JSON report that :func:`platform.safety.gates.preflight` consumes.

The gate expects a flat ``{category: score}`` mapping with one entry per
:data:`platform.safety.gates.CATEGORIES` value. We compute each per-category
score as the **mean harm score across all probes that touched that category**
through any judge — i.e. higher = worse, and 0 means every probe in that
category was refused (or judged safe).

The report also carries ``_per_suite`` (per-suite refusal rates + per-category
breakdowns) and ``_meta`` (judge identity, probe count, schema version) so the
gate report is auditable. The top-level category keys are flat strings so
:func:`gates.preflight` can read them without changes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .probes import CATEGORIES

_SCHEMA_VERSION = 1


def build_report(results: dict[str, dict], *, meta: dict | None = None) -> dict:
    """Aggregate ``run_all(...)`` output into a gates.preflight-compatible report.

    ``results`` is what :func:`run_all` returns: ``{suite: {n_probes, details,
    refusal_rate, ...}}``. Per-category score = mean of
    ``details[*].category_scores[category]`` across every probe in any suite.
    """
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    per_suite: dict[str, dict] = {}
    total_probes = 0

    for suite_name, suite_report in results.items():
        details = suite_report.get("details", [])
        n_probes = int(suite_report.get("n_probes", len(details)))
        total_probes += n_probes
        suite_sums: dict[str, float] = defaultdict(float)
        suite_counts: dict[str, int] = defaultdict(int)
        for d in details:
            for c in CATEGORIES:
                if c in d.get("category_scores", {}):
                    score = float(d["category_scores"][c])
                    sums[c] += score
                    counts[c] += 1
                    suite_sums[c] += score
                    suite_counts[c] += 1
        per_suite[suite_name] = {
            "n_probes": n_probes,
            "refusal_rate": float(suite_report.get("refusal_rate", 0.0)),
            "category_scores": {
                c: (suite_sums[c] / suite_counts[c]) if suite_counts[c] else 0.0
                for c in CATEGORIES
            },
        }

    report: dict = {
        c: (sums[c] / counts[c]) if counts[c] else 0.0 for c in CATEGORIES
    }
    report["_per_suite"] = per_suite
    report["_meta"] = {
        "schema_version": _SCHEMA_VERSION,
        "n_suites": len(results),
        "n_probes": total_probes,
        **(meta or {}),
    }
    return report


def write_report(path: str | Path, results: dict[str, dict], *, meta: dict | None = None) -> Path:
    """Build a report and write it to ``path`` as JSON. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_report(results, meta=meta), indent=2), encoding="utf-8")
    return p
