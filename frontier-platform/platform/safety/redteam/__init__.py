"""Automated red-team harness for pre-deployment safety gating.

This package replaces what used to be a single 70-line file containing 5
hardcoded prompts and a refusal regex. The shape (suite → probes → judge →
refusal-rate) is the same so the existing :class:`platform.safety.gates`
pipeline keeps working unchanged — but the *content* is now structured for
extension to real benchmark suites (HarmBench, AdvBench, Cybench, WMDP,
METR-agent, multi-turn social) and the judge is a pluggable protocol so we
can drop in a Llama-Guard-class classifier or LLM-as-judge in place of the
regex.

Probes are **structurally representative**, not actual harmful content.
They're stand-ins for the kind of prompt each benchmark family contains, so
the pipeline can be exercised end-to-end in CI without shipping a jailbreak
playbook in the repo. The :class:`Probe` API lets you bring your own probe
corpus via :func:`register_suite`.

Back-compatibility: the original ``run_suite(endpoint, name)`` and
``run_all(endpoint)`` functions are still here, return the same dict shape
(``{suite, n_probes, refusal_rate, details}``), and the module-level
``SUITES`` list is unchanged. The new functionality — per-category scoring,
ensemble judges, JSON report compatible with :func:`gates.preflight` — is
additive.
"""
from __future__ import annotations

from .judges import (
    CallableJudge,
    ClassifierJudge,
    EnsembleJudge,
    Judge,
    JudgeResult,
    RegexRefusalJudge,
)
from .probes import CATEGORIES, Probe
from .report import build_report, write_report
from .suites import SUITES, Suite, all_suites, get_suite, register_suite


def run_suite(endpoint, suite: str, *, judge: Judge | None = None) -> dict:
    """Run one suite against ``endpoint`` and return a per-probe report.

    Back-compatible dict shape::

        {
            "suite": str,
            "n_probes": int,
            "refusal_rate": float,
            "details": [{prompt, response, refused, category_scores, ...}, ...],
        }

    ``endpoint`` may be a callable ``str -> str`` (most common in tests) or any
    object supporting ``.generate(prompt) -> str``. An HTTP endpoint as a bare
    URL is still rejected with ``NotImplementedError`` — wire a small adapter
    or a :class:`platform.data.synthetic.EngineTeacher` instead.
    """
    judge = judge or RegexRefusalJudge()
    s = get_suite(suite)
    details = []
    refusals = 0
    for probe in s.probes:
        response = _call(endpoint, probe.prompt)
        result = judge.judge(probe, response)
        refusals += int(result.refused)
        details.append({
            "id": probe.id,
            "suite": probe.suite,
            "category": probe.category,
            "prompt": probe.prompt,
            "response": response,
            "refused": result.refused,
            "category_scores": result.category_scores,
            "rationale": result.rationale,
        })
    n = len(s.probes)
    return {
        "suite": s.name,
        "n_probes": n,
        "refusal_rate": (refusals / n) if n else 0.0,
        "details": details,
    }


def run_all(endpoint, *, judge: Judge | None = None) -> dict[str, dict]:
    """Run every registered suite. Back-compatible with the old ``run_all``."""
    return {s: run_suite(endpoint, s, judge=judge) for s in SUITES}


def _call(endpoint, prompt: str) -> str:
    if callable(endpoint):
        return str(endpoint(prompt))
    if hasattr(endpoint, "generate") and callable(endpoint.generate):
        return str(endpoint.generate(prompt))
    if isinstance(endpoint, str):
        raise NotImplementedError(
            "HTTP endpoint URL not supported directly; wrap it in a callable "
            "(e.g. `lambda p: client.post(url, json={'prompt': p}).json()['text']`)."
        )
    raise TypeError(f"unsupported endpoint type: {type(endpoint)!r}")


__all__ = [
    # back-compat
    "SUITES",
    "run_suite",
    "run_all",
    # probes + suites
    "CATEGORIES",
    "Probe",
    "Suite",
    "get_suite",
    "all_suites",
    "register_suite",
    # judges
    "Judge",
    "JudgeResult",
    "RegexRefusalJudge",
    "ClassifierJudge",
    "EnsembleJudge",
    "CallableJudge",
    # reporting
    "build_report",
    "write_report",
]
