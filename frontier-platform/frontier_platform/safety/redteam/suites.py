"""Suite registry: maps a suite name to its probe corpus.

Keeps the original :data:`SUITES` list constant so back-compat tests pass, but
lets new code call :func:`register_suite` to add HarmBench / AdvBench / WMDP /
METR-agent corpora from disk.
"""
from __future__ import annotations

from dataclasses import dataclass

from .probes import Probe, _all_builtin


@dataclass
class Suite:
    """A named collection of probes."""

    name: str
    probes: list[Probe]
    description: str = ""


_REGISTRY: dict[str, Suite] = {
    name: Suite(name=name, probes=list(probes),
                description=f"Built-in {name} stand-in probes (structural reps).")
    for name, probes in _all_builtin().items()
}


# Back-compatible module-level list — the original API exposed this.
SUITES: list[str] = list(_REGISTRY.keys())


def register_suite(suite: Suite, *, overwrite: bool = False) -> None:
    """Register a new (or replacement) suite.

    By default refuses to overwrite an existing suite to avoid silent
    behaviour drift in tests; pass ``overwrite=True`` if you really mean to.
    """
    if suite.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"suite {suite.name!r} already registered; pass overwrite=True to replace"
        )
    _REGISTRY[suite.name] = suite
    if suite.name not in SUITES:
        SUITES.append(suite.name)


def get_suite(name: str) -> Suite:
    if name not in _REGISTRY:
        raise KeyError(f"unknown suite: {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_suites() -> list[Suite]:
    return [_REGISTRY[n] for n in SUITES]
