"""Offline test doubles at the isolated-executor boundary; never execute source."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.python_jail import PythonResult  # noqa: E402


class RecordingExecutor:
    """Return scripted verdicts while recording, not running, candidate programs."""

    def __init__(self, verdicts: list[tuple[bool | None, str]]) -> None:
        self.verdicts = list(verdicts)
        self.programs: list[str] = []
        self.preflight_calls = 0

    def identity(self) -> dict:
        """Clearly label fake evidence so it cannot be mistaken for containment."""
        return {"backend": "test-double-no-execution"}

    def require_available(self) -> None:
        """Record preflight without contacting Docker."""
        self.preflight_calls += 1

    def run(self, program: str) -> PythonResult:
        """Associate the next scripted result with the exact submitted source."""
        index = len(self.programs)
        self.programs.append(program)
        passed, outcome = self.verdicts[index]
        return PythonResult(passed, outcome, hashlib.sha256(program.encode()).hexdigest(),
                            self.identity(), cleanup="removed")

    def run_many(self, programs: list[str]) -> list[PythonResult]:
        """Keep the order and cardinality of a batch without evaluating its text."""
        return [self.run(program) for program in programs]


@pytest.fixture
def fake_executor() -> type[RecordingExecutor]:
    """Create a fresh scripted executor for each API/reward test."""
    return RecordingExecutor
