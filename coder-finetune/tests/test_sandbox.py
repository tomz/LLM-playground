"""Public isolated-execution contracts, not host-subprocess sandbox claims.

Container command policy and optional live containment checks live in
``test_python_jail.py``. These tests never execute candidate text, start Docker,
or use the separately named trusted-fixture subprocess helper.
"""
from __future__ import annotations

import hashlib

import pytest

from eval import run_humaneval
from eval.python_jail import PythonResult
from eval.run_humaneval import run_many, run_one


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("image", [None, "python:latest", "python:3.14", "sha256:bad"])
def test_public_execution_requires_explicit_immutable_image(batch: bool, image: str | None) -> None:
    """Neither API may silently choose a mutable image or host executor."""
    with pytest.raises(ValueError, match="immutable sha256 image ID"):
        if batch:
            run_many(["assert True"], image=image)
        else:
            run_one("assert True", image=image)


def test_empty_batch_does_not_bypass_explicit_backend_requirement() -> None:
    with pytest.raises(ValueError, match="immutable sha256 image ID"):
        run_many([])


@pytest.mark.parametrize("kwargs", [
    {"limits": False}, {"mp_mode": "fork"}, {"mp_mode": "spawn"},
    {"silence_output": False}, {"max_workers": 2},
])
@pytest.mark.parametrize("batch", [False, True])
def test_legacy_host_execution_options_are_removed(kwargs: dict, batch: bool, fake_executor) -> None:
    executor = fake_executor([])
    with pytest.raises(TypeError):
        if batch:
            run_many(["assert True"], executor=executor, **kwargs)
        else:
            run_one("assert True", executor=executor, **kwargs)
    assert executor.programs == []


def test_run_one_returns_structured_verdict(fake_executor) -> None:
    executor = fake_executor([(True, "passed")])
    program = "assert 1 == 1\n"
    result = run_one(program, executor=executor)
    assert isinstance(result, PythonResult)
    assert result.passed is True and result.outcome == "passed"
    assert result.source_sha256 == hashlib.sha256(program.encode()).hexdigest()
    assert result.evaluator == executor.identity()
    assert result.cleanup == "removed"
    assert executor.programs == [program]


def test_run_many_preserves_order_and_unknown_verdicts(fake_executor) -> None:
    verdicts = [(True, "passed"), (False, "failed"), (None, "image_unavailable"), (True, "passed")]
    executor = fake_executor(verdicts)
    programs = [f"# candidate {index}\n" for index in range(len(verdicts))]
    results = run_many(programs, executor=executor)
    assert [(result.passed, result.outcome) for result in results] == verdicts
    assert [result.source_sha256 for result in results] == [
        hashlib.sha256(program.encode()).hexdigest() for program in programs
    ]
    assert executor.programs == programs


def test_run_many_empty_list_with_explicit_executor(fake_executor) -> None:
    executor = fake_executor([])
    assert run_many([], executor=executor) == []
    assert executor.programs == []


@pytest.mark.parametrize("batch", [False, True])
def test_image_and_timeout_configure_default_container_executor(monkeypatch, fake_executor, batch: bool) -> None:
    """A timeout configures container limits, not an unsafe host-code path."""
    image = "sha256:" + "a" * 64  # Synthetic ID used only with the mocked constructor.
    executor = fake_executor([(True, "passed")])
    calls: list[tuple] = []

    def construct(image_arg: str, *, limits):
        calls.append((image_arg, limits))
        return executor

    monkeypatch.setattr(run_humaneval, "DockerPythonExecutor", construct)
    if batch:
        results = run_many(["assert True"], timeout=2.5, image=image)
        assert results[0].passed is True
    else:
        assert run_one("assert True", timeout=2.5, image=image).passed is True
    assert len(calls) == 1
    assert calls[0][0] == image and calls[0][1].wall_seconds == 2.5
    assert executor.programs == ["assert True"]
