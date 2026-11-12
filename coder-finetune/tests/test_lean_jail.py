"""Hermetic jail contract tests; real Lean is opt-in and never pulls images."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.lean_jail import CommandResult, DockerLeanChecker, LeanContract, bounded_command, digest
from eval.structured_proof import evaluate_submission

IMAGE = "sha256:" + "a" * 64
CONTRACT = LeanContract("∀ n : Nat, n + 0 = n")
CERTIFICATE = "by intro n; rfl"


class Runner:
    def __init__(self, result=None, cleanup=None):
        self.calls = []
        self.result = result or CommandResult("exited", 0, "'certified' does not depend on any axioms")
        self.cleanup = cleanup or CommandResult("exited", 0, "removed")
        self.source = None
        self.source_path = None

    def __call__(self, args, timeout, output_limit):
        self.calls.append((args, timeout, output_limit))
        if args[1:3] == ["image", "inspect"]:
            return CommandResult("exited", 0, IMAGE + "\n")
        if args[1] == "rm":
            return self.cleanup
        mount = args[args.index("--mount") + 1]
        self.source_path = Path(mount.split("src=", 1)[1].split(",", 1)[0]) / "Proof.lean"
        self.source = self.source_path.read_text()
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("certificate", [
    "by sorry", "by admit", "by exact sorry", "by rfl\naxiom bad : False",
    "by rfl\n theorem forged : True := by trivial", "by rfl --comment",
    "by\n  rfl", "by native_decide", "by run_tac sorry", "by exact h0",
    "by rfl\nend\nnamespace Other", "by rfl\x00", "by rfl\n#eval IO.println 1",
])
def test_reject_injection_and_holes_without_runner(certificate):
    runner = Runner()
    result = DockerLeanChecker(CONTRACT, IMAGE, runner=runner).check(CONTRACT.theorem, certificate)
    assert result.outcome == "certificate_rejected" and not result.valid
    assert runner.calls == []


def test_exact_host_theorem_and_premises_pins():
    runner = Runner()
    contract = LeanContract("True", ("True",))
    checker = DockerLeanChecker(contract, IMAGE, runner=runner)
    assert checker.check(" True", "by assumption").outcome == "theorem_mismatch"
    assert not runner.calls
    result = checker.check("True", "by assumption")
    assert result.valid
    assert "theorem certified (h0 : True) : True := by assumption" in runner.source
    assert result.theorem_sha256 == digest("True")
    assert result.premises_sha256 == digest(json.dumps(("True",), ensure_ascii=False))
    assert result.source_sha256 == digest(runner.source)
    assert result.certificate_sha256 == digest("by assumption")
    assert len(result.checker_sha256) == 64


def test_hardened_container_args_and_cleanup():
    runner = Runner()
    checker = DockerLeanChecker(CONTRACT, IMAGE, runner=runner)
    result = checker.check(CONTRACT.theorem, CERTIFICATE)
    assert result.valid and result.cleanup == "removed"
    args, timeout, limit = runner.calls[1]
    for flag in ("--pull=never", "--network=none", "--read-only", "--user=65534:65534",
                 "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=256m",
                 "--memory-swap=256m", "--cpus=1", "--pids-limit=32", "--log-driver=none"):
        assert flag in args
    assert "cpu=10:10" in args and "fsize=1048576:1048576" in args
    assert args[-3:] == [IMAGE, "--threads=1", "Proof.lean"]
    assert args[args.index("--entrypoint") + 1] == "/opt/lean/bin/lean"
    assert "readonly" in args[args.index("--mount") + 1]
    assert timeout == 10 and limit == 16384
    assert runner.calls[-1][0] == ["docker", "rm", "--force", args[args.index("--name") + 1]]
    assert not runner.source_path.exists()
    assert runner.source == CONTRACT.source(CERTIFICATE)


@pytest.mark.parametrize("command_result,outcome", [
    (CommandResult("exited", 1, "type mismatch"), "lean_rejected"),
    (CommandResult("exited", 0, "uses sorryAx"), "lean_rejected"),
    (CommandResult("timeout", -9, ""), "timeout"),
    (CommandResult("output_limit", -9, "x"), "output_limit"),
    (OSError("runner missing"), "runner_error"),
])
def test_failure_outcomes_cleanup_and_no_fallback(command_result, outcome):
    runner = Runner(command_result)
    result = DockerLeanChecker(CONTRACT, IMAGE, runner=runner).check(CONTRACT.theorem, CERTIFICATE)
    assert not result.valid and result.outcome == outcome
    assert result.cleanup == "removed" and len(runner.calls) == 3


def test_cleanup_failure_cannot_pass():
    runner = Runner(cleanup=CommandResult("timeout", -9, "daemon unavailable"))
    result = DockerLeanChecker(CONTRACT, IMAGE, runner=runner).check(CONTRACT.theorem, CERTIFICATE)
    assert not result.valid and result.outcome == "cleanup_failed"
    assert "failed" in result.cleanup


@pytest.mark.parametrize("result", [CommandResult("exited", 1, "not installed"),
                                     CommandResult("exited", 0, "sha256:wrong"),
                                     CommandResult("timeout", -9, "daemon timeout")])
def test_unavailable_image_never_runs_or_pulls(result):
    calls = []
    def runner(args, timeout, output_limit):
        calls.append(args)
        return result
    checked = DockerLeanChecker(CONTRACT, IMAGE, runner=runner).check(CONTRACT.theorem, CERTIFICATE)
    assert not checked.valid and checked.outcome == "image_unavailable"
    assert len(calls) == 1 and calls[0][1:3] == ["image", "inspect"]


@pytest.mark.parametrize("kwargs", [{"image": "lean:latest"}, {"timeout": 0},
                                    {"timeout": float("nan")}, {"timeout": 31},
                                    {"output_limit": 100000}, {"lean_path": "lean"}])
def test_invalid_limits_and_mutable_image(kwargs):
    with pytest.raises(ValueError):
        DockerLeanChecker(CONTRACT, **({"image": IMAGE} | kwargs))


@pytest.mark.parametrize("payload", ["{", "{}", "null", "[]", '{"theorem":1}',
    json.dumps({"theorem": "True", "certificate": "by decide", "informal_argument": "", "citations": [1]})])
def test_negative_schema_never_invokes_jail(payload):
    runner = Runner()
    result = evaluate_submission(payload, CONTRACT.theorem, DockerLeanChecker(CONTRACT, IMAGE, runner=runner))
    assert not result.schema_valid and not runner.calls


def test_schema_target_mismatch_and_novelty_gate():
    runner = Runner()
    payload = {"theorem": "False", "certificate": CERTIFICATE, "informal_argument": "", "citations": []}
    checker = DockerLeanChecker(CONTRACT, IMAGE, runner=runner)
    assert not evaluate_submission(json.dumps(payload), CONTRACT.theorem, checker).theorem_matches
    assert not runner.calls
    payload["theorem"] = CONTRACT.theorem
    result = evaluate_submission(json.dumps(payload), CONTRACT.theorem, checker)
    assert result.certificate_valid and not result.releasable
    assert json.loads(result.message)["image"] == IMAGE


def test_real_bounded_runner_output_timeout_and_missing_executable():
    result = bounded_command([sys.executable, "-c", "print('x' * 100000)"], 5, 256)
    assert result.status == "output_limit" and len(result.output.encode()) <= 256
    result = bounded_command([sys.executable, "-c", "import time; time.sleep(10)"], .05, 256)
    assert result.status == "timeout"
    assert bounded_command(["/nonexistent/lean-test-command"], 1, 256).status == "runner_error"


@pytest.mark.skipif(not os.environ.get("LEAN_TEST_IMAGE"), reason="no explicit local pinned Lean image; never pull")
def test_local_lean_integration():
    checker = DockerLeanChecker(CONTRACT, os.environ["LEAN_TEST_IMAGE"],
                                lean_path=os.environ.get("LEAN_TEST_PATH", "/opt/lean/bin/lean"))
    result = checker.check(CONTRACT.theorem, CERTIFICATE)
    if result.outcome == "image_unavailable":
        pytest.skip(f"local Docker/image unavailable: {result.output}")
    assert result.valid, result
    false = DockerLeanChecker(LeanContract("False"), os.environ["LEAN_TEST_IMAGE"],
                              lean_path=os.environ.get("LEAN_TEST_PATH", "/opt/lean/bin/lean"))
    assert false.check("False", "by decide").outcome == "lean_rejected"
