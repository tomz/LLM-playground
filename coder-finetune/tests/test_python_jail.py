"""Python executor contracts plus opt-in containment checks using trusted fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.python_jail import (CommandResult, DockerPythonExecutor, ExecutorUnavailable,
                              PythonLimits, bounded_command, executor_from_config)

IMAGE = "sha256:" + "b" * 64
COMPLETED = "\nCODER_FINETUNE_TESTS_COMPLETED_v1\n"


class Runner:
    def __init__(self, result=None, cleanup=None, inspect=None):
        self.calls = []
        self.result = result or CommandResult("exited", 0, COMPLETED)
        self.cleanup = cleanup or CommandResult("exited", 0, "removed")
        self.inspect = inspect or CommandResult("exited", 0, IMAGE)
        self.job = None
        self.files = None

    def __call__(self, args, timeout, output_limit):
        self.calls.append((args, timeout, output_limit))
        if "inspect" in args:
            return self.inspect
        if "rm" in args:
            if isinstance(self.cleanup, BaseException):
                raise self.cleanup
            return self.cleanup
        mount = args[args.index("--mount") + 1]
        self.job = Path(mount.split("src=", 1)[1].split(",", 1)[0])
        self.files = {p.name: p.read_text() for p in self.job.iterdir()}
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_minimal_mounts_limits_and_no_host_metadata(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "host-secret-do-not-copy")
    runner = Runner()
    result = DockerPythonExecutor(IMAGE, runner=runner).run("assert 1 + 1 == 2")
    assert result.passed is True and result.cleanup == "removed"
    args, timeout, limit = runner.calls[1]
    for flag in ("--pull=never", "--network=none", "--ipc=none", "--read-only", "--user=65534:65534",
                 "--cap-drop=ALL", "--security-opt=no-new-privileges", "--memory=256m",
                 "--memory-swap=256m", "--cpus=1", "--pids-limit=32", "--log-driver=none"):
        assert flag in args
    assert args.count("--mount") == 1 and "readonly" in args[args.index("--mount") + 1]
    assert "cpu=5:5" in args and "fsize=1048576:1048576" in args and "nofile=64:64" in args
    assert args[-8:] == ["-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "LANG=C.UTF-8",
                          "/usr/local/bin/python3", "-I", "-B", "/job/runner.py"]
    assert timeout == 5 and limit == 16384
    assert set(runner.files) == {"program.py", "runner.py"}
    assert runner.files["program.py"] == "assert 1 + 1 == 2"
    assert not runner.job.exists()
    assert runner.calls[-1][0][-3:] == ["rm", "--force", args[args.index("--name") + 1]]
    record = json.dumps(result.to_dict())
    assert "host-secret-do-not-copy" not in record and str(runner.job) not in record
    assert str(Path.home()) not in record


@pytest.mark.parametrize("command,outcome,passed", [
    (CommandResult("exited", 1, "AssertionError"), "failed", False),
    (CommandResult("exited", 0, ""), "protocol_error", None),
    (CommandResult("exited", 125, "daemon failed"), "runner_error", None),
    (CommandResult("timeout", -9), "timeout", False),
    (CommandResult("output_limit", -9, "x"), "output_limit", False),
    (OSError("host-secret"), "runner_error", None),
])
def test_failure_verdicts_always_cleanup(command, outcome, passed):
    runner = Runner(result=command)
    result = DockerPythonExecutor(IMAGE, runner=runner).run("assert True")
    assert result.passed is passed and result.outcome == outcome
    assert result.cleanup == "removed" and len(runner.calls) == 3
    assert "host-secret" not in json.dumps(result.to_dict())
    assert not runner.job.exists()


@pytest.mark.parametrize("cleanup", [CommandResult("timeout", -9), OSError("host-secret")])
def test_cleanup_failure_invalidates_success(cleanup):
    result = DockerPythonExecutor(IMAGE, runner=Runner(cleanup=cleanup)).run("assert True")
    assert result.passed is None and result.outcome == "cleanup_failed"
    assert result.cleanup == "failed"


def test_keyboard_interrupt_still_cleans_whole_job():
    runner = Runner(result=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        DockerPythonExecutor(IMAGE, runner=runner).run("assert True")
    assert "rm" in runner.calls[-1][0] and not runner.job.exists()


@pytest.mark.parametrize("inspected", [CommandResult("exited", 1), CommandResult("exited", 0, "wrong"),
                                        CommandResult("timeout", -9)])
def test_missing_local_image_never_runs_or_pulls(inspected):
    runner = Runner(inspect=inspected)
    executor = DockerPythonExecutor(IMAGE, runner=runner)
    result = executor.run("assert True")
    assert result.passed is None and result.outcome == "image_unavailable"
    assert len(runner.calls) == 1 and "inspect" in runner.calls[0][0]
    with pytest.raises(ExecutorUnavailable):
        executor.require_available()


@pytest.mark.parametrize("program", ["", "x" * 262145, "x\x00", "\ud800", 7])
def test_invalid_request_never_invokes_runner(program):
    runner = Runner()
    assert DockerPythonExecutor(IMAGE, runner=runner).run(program).outcome == "invalid_request"
    assert not runner.calls


@pytest.mark.parametrize("kwargs", [{"wall_seconds": 0}, {"wall_seconds": float("nan")},
    {"wall_seconds": 31}, {"output_bytes": 100000}, {"memory_mib": 8}, {"pids": 0}, {"cpus": float("inf")}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        PythonLimits(**kwargs)


@pytest.mark.parametrize("config", [{}, {"image": "python:latest"}, {"image": IMAGE, "backend": "host"},
    {"image": IMAGE, "allow_fallback": True}, {"image": IMAGE, "docker_socket": "tcp://host:2375"}])
def test_invalid_trust_configuration(config):
    with pytest.raises(ValueError):
        executor_from_config(config)


def test_real_control_runner_bounds_utf8_timeout_and_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "host-secret")
    monkeypatch.setenv("DOCKER_HOST", "tcp://untrusted:2375")
    result = bounded_command([sys.executable, "-c", "import os; print(dict(os.environ))"], 5, 2048)
    assert result.returncode == 0 and "host-secret" not in result.output and "DOCKER_HOST" not in result.output
    result = bounded_command([sys.executable, "-c", "import os; os.write(1, b'\\xff' * 100000)"], 5, 256)
    assert result.status == "output_limit" and len(result.output.encode()) <= 256
    started = time.monotonic()
    result = bounded_command([sys.executable, "-c", "import time; time.sleep(10)"], .05, 256)
    assert result.status == "timeout" and time.monotonic() - started < 2
    assert bounded_command(["/nonexistent/coder-control"], 1, 256).status == "runner_error"


@pytest.fixture
def local_executor():
    image = os.environ.get("PYTHON_TEST_IMAGE")
    if not image:
        pytest.skip("no explicitly trusted local PYTHON_TEST_IMAGE; never pull or build")
    executor = DockerPythonExecutor(image, python_path=os.environ.get("PYTHON_TEST_PATH", "/usr/local/bin/python3"),
                                    docker_socket=os.environ.get("PYTHON_TEST_SOCKET", "/var/run/docker.sock"))
    if not executor.available():
        pytest.skip("configured local Docker/image unavailable; never pull or build")
    return executor


def test_local_container_trust_boundaries(local_executor, tmp_path, monkeypatch):
    sentinel = tmp_path / "host-sentinel"
    sentinel.write_text("secret-host-only")
    monkeypatch.setenv("CF_HOST_CREDENTIAL", "not-for-the-candidate")
    source = f'''import os, socket
from pathlib import Path
assert os.getuid() == 65534
assert "CF_HOST_CREDENTIAL" not in os.environ
assert not Path({str(sentinel)!r}).exists()
assert not Path("/var/run/docker.sock").exists()
assert sorted(p.name for p in Path("/job").iterdir()) == ["program.py", "runner.py"]
try:
    Path("/job/should-not-write").write_text("no")
except OSError:
    pass
else:
    raise AssertionError("job mount is writable")
sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect(("198.51.100.1", 80))
except OSError:
    pass
else:
    raise AssertionError("network must be denied")
finally:
    sock.close()
'''
    result = local_executor.run(source)
    assert result.passed is True, result
    assert result.cleanup == "removed"


@pytest.mark.parametrize("source,outcomes", [
    ("assert False", {"failed"}),
    ("import os; os._exit(0)", {"protocol_error"}),
    ("print('x' * 100000)", {"output_limit"}),
    ("import time; time.sleep(60)", {"timeout"}),
    ("x = bytearray(512 * 1024 * 1024)", {"failed"}),
    ("import subprocess; subprocess.Popen(['sleep', '60'])", {"passed", "timeout"}),
])
def test_local_limits_and_descendant_cleanup(local_executor, source, outcomes):
    result = local_executor.run(source)
    assert result.outcome in outcomes, result
    assert result.cleanup == "removed"
    # Check only our prefix, not any other job/container on the worker.
    checked = bounded_command(local_executor._prefix + ["ps", "--all", "--filter", "name=coder-python-", "--format", "{{.Names}}"], 5, 4096)
    assert checked.status == "exited" and checked.returncode == 0 and not checked.output.strip()
