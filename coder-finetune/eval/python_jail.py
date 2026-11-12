"""Fail-closed, local Docker execution for generated Python.

The operator trusts the Docker daemon, its kernel/runtime, and an explicitly
supplied immutable image ID. No image is pulled or built. Candidates receive
only two temporary, read-only files, not the trainer's environment, repository,
credentials, or Docker socket. Docker is not a VM: use a dedicated worker and a
stronger runtime for hostile multi-tenant workloads. Unit tests of flags are
not evidence of live containment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time
from typing import Callable
import uuid


# A missing completion marker must never turn os._exit(0) into a passing test.
# This is an integrity check, NOT a defense against a candidate deliberately
# tampering with the Python test harness in the same interpreter.
_COMPLETED = "\nCODER_FINETUNE_TESTS_COMPLETED_v1\n"
_WRAPPER = '''import sys
from pathlib import Path

try:
    source = Path("/job/program.py").read_text(encoding="utf-8")
    exec(compile(source, "/job/program.py", "exec"), {"__name__": "__main__"})
except BaseException as exc:
    print(type(exc).__name__ + ": " + str(exc)[:1000], file=sys.stderr)
    raise SystemExit(1)
else:
    print("\\nCODER_FINETUNE_TESTS_COMPLETED_v1", flush=True)
'''


def _bounded_text(data: bytes, limit: int) -> str:
    """Keep the UTF-8 encoded result within the byte limit, even for bad bytes."""
    return data[:limit].decode("utf-8", "replace").encode("utf-8")[:limit].decode("utf-8", "ignore")


@dataclass(frozen=True)
class CommandResult:
    """Bounded Docker control-command result, not a candidate verdict."""

    status: str
    returncode: int | None
    output: str = ""


def bounded_command(args: list[str], timeout: float, output_limit: int) -> CommandResult:
    """Run without a shell or inherited secrets; kill the whole CLI process group.

    Killing a Docker client does not kill its container. DockerPythonExecutor
    always removes its named container separately, including on cancellation.
    Only trusted test fixtures may call this host-side helper with Python args.
    """
    if not math.isfinite(timeout) or timeout <= 0 or type(output_limit) is not int or output_limit < 1:
        raise ValueError("a finite positive timeout and output byte limit are required")
    if os.name != "posix":
        return CommandResult("runner_error", None, "the local Docker executor requires POSIX")
    started = time.monotonic()
    output = bytearray()
    # No DOCKER_HOST/CONTEXT, auth configuration, proxy credentials, SSH agent,
    # LD_PRELOAD, PYTHONPATH, HF_TOKEN, or other trainer environment is forwarded.
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent",
                   "DOCKER_CONFIG": "/nonexistent", "LANG": "C.UTF-8"}
    try:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, close_fds=True,
                                start_new_session=True, env=environment)
    except OSError:
        return CommandResult("runner_error", None, "control executable unavailable")
    status = "exited"
    assert proc.stdout is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, min(4096, output_limit + 1 - len(output)))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(chunk)
                    if len(output) > output_limit:
                        status = "output_limit"
                        break
                if status != "exited":
                    break
            if status == "exited":
                try:
                    proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    status = "timeout"
    finally:
        # Do this even if the group leader exited: descendants may still hold
        # the output pipe open or keep running after an apparently normal exit.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        proc.stdout.close()
    return CommandResult(status, proc.returncode, _bounded_text(bytes(output), output_limit))


@dataclass(frozen=True)
class PythonLimits:
    """Operator-selected limits inside fixed safety ceilings."""

    wall_seconds: float = 5.0
    output_bytes: int = 16384
    source_bytes: int = 262144
    memory_mib: int = 256
    pids: int = 32
    cpus: float = 1.0
    control_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, lo, hi in (("wall_seconds", 0, 30), ("cpus", 0, 2), ("control_seconds", 0, 10)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo < value <= hi:
                raise ValueError(f"{name} must be finite and in ({lo}, {hi}]")
        for name, lo, hi in (("output_bytes", 256, 65536), ("source_bytes", 1024, 1048576),
                             ("memory_mib", 64, 1024), ("pids", 8, 128)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")


@dataclass(frozen=True)
class PythonResult:
    """None means unassessed (infrastructure/protocol failure), never 'wrong'."""

    passed: bool | None
    outcome: str
    source_sha256: str | None
    evaluator: dict
    output: str = ""
    returncode: int | None = None
    cleanup: str = "not_needed"
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Return JSON-safe evidence without host mount paths or environment."""
        return asdict(self)


class ExecutorUnavailable(RuntimeError):
    """The verifier could not assess a candidate; do not train on a fake zero."""


class DockerPythonExecutor:
    """An explicit image argument is the operator's assertion of image trust.

    The image needs /usr/bin/env and the configured Python executable. It must
    be reviewed, contain no credentials, and already exist in the local daemon.
    Remote daemons are deliberately unsupported; rootless local sockets may be
    supplied explicitly. No trusted-subprocess fallback exists here.
    """

    def __init__(self, image: str, *, limits: PythonLimits | None = None,
                 python_path: str = "/usr/local/bin/python3",
                 docker_socket: str = "/var/run/docker.sock",
                 runner: Callable[[list[str], float, int], CommandResult] = bounded_command):
        if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("supply an explicitly trusted local immutable sha256 image ID; tags are forbidden")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", python_path) or ".." in Path(python_path).parts:
            raise ValueError("python_path must be an absolute executable path inside the image")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+\.sock", docker_socket) or ".." in Path(docker_socket).parts:
            raise ValueError("docker_socket must be an absolute local Unix socket path")
        self.image = image
        self.limits = limits or PythonLimits()
        self.python_path = python_path
        self._prefix = ["docker", f"--host=unix://{docker_socket}"]
        self._runner = runner

    def identity(self) -> dict:
        """Stable policy evidence; intentionally omit host paths and env values."""
        return {"backend": "local-docker-python/v1", "image": self.image,
                "executor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "wrapper_sha256": hashlib.sha256(_WRAPPER.encode()).hexdigest(),
                "python_path": self.python_path, "limits": asdict(self.limits),
                "network": "none", "user": "65534:65534", "read_only": True,
                "mounts": ["temporary program.py and runner.py (read-only)"],
                "environment": "empty except fixed PATH/HOME/LANG", "pull": "never"}

    def _command(self, args: list[str], timeout: float) -> CommandResult:
        try:
            result = self._runner(self._prefix + args, timeout, self.limits.output_bytes)
            if not isinstance(result, CommandResult):
                return CommandResult("runner_error", None, "invalid command result")
            # Keep fake/custom runner results bounded too.
            return CommandResult(result.status, result.returncode,
                                 _bounded_text(result.output.encode("utf-8", "replace"), self.limits.output_bytes))
        except Exception:
            # Exception messages from Docker clients can contain host paths or
            # credentials. Deliberately do not serialize arbitrary exception text.
            return CommandResult("runner_error", None, "Docker control command failed")

    def available(self) -> bool:
        """Inspect only the requested local image; never resolve tags or pull."""
        result = self._command(["image", "inspect", "--format", "{{.Id}}", self.image],
                               self.limits.control_seconds)
        return result.status == "exited" and result.returncode == 0 and result.output.strip() == self.image

    def require_available(self) -> None:
        """Check image, interpreter, protocol and cleanup before loading weights."""
        result = self.run("assert 1 + 1 == 2\n")
        if result.passed is not True:
            raise ExecutorUnavailable(f"Python verifier preflight failed: {result.outcome}; no pull or host fallback")

    def run(self, program: str) -> PythonResult:
        """Execute a bounded job and fail closed on every cleanup/control failure."""
        started = time.monotonic()
        identity = self.identity()
        if not isinstance(program, str):
            return PythonResult(None, "invalid_request", None, identity)
        try:
            source = program.encode("utf-8")
        except UnicodeEncodeError:
            return PythonResult(None, "invalid_request", None, identity)
        source_hash = hashlib.sha256(source).hexdigest()
        if not source or len(source) > self.limits.source_bytes or b"\x00" in source:
            return PythonResult(None, "invalid_request", source_hash, identity)
        if not self.available():
            return PythonResult(None, "image_unavailable", source_hash, identity,
                                elapsed_seconds=time.monotonic() - started)

        name = f"coder-python-{uuid.uuid4().hex}"
        result = CommandResult("runner_error", None)
        cleanup = CommandResult("runner_error", None)
        with tempfile.TemporaryDirectory(prefix="coder-python-") as directory:
            job = Path(directory)
            job.chmod(0o755)
            (job / "program.py").write_bytes(source)
            (job / "runner.py").write_text(_WRAPPER, encoding="utf-8")
            for path in job.iterdir():
                path.chmod(0o444)
            cpu = max(1, math.ceil(self.limits.wall_seconds))
            args = ["run", "--pull=never", "--name", name, "--network=none", "--ipc=none",
                    "--read-only", "--user=65534:65534", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", f"--memory={self.limits.memory_mib}m",
                    f"--memory-swap={self.limits.memory_mib}m", f"--cpus={self.limits.cpus:g}",
                    f"--pids-limit={self.limits.pids}", "--ulimit", f"cpu={cpu}:{cpu}",
                    "--ulimit", "fsize=1048576:1048576", "--ulimit", "nofile=64:64",
                    "--ulimit", "core=0:0", "--log-driver=none", "--stop-timeout=0",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
                    "--workdir=/job", "--mount", f"type=bind,src={job},dst=/job,readonly",
                    "--entrypoint=/usr/bin/env", self.image, "-i", "PATH=/usr/local/bin:/usr/bin:/bin",
                    "HOME=/tmp", "LANG=C.UTF-8", self.python_path, "-I", "-B", "/job/runner.py"]
            try:
                result = self._command(args, self.limits.wall_seconds)
            finally:
                cleanup = self._command(["rm", "--force", name], self.limits.control_seconds)
            output = result.output.replace(directory, "<temporary-job>")
        cleaned = cleanup.status == "exited" and cleanup.returncode == 0
        outcome, passed = result.status, None
        if outcome == "exited":
            if result.returncode == 0:
                if output.endswith(_COMPLETED):
                    outcome, passed = "passed", True
                    output = output[:-len(_COMPLETED)]
                else:
                    outcome = "protocol_error"
            elif result.returncode in (125, 126, 127):
                outcome = "runner_error"
            else:
                outcome, passed = "failed", False
        elif outcome in ("timeout", "output_limit"):
            passed = False
        else:
            outcome = "runner_error"
        if not cleaned:
            outcome, passed = "cleanup_failed", None
        return PythonResult(passed, outcome, source_hash, identity, output=output,
                            returncode=result.returncode, cleanup="removed" if cleaned else "failed",
                            elapsed_seconds=time.monotonic() - started)

    def run_many(self, programs: list[str]) -> list[PythonResult]:
        """Execute sequentially with one isolated container per candidate."""
        return [self.run(program) for program in programs]


def executor_from_config(cfg: dict) -> DockerPythonExecutor:
    """Strict adapter for the shared `execution` YAML block."""
    if not isinstance(cfg, dict):
        raise ValueError("execution must configure an explicitly trusted local Docker image")
    allowed = {"backend", "image", "python_path", "docker_socket", "limits"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"unsupported execution options: {sorted(unknown)}")
    if cfg.get("backend", "docker") != "docker":
        raise ValueError("only the fail-closed Docker backend is supported for generated Python")
    if not cfg.get("image"):
        raise ValueError("execution.image must be a trusted local immutable sha256 image ID")
    return DockerPythonExecutor(cfg["image"], limits=PythonLimits(**cfg.get("limits", {})),
                                python_path=cfg.get("python_path", "/usr/local/bin/python3"),
                                docker_socket=cfg.get("docker_socket", "/var/run/docker.sock"))
