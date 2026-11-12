"""Fail-closed Docker execution for a small, closed Lean certificate language.

The host contract and the pinned image are trusted; submissions are not. This
is not an arbitrary Lean source runner. Docker is a boundary, not a VM: use a
separate worker or stronger runtime for hostile multi-tenant workloads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

# No identifiers, comments, macros, imports or user-defined tactics can enter
# through the certificate. In particular, do NOT replace this with a denylist.
CERTIFICATES = frozenset({"by rfl", "by decide", "by intro n; rfl", "by assumption"})


def digest(text: str) -> str:
    """Hash exact UTF-8 bytes, without whitespace normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CommandResult:
    status: str
    returncode: int | None
    output: str


def bounded_command(args: list[str], timeout: float, output_limit: int) -> CommandResult:
    """Drain merged output incrementally; kill on wall time or byte overflow.

    Linux/POSIX only. Never invoke a shell and never buffer unlimited output.
    Killing the Docker CLI does not kill its container; the adapter separately
    removes its uniquely named container in a finally block.
    """
    started = time.monotonic()
    output = bytearray()
    try:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        return CommandResult("runner_error", None, str(exc)[:output_limit])
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
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()
        proc.stdout.close()
    return CommandResult(status, proc.returncode, bytes(output[:output_limit]).decode("utf-8", "replace"))


@dataclass(frozen=True)
class LeanContract:
    """Host-owned proposition and explicit premises (named h0, h1, ...).

    Types are trusted Lean source, NOT copied from model output. A reviewer
    must inspect this contract for vacuous/contradictory premises and imports.
    Only Init is imported by the wrapper; arbitrary Mathlib proofs are out of
    scope. Hashes establish identity, not the soundness of host input or image.
    """
    theorem: str
    premises: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.theorem, str) or not self.theorem or len(self.theorem) > 4096:
            raise ValueError("host theorem must contain 1..4096 characters")
        if not isinstance(self.premises, tuple) or len(self.premises) > 16 or any(
            not isinstance(p, str) or not p or len(p) > 4096 for p in self.premises
        ):
            raise ValueError("host premises must be a tuple of at most 16 nonempty types")

    def source(self, certificate: str) -> str:
        """Render only an allowlisted certificate into the exact host contract."""
        if certificate not in CERTIFICATES:
            raise ValueError("certificate outside the closed tactic subset")
        premises = "".join(f" (h{i} : {p})" for i, p in enumerate(self.premises))
        return ("import Init\nset_option maxHeartbeats 100000\n"
                "set_option maxRecDepth 512\n"
                f"theorem certified{premises} : {self.theorem} := {certificate}\n"
                "#print axioms certified\n")


@dataclass(frozen=True)
class LeanResult:
    valid: bool
    outcome: str
    theorem_sha256: str
    premises_sha256: str
    certificate_sha256: str
    source_sha256: str | None
    image: str
    checker_sha256: str
    limits: dict
    command: list[str]
    output: str = ""
    returncode: int | None = None
    cleanup: str = "not_needed"


class DockerLeanChecker:
    """Callable adapter compatible with structured_proof.CertificateChecker."""

    def __init__(self, contract: LeanContract, image: str, *, timeout: float = 10,
                 output_limit: int = 16384, lean_path: str = "/opt/lean/bin/lean",
                 runner: Callable[[list[str], float, int], CommandResult] = bounded_command):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("image must be an explicit local immutable sha256 image ID")
        if not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError("timeout must be in (0, 30] seconds")
        if type(output_limit) is not int or not 256 <= output_limit <= 65536:
            raise ValueError("output_limit must be 256..65536 bytes")
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", lean_path):
            raise ValueError("lean_path must be an absolute executable path in the image")
        self.contract, self.image, self.timeout = contract, image, timeout
        self.output_limit, self.lean_path, self.runner = output_limit, lean_path, runner

    def __call__(self, theorem: str, certificate: str) -> tuple[bool, str]:
        result = self.check(theorem, certificate)
        return result.valid, json.dumps(asdict(result), sort_keys=True)

    def check(self, theorem: str, certificate: str) -> LeanResult:
        """Pin all inputs; fail closed on rejection, timeout or runner failure."""
        record = dict(
            theorem_sha256=digest(self.contract.theorem),
            premises_sha256=digest(json.dumps(self.contract.premises, ensure_ascii=False)),
            certificate_sha256=digest(certificate), source_sha256=None, image=self.image,
            checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limits={"wall_seconds": self.timeout, "control_seconds": 5,
                    "output_bytes": self.output_limit, "memory_mib": 256,
                    "cpus": 1, "pids": 32, "cpu_seconds": math.ceil(self.timeout)},
            command=[],
        )
        if theorem != self.contract.theorem:
            return LeanResult(False, "theorem_mismatch", **record)
        if certificate not in CERTIFICATES:
            return LeanResult(False, "certificate_rejected", **record)
        source = self.contract.source(certificate)
        record["source_sha256"] = digest(source)
        try:
            inspected = self.runner(["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                                    5, self.output_limit)
        except Exception as exc:
            return LeanResult(False, "runner_error", **record, output=str(exc)[:self.output_limit])
        if inspected.status != "exited" or inspected.returncode != 0 or inspected.output.strip() != self.image:
            return LeanResult(False, "image_unavailable", **record,
                              output=inspected.output, returncode=inspected.returncode)
        name = f"proof-{uuid.uuid4().hex}"
        result = CommandResult("runner_error", None, "container not started")
        cleanup = CommandResult("runner_error", None, "cleanup not attempted")
        with tempfile.TemporaryDirectory(prefix="lean-proof-") as directory:
            path = Path(directory)
            path.chmod(0o755)
            (path / "Proof.lean").write_text(source, encoding="utf-8")
            (path / "Proof.lean").chmod(0o444)
            cpu = math.ceil(self.timeout)
            command = [
                "docker", "run", "--pull=never", "--name", name,
                "--network=none", "--read-only", "--user=65534:65534", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--memory=256m", "--memory-swap=256m",
                "--cpus=1", "--pids-limit=32", "--ulimit", f"cpu={cpu}:{cpu}",
                "--ulimit", "fsize=1048576:1048576", "--log-driver=none",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--workdir=/proof",
                "--mount", f"type=bind,src={path},dst=/proof,readonly",
                "--entrypoint", self.lean_path, self.image, "--threads=1", "Proof.lean",
            ]
            record["command"] = command
            try:
                result = self.runner(command, self.timeout, self.output_limit)
            except Exception as exc:
                result = CommandResult("runner_error", None, str(exc)[:self.output_limit])
            finally:
                try:
                    cleanup = self.runner(["docker", "rm", "--force", name], 5, self.output_limit)
                except Exception as exc:
                    cleanup = CommandResult("runner_error", None, str(exc)[:self.output_limit])
        cleaned = cleanup.status == "exited" and cleanup.returncode == 0
        outcome = result.status
        if outcome == "exited":
            outcome = "accepted" if result.returncode == 0 and "sorryAx" not in result.output else "lean_rejected"
        if not cleaned and outcome == "accepted":
            outcome = "cleanup_failed"
        return LeanResult(outcome == "accepted", outcome, **record,
                          output=result.output, returncode=result.returncode,
                          cleanup="removed" if cleaned else f"failed: {cleanup.status}: {cleanup.output}")
