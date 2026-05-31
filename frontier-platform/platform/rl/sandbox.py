"""Sandboxed execution of untrusted model-generated code for RLVR verifiers.

The RLVR code verifier must run model output against hidden unit tests. That
output is *untrusted* — it can loop forever, allocate all memory, or try to read
files / open sockets. This module runs it in a **separate OS process** with hard
resource limits (CPU seconds, address space, file size) via the POSIX
``resource`` module, a wall-clock timeout, and a scrubbed environment.

Security posture (be honest about it):
  * This is a *real* in-process-free sandbox: a crashing/looping/OOMing solution
    cannot take down the trainer, and rlimits cap CPU/memory/output.
  * It is **not** a complete jail. A determined adversary could still touch the
    filesystem (read-only of the host) since we don't namespace the FS or block
    syscalls. Production must wrap this in gVisor / Firecracker / nsjail and a
    network namespace with no routes. The interface here (submit code + tests,
    get a pass-fraction) is identical, so swapping the backend is a config change.

On non-POSIX platforms (no ``resource``), we still run in a subprocess with a
timeout but without rlimits, and emit a one-time warning.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import warnings
from dataclasses import dataclass

try:
    import resource  # POSIX only
    _HAS_RESOURCE = True
except ImportError:  # pragma: no cover - non-POSIX
    _HAS_RESOURCE = False


@dataclass
class SandboxLimits:
    cpu_seconds: int = 5            # hard CPU-time limit (SIGXCPU)
    wall_seconds: float = 10.0     # wall-clock timeout (subprocess.kill)
    address_space_mb: int = 512    # max virtual memory
    output_bytes: int = 64_000     # cap stdout/stderr capture


@dataclass
class SandboxResult:
    ok: bool                       # process exited 0 (all tests passed)
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


_RUNNER_TEMPLATE = """\
import sys

# --- candidate solution ---
{solution}

# --- hidden tests (each must raise on failure) ---
{tests}
print("__ALL_TESTS_PASSED__")
"""


def _preexec(limits: SandboxLimits):  # pragma: no cover - runs in child
    """Apply rlimits in the child before exec. POSIX only."""
    if not _HAS_RESOURCE:
        return
    # CPU time (seconds) — SIGXCPU then SIGKILL.
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    # Address space (bytes).
    nbytes = limits.address_space_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ValueError, OSError):
        pass
    # Max output file size.
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE,
                           (limits.output_bytes * 4, limits.output_bytes * 4))
    except (ValueError, OSError):
        pass
    # New session so we can't signal the parent's process group.
    os.setsid()


def run_in_sandbox(solution_code: str, test_code: str,
                   limits: SandboxLimits | None = None,
                   *, jailer=None) -> SandboxResult:
    """Run ``solution_code`` then ``test_code`` in a limited subprocess.

    Tests should ``assert`` / raise on failure (pytest-free). Success is signalled
    by a clean exit-0 with the sentinel line printed. Returns a
    :class:`SandboxResult`.

    Pass ``jailer=`` (a :class:`platform.rl.jail.Jailer`) to wrap the child in
    a bubblewrap / nsjail / firejail process jail with filesystem + network
    isolation. By default this stays at the original (rlimit-only) behaviour
    so existing callers and tests are unchanged; new code should prefer
    :func:`platform.rl.jail.run_in_jailed_sandbox`, which auto-detects the
    best available jailer.
    """
    limits = limits or SandboxLimits()
    if not _HAS_RESOURCE:
        warnings.warn(
            "resource module unavailable (non-POSIX): running code without "
            "rlimits, timeout only. Do NOT use for real untrusted code here.",
            stacklevel=2,
        )

    script = _RUNNER_TEMPLATE.format(
        solution=textwrap.dedent(solution_code),
        tests=textwrap.dedent(test_code),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        # Scrubbed environment: no inherited secrets, restricted PATH.
        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            # Hint to libraries not to spin up thread pools that eat the CPU limit.
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        preexec = (lambda: _preexec(limits)) if _HAS_RESOURCE else None
        argv = [sys.executable, "-I", "-S", path]   # -I isolate, -S no site
        if jailer is not None:
            argv = jailer.wrap(argv, workdir=tmp)
        try:
            proc = subprocess.run(
                argv,
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=limits.wall_seconds,
                preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired as e:
            return SandboxResult(
                ok=False, returncode=-1,
                stdout=(e.stdout or "")[: limits.output_bytes] if isinstance(e.stdout, str) else "",
                stderr="timeout", timed_out=True,
            )

        out = proc.stdout[: limits.output_bytes]
        err = proc.stderr[: limits.output_bytes]
        ok = proc.returncode == 0 and "__ALL_TESTS_PASSED__" in out
        return SandboxResult(ok=ok, returncode=proc.returncode, stdout=out, stderr=err)
