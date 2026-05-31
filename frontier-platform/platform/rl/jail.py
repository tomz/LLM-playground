"""Jailed sandbox wrap for RLVR code execution.

The existing :mod:`platform.rl.sandbox` runs untrusted model code in a
subprocess with POSIX rlimits (CPU / memory / output) and a wall-clock
timeout. That stops runaway / OOM / crash code from taking down the trainer,
but the file's docstring openly admits it is **not a complete jail**: a
candidate solution can still read host files (the trainer's checkpoints, the
operator's SSH keys, the cluster's /etc/shadow) and open arbitrary network
sockets.

This module closes both gaps by wrapping the subprocess invocation in a
process-jailer that gives the child its own filesystem and network
namespaces. We support three jailers in priority order:

* **bubblewrap** (``bwrap``) — userspace, available everywhere
  Flatpak / Steam runs, no SUID needed in most distros.
* **nsjail** — Google's hardened jailer, the standard at the top of every
  CTF and Kubernetes-isolation stack.
* **firejail** — well-known Linux jail, SUID-installed on most desktops.

We auto-detect at module load and prefer ``bwrap`` (least-privilege, most
portable). When none of the three is available, :func:`detect_jailer`
returns :class:`NoJailer` (a passthrough) and emits a one-time warning —
the existing rlimit sandbox is still applied, so untrusted code is still
prevented from looping forever or eating all memory.

Public API:
    :class:`Jailer`              — protocol every jailer implements.
    :class:`NoJailer`            — passthrough (no jail).
    :class:`BubblewrapJailer`,
    :class:`NsjailJailer`,
    :class:`FirejailJailer`      — concrete jailers.
    :func:`detect_jailer`        — auto-detect the best available jailer.
    :func:`run_in_jailed_sandbox` — drop-in for ``run_in_sandbox`` that adds
                                    a filesystem + network namespace.

``run_in_sandbox`` itself now grows an optional ``jailer=`` keyword so the
existing verifier (:class:`platform.rl.verifiers.CodeUnitTestVerifier`) can
opt into a jail without changing its signature or call shape.
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol


def _interp_bind_paths(argv: list[str]) -> list[str]:
    """Return host paths that must be read-only-bind-mounted so ``argv[0]``
    can actually exec inside the jail.

    Handles three common cases:
    * argv[0] is an absolute interpreter path (most common: ``sys.executable``).
    * argv[0] is a venv's python, which is usually a symlink to the system
      interpreter — we follow the symlink so the real binary is reachable too.
    * argv[0] is just a name like ``"python3"`` — caller handles PATH lookup,
      but we still expose the resolved location for safety.

    Returned paths are *directories* (the interpreter's parent + the parent of
    its symlink target). Callers add them to the jailer's bind-mount list.
    """
    out: set[str] = set()
    if not argv:
        return []
    candidate = shutil.which(argv[0]) or argv[0]
    if not os.path.isabs(candidate):
        return []
    p = Path(candidate)
    if p.exists():
        out.add(str(p.parent))
        # Follow symlinks (a venv's python is almost always a symlink).
        try:
            real = p.resolve()
            if real != p and real.exists():
                out.add(str(real.parent))
        except OSError:
            pass
    return sorted(out)


class Jailer(Protocol):
    """Wrap an argv with a process-jailer command.

    Implementations should return a command whose *innermost* exec is the
    original ``argv`` (typically ``[python, -I, -S, script_path]``). The
    jailer is responsible for:

    * giving the child a fresh PID namespace so signal escape is impossible,
    * blocking network access (no routes; ideally a disjoint net namespace),
    * confining the filesystem to a read-only view of the host's libraries
      + a writable bind of ``workdir`` (the candidate's tmpdir), and
    * propagating SIGTERM if the parent dies.

    ``workdir`` is bind-mounted *at its host path inside the jail* so the
    caller doesn't have to rewrite paths inside the script.
    """

    name: str
    available: bool

    def wrap(self, argv: list[str], *, workdir: str) -> list[str]: ...


# ----------------------------------------------------------------------------
# Concrete jailers
# ----------------------------------------------------------------------------


@dataclass
class NoJailer:
    """Passthrough: returns the original argv unchanged.

    Used when no real jailer is installed. The rlimit sandbox in
    :mod:`platform.rl.sandbox` is still applied, so this is the same posture
    the code shipped with before — just made explicit rather than implicit.
    """

    name: str = "none"
    available: bool = True

    def wrap(self, argv: list[str], *, workdir: str) -> list[str]:
        return list(argv)


@dataclass
class BubblewrapJailer:
    """Wrap argv with ``bwrap``.

    Strategy: unshare every namespace we can (network, IPC, PID, UTS,
    cgroup, user). Bind-mount common system directories read-only so Python
    and its stdlib are reachable, then bind-mount the workdir read-write at
    its host path. New /tmp + /proc + /dev so the candidate cannot see host
    processes or devices, and ``--die-with-parent`` so a trainer crash takes
    the jailed child down too. ``extra_ro_binds`` lets callers expose the
    interpreter's own directory (e.g. a venv's bin/) which isn't under /usr.
    """

    name: str = "bwrap"
    extra_ro_binds: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = shutil.which("bwrap")
        self.available = self.path is not None

    def wrap(self, argv: list[str], *, workdir: str) -> list[str]:
        if not self.available:
            raise RuntimeError("bwrap not available")
        ro_binds: list[str] = []
        # Default host system paths needed for a vanilla Python to start.
        for p in (
            "/usr", "/lib", "/lib32", "/lib64", "/bin", "/sbin",
            "/etc/alternatives", "/etc/ld.so.cache",
        ):
            ro_binds.extend(["--ro-bind-try", p, p])
        # Expose the interpreter's directory (and the resolved-symlink target).
        seen: set[str] = set()
        for p in (*self.extra_ro_binds, *_interp_bind_paths(argv)):
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            ro_binds.extend(["--ro-bind-try", p, p])
        return [
            self.path,
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-net",
            "--unshare-uts",
            "--unshare-cgroup-try",
            "--die-with-parent",
            "--new-session",
            *ro_binds,
            # Pseudo-filesystems the child needs to start.
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            # Writable workdir at its host path so the script's path is unchanged.
            "--bind", workdir, workdir,
            "--chdir", workdir,
            # Clear inheritable / file caps before exec.
            "--cap-drop", "ALL",
            "--",
            *argv,
        ]


@dataclass
class NsjailJailer:
    """Wrap argv with ``nsjail``.

    Uses ``-Mo`` (one-shot mode), unshares every namespace, mounts a
    read-only ``/usr`` + read-write workdir, kills the child after
    ``wall_seconds`` (kept in sync with ``SandboxLimits.wall_seconds`` by
    the caller via ``-t``). ``extra_ro_binds`` works the same as for bwrap.
    """

    name: str = "nsjail"
    extra_ro_binds: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = shutil.which("nsjail")
        self.available = self.path is not None

    def wrap(self, argv: list[str], *, workdir: str) -> list[str]:
        if not self.available:
            raise RuntimeError("nsjail not available")
        ro_mounts: list[str] = []
        for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/ld.so.cache"):
            if os.path.exists(p):
                ro_mounts.extend(["--bindmount_ro", p])
        seen: set[str] = set()
        for p in (*self.extra_ro_binds, *_interp_bind_paths(argv)):
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            ro_mounts.extend(["--bindmount_ro", p])
        return [
            self.path,
            "-Mo",                      # one-shot mode
            "--disable_proc",
            "--really_quiet",
            "--user", "99999",
            "--group", "99999",
            *ro_mounts,
            # Writable workdir at its host path.
            "--bindmount", f"{workdir}:{workdir}",
            "--cwd", workdir,
            # Net is unshared (no routes); IPC + PID + UTS likewise unshared by default.
            "--",
            *argv,
        ]


@dataclass
class FirejailJailer:
    """Wrap argv with ``firejail``.

    Defensive defaults: ``--noprofile`` (don't pick up a system profile),
    ``--net=none`` (no network), ``--private-tmp``, ``--read-only=/``,
    ``--whitelist=workdir`` to expose only the candidate's tmpdir. Any
    ``extra_ro_binds`` are added to the whitelist so the interpreter is
    reachable.
    """

    name: str = "firejail"
    extra_ro_binds: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = shutil.which("firejail")
        self.available = self.path is not None

    def wrap(self, argv: list[str], *, workdir: str) -> list[str]:
        if not self.available:
            raise RuntimeError("firejail not available")
        whitelist: list[str] = [f"--whitelist={workdir}"]
        seen: set[str] = set()
        for p in (*self.extra_ro_binds, *_interp_bind_paths(argv)):
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            whitelist.append(f"--whitelist={p}")
        return [
            self.path,
            "--quiet",
            "--noprofile",
            "--net=none",
            "--private-tmp",
            "--ipc-namespace",
            "--caps.drop=all",
            "--seccomp",
            "--nonewprivs",
            "--read-only=/",
            *whitelist,
            "--",
            *argv,
        ]


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------


# Preference order: bwrap (most portable, least privilege) → nsjail → firejail.
_JAILER_ORDER: tuple[type[Jailer], ...] = (
    BubblewrapJailer,
    NsjailJailer,
    FirejailJailer,
)


@lru_cache(maxsize=1)
def detect_jailer() -> Jailer:
    """Return the highest-priority jailer that is installed.

    Falls back to :class:`NoJailer` with a one-time warning when none of the
    real jailers are available. The result is cached, so re-installing a
    jailer at runtime requires calling :func:`detect_jailer.cache_clear`.
    """
    for cls in _JAILER_ORDER:
        candidate = cls()
        if candidate.available:
            return candidate
    warnings.warn(
        "No process jailer (bwrap/nsjail/firejail) found on PATH. "
        "Falling back to the rlimit-only sandbox — untrusted code can still "
        "read host files and open network sockets. Install bubblewrap "
        "(`apt install bubblewrap`) or nsjail for production use.",
        RuntimeWarning,
        stacklevel=2,
    )
    return NoJailer()


def available_jailers() -> dict[str, bool]:
    """Diagnostic: what jailers are currently visible on PATH?"""
    return {cls().name: cls().available for cls in _JAILER_ORDER}


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def run_in_jailed_sandbox(solution_code: str, test_code: str,
                          limits=None, *, jailer: Jailer | None = None):
    """Drop-in for ``run_in_sandbox`` that wraps the child in a real jail.

    Adds a network namespace (no routes), a fresh PID namespace (the child
    can't signal the parent's process group), and a filesystem view limited
    to ``/usr``, the Python stdlib, and the candidate's own tmpdir. The
    rlimit caps (CPU / memory / output) and the wall-clock timeout from
    :mod:`platform.rl.sandbox` still apply.

    Pass ``jailer=NoJailer()`` to opt back to the original behaviour
    (useful for differential tests that want to compare jailed vs. unjailed
    outputs).
    """
    # Local import to avoid a circular import: sandbox.py imports from us
    # only lazily via the optional ``jailer=`` argument.
    from .sandbox import run_in_sandbox

    if jailer is None:
        jailer = detect_jailer()

    # Make sure the jailer can find sys.executable inside its FS view.
    if hasattr(jailer, "extra_ro_binds"):
        for p in _interp_bind_paths([sys.executable]):
            if p not in jailer.extra_ro_binds:
                jailer.extra_ro_binds.append(p)

    return run_in_sandbox(solution_code, test_code, limits=limits, jailer=jailer)


__all__ = [
    "Jailer",
    "NoJailer",
    "BubblewrapJailer",
    "NsjailJailer",
    "FirejailJailer",
    "detect_jailer",
    "available_jailers",
    "run_in_jailed_sandbox",
]
