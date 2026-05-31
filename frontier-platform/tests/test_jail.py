"""Tests for the jailed sandbox wrap (platform/rl/jail.py).

These tests assert two things:

* **API surface**: argv composition, detection, registry, opt-in.
* **Actual jailing**: when a real jailer (bubblewrap / nsjail / firejail) is
  installed, network and host-filesystem access from inside the jail are
  blocked. When none is installed we skip the security assertions but still
  exercise the passthrough path.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

from platform.rl.jail import (
    BubblewrapJailer,
    FirejailJailer,
    NoJailer,
    NsjailJailer,
    available_jailers,
    detect_jailer,
    run_in_jailed_sandbox,
)
from platform.rl.sandbox import SandboxLimits, run_in_sandbox


# ---- detection / registry ---------------------------------------------------


def test_available_jailers_keys_and_types():
    av = available_jailers()
    assert set(av) == {"bwrap", "nsjail", "firejail"}
    assert all(isinstance(v, bool) for v in av.values())


def test_detect_jailer_returns_a_real_jailer_or_no_jailer():
    detect_jailer.cache_clear()
    j = detect_jailer()
    if any(available_jailers().values()):
        assert j.name in {"bwrap", "nsjail", "firejail"}
        assert j.available
    else:
        assert isinstance(j, NoJailer)


def test_no_jailer_is_passthrough():
    j = NoJailer()
    argv = ["/usr/bin/python3", "-c", "print(1)"]
    assert j.wrap(argv, workdir="/tmp/x") == argv


# ---- jailer argv composition ------------------------------------------------


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not installed")
def test_bubblewrap_argv_contains_unshare_flags():
    j = BubblewrapJailer()
    cmd = j.wrap([sys.executable, "-c", "pass"], workdir="/tmp")
    assert cmd[0] == j.path
    for flag in ("--unshare-net", "--unshare-pid", "--die-with-parent",
                 "--cap-drop"):
        assert flag in cmd, f"bwrap argv missing {flag}: {cmd}"
    # Innermost exec must be the original argv.
    assert cmd[-3:] == [sys.executable, "-c", "pass"]


@pytest.mark.skipif(not shutil.which("nsjail"), reason="nsjail not installed")
def test_nsjail_argv_contains_oneshot_and_workdir():
    j = NsjailJailer()
    cmd = j.wrap([sys.executable, "-c", "pass"], workdir="/tmp/work")
    assert cmd[0] == j.path
    assert "-Mo" in cmd
    assert "--cwd" in cmd and "/tmp/work" in cmd


@pytest.mark.skipif(not shutil.which("firejail"), reason="firejail not installed")
def test_firejail_argv_uses_net_none_and_seccomp():
    j = FirejailJailer()
    cmd = j.wrap([sys.executable, "-c", "pass"], workdir="/tmp/work")
    assert cmd[0] == j.path
    for flag in ("--net=none", "--seccomp", "--nonewprivs", "--read-only=/"):
        assert flag in cmd


def test_unavailable_jailer_raises_when_wrapped():
    """Constructing a jailer when its binary is missing leaves available=False;
    wrap() must refuse to silently no-op."""
    fake = BubblewrapJailer()
    # Force unavailable so we exercise the guard regardless of the host.
    fake.path = None
    fake.available = False
    with pytest.raises(RuntimeError):
        fake.wrap([sys.executable], workdir="/tmp")


# ---- sandbox.run_in_sandbox grew a jailer kwarg -----------------------------


def test_run_in_sandbox_accepts_jailer_kwarg_default_compat():
    """Existing callers pass no jailer; behaviour must be identical to before."""
    res = run_in_sandbox("def f():\n    return 7", "assert f() == 7")
    assert res.ok


def test_run_in_sandbox_with_no_jailer_explicit():
    res = run_in_sandbox("def f():\n    return 7", "assert f() == 7", jailer=NoJailer())
    assert res.ok


def test_run_in_sandbox_propagates_failure_under_jail():
    """A failing assert is still surfaced as a failed sandbox result when
    the child runs inside a real jail."""
    if not any(available_jailers().values()):
        pytest.skip("no jailer installed")
    res = run_in_jailed_sandbox("def f():\n    return 7", "assert f() == 8")
    assert not res.ok


# ---- security: actual confinement (skipped if no jailer) --------------------


_HAVE_REAL_JAILER = any(available_jailers().values())


@pytest.mark.skipif(not _HAVE_REAL_JAILER, reason="no real jailer installed")
def test_jail_blocks_network_access():
    """Code running inside the jail cannot open a TCP socket to the outside."""
    res = run_in_jailed_sandbox(
        """
import socket, sys
try:
    s = socket.socket()
    s.settimeout(2)
    s.connect(("8.8.8.8", 53))
    sys.exit(0)   # connect succeeded => jail leaks network
except OSError:
    sys.exit(42)  # blocked => good
""",
        "assert True",
        SandboxLimits(cpu_seconds=5, wall_seconds=10.0),
    )
    # The candidate exits 42 on success-of-blocking. Sandbox flags ok=False
    # because the test sentinel never prints (the script sys.exit's before
    # the test assertion runs), but the returncode is the signal we want.
    assert res.returncode == 42, (
        f"network not blocked by jail: rc={res.returncode} stderr={res.stderr[:200]}"
    )


@pytest.mark.skipif(not _HAVE_REAL_JAILER, reason="no real jailer installed")
def test_jail_does_not_mutate_host_filesystem(tmp_path):
    """A write attempt against a host path must not actually touch the host."""
    canary = tmp_path / "canary.txt"
    canary.write_text("original")
    original_mtime = canary.stat().st_mtime
    target = str(canary)

    res = run_in_jailed_sandbox(
        f"""
import sys
try:
    open({target!r}, "w").write("PWNED")
    sys.exit(0)   # write succeeded against host path (would mean jail leak)
except OSError:
    sys.exit(42)  # blocked
""",
        "assert True",
        SandboxLimits(cpu_seconds=5, wall_seconds=10.0),
    )
    # The host file must be byte-identical to before, regardless of what the
    # in-jail write thought it did.
    assert canary.read_text() == "original", (
        "host filesystem mutated through the jail "
        f"(jail rc={res.returncode}, stderr={res.stderr[:200]})"
    )
    assert canary.stat().st_mtime == original_mtime


@pytest.mark.skipif(not _HAVE_REAL_JAILER, reason="no real jailer installed")
def test_jail_workdir_is_writable():
    """The candidate's tmpdir is bind-mounted writable so the script itself
    can create files inside its own cwd (the verifier relies on this)."""
    res = run_in_jailed_sandbox(
        """
import os, pathlib
p = pathlib.Path(os.getcwd()) / "scratch.txt"
p.write_text("ok")
assert p.read_text() == "ok"
""",
        "assert True",
    )
    assert res.ok, res.stderr[:300]


# ---- correctness under the jail (same answers as unjailed) ------------------


@pytest.mark.skipif(not _HAVE_REAL_JAILER, reason="no real jailer installed")
def test_jailed_correct_solution_passes():
    res = run_in_jailed_sandbox(
        "def add(a, b):\n    return a + b",
        "assert add(2, 3) == 5",
    )
    assert res.ok


@pytest.mark.skipif(not _HAVE_REAL_JAILER, reason="no real jailer installed")
def test_jailed_wrong_solution_fails():
    res = run_in_jailed_sandbox(
        "def add(a, b):\n    return a - b",
        "assert add(2, 3) == 5",
    )
    assert not res.ok


@pytest.mark.skipif(not _HAVE_REAL_JAILER, reason="no real jailer installed")
def test_jailed_infinite_loop_times_out():
    res = run_in_jailed_sandbox(
        "def f():\n    while True:\n        pass\nf()",
        "assert True",
        SandboxLimits(cpu_seconds=1, wall_seconds=2.0),
    )
    assert not res.ok
    assert res.timed_out or res.returncode != 0


# ---- code verifier picks up the jail when asked -----------------------------


def test_code_verifier_can_use_jailed_sandbox():
    """The :class:`CodeUnitTestVerifier` doesn't grow a new API: callers wrap
    with ``run_in_jailed_sandbox`` directly. Verify the verifier's logic still
    works when the underlying sandbox is jailed."""
    from platform.rl.verifiers import CodeUnitTestVerifier
    v = CodeUnitTestVerifier(tests=["assert solve(4) == 16"])
    # The verifier imports run_in_sandbox lazily inside __call__ — see
    # platform/rl/verifiers.py — so monkey-patching the sandbox module works.
    import platform.rl.sandbox as sb_mod
    orig_runner = sb_mod.run_in_sandbox

    def _jailed_runner(sol, test, limits=None, *, jailer=None):
        return orig_runner(sol, test, limits=limits, jailer=jailer or detect_jailer())

    sb_mod.run_in_sandbox = _jailed_runner
    try:
        assert v("square", "def solve(n):\n    return n * n") == 1.0
    finally:
        sb_mod.run_in_sandbox = orig_runner
