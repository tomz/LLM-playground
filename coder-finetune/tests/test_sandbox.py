"""Tier 8.2 tests: subprocess hardening + parallel reward executor.

Pins:
  * stdout/stderr from generated code are silenced (so a noisy ``print(...)``
    in a model completion can't flood the trainer log)
  * ``RLIMIT_FSIZE=0`` blocks the generated code from writing files
  * ``RLIMIT_CPU`` kills runaway CPU loops as a CPU-budget floor under the
    wall-clock timeout
  * ``spawn`` mode actually starts a fresh interpreter (no inherited heap)
  * ``run_many`` runs N programs concurrently AND preserves the result order
  * ``code_unit_test_reward`` order is preserved end-to-end even when
    interleaved with invalid (empty-test) slots
"""
from __future__ import annotations

import os
import sys
import pathlib
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.run_humaneval import (  # noqa: E402
    DEFAULT_LIMITS,
    run_many,
    run_one,
)
from cf_rl.reward import code_unit_test_reward  # noqa: E402


# ---------------------------------------------------------------------------
# Output suppression
# ---------------------------------------------------------------------------


def test_run_one_silences_child_stdout(capfd):
    """A model that emits ``print()`` floods the trainer log. The hardened
    executor must silence the child's fd 1/2 so the parent's captured output
    stays clean. We deliberately print a big banner that would be impossible
    to miss if it leaked through.

    Under pytest, fd 1 is replaced by a pipe (capfd's mechanism). The child
    fork inherits that pipe — without our os.dup2(/dev/null, 1) in
    _exec_target, the banner would appear in capfd's capture. With it, the
    child writes to /dev/null and capfd sees nothing.
    """
    prog = "print('BANNER_THAT_MUST_NOT_LEAK' * 100)\n"
    ok, msg = run_one(prog, timeout=2.0)
    assert ok, f"silenced print should still succeed; got msg={msg!r}"
    out, err = capfd.readouterr()
    assert "BANNER_THAT_MUST_NOT_LEAK" not in out
    assert "BANNER_THAT_MUST_NOT_LEAK" not in err


def test_run_one_silence_can_be_disabled(capfd):
    """``silence_output=False`` must restore the old behavior — useful for
    debugging a flaky reward (you want to see the child's traceback).
    We also pass ``limits=False`` because under pytest's capfd the inherited
    stdout points at a regular capture file; without silencing it, a write
    would trip RLIMIT_FSIZE. (In a real trainer the parent's stdout is a
    terminal/pipe, not a regular file, so this combination only matters
    inside pytest.)"""
    prog = "print('CHILD_PRINT_LEAK_OK')\n"
    ok, msg = run_one(prog, timeout=2.0, silence_output=False, limits=False)
    assert ok, f"with silence_output=False + limits=False, run should succeed; msg={msg!r}"


# ---------------------------------------------------------------------------
# File-write blocking
# ---------------------------------------------------------------------------


def test_rlimit_fsize_blocks_file_write(tmp_path):
    """RLIMIT_FSIZE=0 must stop a model from writing files. The child's
    write should fail (we report it as an err).

    Note: Python's text-mode ``open('w').write(s)`` *buffers* the data and
    the EFBIG only surfaces at flush/close — which happens after exec()
    returns, in cleanup. We use an explicit ``flush()`` call so the failure
    happens inside exec() and is reported as the run's outcome. The
    write-to-disk attempt is what we care about pinning anyway."""
    target = tmp_path / "should_not_exist.txt"
    prog = (
        f"f = open({str(target)!r}, 'w')\n"
        "f.write('pwned')\n"
        "f.flush()\n"  # forces the EFBIG to surface during exec()
        "f.close()\n"
    )
    ok, msg = run_one(prog, timeout=2.0)
    assert not ok
    # And the file truly contains no data.
    assert not target.exists() or target.read_bytes() == b"", \
        f"rlimit failed: file written with content {target.read_bytes()!r}"
    # Message should mention the failure (OSError / IOError / etc).
    assert "File too large" in msg or "OSError" in msg, msg


def test_rlimit_can_be_disabled(tmp_path):
    """``limits=False`` opts out — must allow a write that the default would
    block. Pin so the escape hatch keeps working (useful for trusted in-house
    test harnesses that need to call fixtures)."""
    target = tmp_path / "with_limits_off.txt"
    prog = f"open({str(target)!r}, 'w').write('ok')\n"
    ok, _ = run_one(prog, timeout=2.0, limits=False)
    assert ok, "limits=False should permit file write"
    assert target.exists() and target.read_text() == "ok"


# ---------------------------------------------------------------------------
# CPU/timeout
# ---------------------------------------------------------------------------


def test_infinite_loop_is_killed_by_wall_clock(capfd):
    """The wall-clock timeout remains the primary kill mechanism (rlimit
    RLIMIT_CPU is a backstop). A 1-second wall-clock cap on an infinite
    busy loop must return in roughly that time."""
    prog = "while True: pass\n"
    t0 = time.perf_counter()
    ok, msg = run_one(prog, timeout=1.0)
    dt = time.perf_counter() - t0
    assert not ok
    # tag is either "timeout" (wall-clock) or "killed-SIGXCPU" (RLIMIT_CPU)
    assert msg in ("timeout",) or "killed" in msg, msg
    assert dt < 5.0, f"took {dt:.1f}s — should have killed by 1s wall-clock"


# ---------------------------------------------------------------------------
# spawn mode actually isolates
# ---------------------------------------------------------------------------


def test_spawn_mode_child_does_not_inherit_parent_module():
    """The headline of spawn vs fork: a spawn child gets a fresh interpreter
    with no inherited imports. We can verify this by importing a sentinel
    module in the parent and checking the child can't see it via the
    ``sys.modules`` dict (it would be there under fork via COW)."""
    # The sentinel: a module the parent imports but the spawn child shouldn't.
    import re  # already imported anyway, use as proxy
    # Build a child program that checks the SENTINEL exists in its own globals.
    # The parent has a SENTINEL global below; under fork the child sees it
    # (it inherits the address space), under spawn it does not (fresh interp).
    prog = (
        "import sys\n"
        "# A *parent-only* attribute set on a builtin module won't survive\n"
        "# a spawn (fresh interp imports re anew). Under fork the attribute\n"
        "# is inherited via COW.\n"
        "import re\n"
        "found = getattr(re, '_CODER_FINETUNE_SENTINEL_8_2', None)\n"
        "assert found is None, f'spawn child saw inherited attr: {found!r}'\n"
    )
    # Plant the sentinel on the parent's re module.
    re._CODER_FINETUNE_SENTINEL_8_2 = "leaked-from-parent"
    try:
        ok, msg = run_one(prog, timeout=5.0, mp_mode="spawn")
    finally:
        del re._CODER_FINETUNE_SENTINEL_8_2
    assert ok, f"spawn child saw parent-only state: {msg}"


def test_spawn_mode_applies_memory_rlimit():
    """Spawn mode applies the memory ceiling (fork mode skips it because the
    fork child inherits the parent's address space, which may already exceed
    the cap — see DEFAULT_LIMITS comment). Allocate ~2 GiB under a 256 MiB
    cap and check it fails."""
    if sys.platform == "win32":
        pytest.skip("RLIMIT_AS not supported on Windows")
    # 256 MiB cap, then try to allocate 1 GiB worth of bytes.
    prog = (
        "buf = bytearray(1 * 1024 * 1024 * 1024)\n"  # 1 GiB
    )
    ok, msg = run_one(prog, timeout=5.0, mp_mode="spawn",
                      limits={"memory_bytes": 256 * 1024 * 1024})
    assert not ok, "allocation under tight memory cap should fail"
    # Either MemoryError (caught inside the child) or a kill-signal report.
    assert ("MemoryError" in msg or "killed" in msg), msg


# ---------------------------------------------------------------------------
# run_many: sequential wrapper that preserves order
# ---------------------------------------------------------------------------
#
# run_many is currently sequential — see its docstring for why concurrent
# fork()-from-threads + mp.Queue is unsafe under Python 3.14. These tests
# pin the order-preservation contract that callers depend on, so that any
# future parallel implementation must keep the same behavior.


def test_run_many_preserves_order():
    """If we hand programs [pass, fail, pass] we must get [(True,_), (False,_),
    (True,_)] in that exact order — any future parallel implementation must
    keep this invariant or reward indexing into the GRPO batch breaks."""
    programs = [
        "assert 1 == 1\n",      # pass
        "assert 1 == 2\n",      # fail (AssertionError)
        "assert 'a' < 'b'\n",   # pass
    ]
    results = run_many(programs, timeout=2.0)
    assert [ok for ok, _ in results] == [True, False, True]


def test_run_many_empty_list():
    """Edge case: an empty completion batch returns an empty list, not crash."""
    assert run_many([]) == []


def test_run_many_passes_kwargs_through_to_run_one():
    """``run_many`` must forward kwargs like ``limits=False`` through to
    each child. Pin so a future parallel implementation can't quietly drop
    the kwarg plumbing (and silently re-enable the rlimit floor that the
    caller deliberately opted out of)."""
    # If limits were silently re-applied we'd get EFBIG on the write.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tf:
        path = tf.name
    try:
        prog = f"f = open({path!r}, 'w'); f.write('ok'); f.flush(); f.close()\n"
        results = run_many([prog], timeout=2.0, limits=False)
        assert results == [(True, "")]
        with open(path) as f:
            assert f.read() == "ok"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# reward: end-to-end order preservation with invalid slots interleaved
# ---------------------------------------------------------------------------


_ADD_TEST = "def check(fn):\n    assert fn(1, 2) == 3\n"
_ADD_GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
_ADD_BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def test_code_reward_preserves_order_with_invalid_slots():
    """Some rows in a GRPO batch may carry empty ``test`` / ``entry_point``
    (e.g. a malformed dataset row). Those slots must return 0.0 *in place*
    so per-prompt indexing into the batch stays sound — the parallel
    executor (if any) must not reorder the runnable slots either."""
    # Pattern: [good, INVALID, bad, INVALID, good]
    completions = [_ADD_GOOD, _ADD_GOOD, _ADD_BAD, _ADD_GOOD, _ADD_GOOD]
    tests = [_ADD_TEST, "", _ADD_TEST, "", _ADD_TEST]
    entries = ["add", "", "add", "", "add"]
    stubs = ["", "", "", "", ""]
    rewards = code_unit_test_reward(
        completions=completions, test=tests, entry_point=entries,
        prompt_code=stubs,
    )
    assert rewards == [1.0, 0.0, 0.0, 0.0, 1.0]


def test_code_reward_batch_deterministic_across_runs():
    """The reward function must produce identical outputs across N runs of
    the same input. Pin so a future parallel/caching implementation can't
    silently introduce nondeterminism that would bias GRPO."""
    completions = [_ADD_GOOD, _ADD_BAD, _ADD_GOOD, _ADD_BAD, _ADD_GOOD]
    tests = [_ADD_TEST] * 5
    entries = ["add"] * 5
    stubs = [""] * 5
    first = code_unit_test_reward(completions=completions, test=tests,
                                  entry_point=entries, prompt_code=stubs)
    for _ in range(3):
        again = code_unit_test_reward(completions=completions, test=tests,
                                      entry_point=entries, prompt_code=stubs)
        assert again == first == [1.0, 0.0, 1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# DEFAULT_LIMITS smoke
# ---------------------------------------------------------------------------


def test_default_limits_keep_fork_compatible():
    """Pin: under fork, the default ``memory_bytes`` must be None so a child
    forked from a pytest+HF interpreter (which often has 5+ GiB VSZ) isn't
    SIGKILL'd immediately by RLIMIT_AS. Spawn mode bumps it back up via
    ``SPAWN_MEMORY_BYTES`` — see run_humaneval.py."""
    assert DEFAULT_LIMITS["memory_bytes"] is None
    assert DEFAULT_LIMITS["max_file_bytes"] == 0
    assert DEFAULT_LIMITS["max_open_fds"] == 64
