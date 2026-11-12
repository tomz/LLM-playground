"""Legacy HOST execution for explicitly trusted test fixtures only.

Neither fork, spawn nor POSIX rlimits isolate filesystem/network/credentials.
Do not use these helpers for model outputs. HumanEval and GRPO use python_jail;
there is deliberately no command-line option selecting this legacy module.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys

DEFAULT_LIMITS = {"memory_bytes": None, "cpu_seconds": None, "max_file_bytes": 0, "max_open_fds": 64}
SPAWN_MEMORY_BYTES = 1024 * 1024 * 1024


def _apply_rlimits(limits: dict) -> None:
    """Best-effort resource controls, never a security boundary."""
    try:
        import resource
    except ImportError:
        return
    names = {"memory_bytes": "RLIMIT_AS", "cpu_seconds": "RLIMIT_CPU",
             "max_file_bytes": "RLIMIT_FSIZE", "max_open_fds": "RLIMIT_NOFILE", "max_processes": "RLIMIT_NPROC"}
    for key, name in names.items():
        value = limits.get(key)
        if value is not None and hasattr(resource, name):
            try:
                resource.setrlimit(getattr(resource, name), (int(value), int(value)))
            except (ValueError, OSError):
                pass


def _trusted_target(program: str, q, limits: dict | None, silence: bool) -> None:
    if silence:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        sys.stdout = sys.stderr = open(os.devnull, "w")
    if limits:
        _apply_rlimits(limits)
    try:
        exec(program, {})
        q.put((True, ""))
    except BaseException as exc:
        q.put((False, f"{type(exc).__name__}: {str(exc)[:200]}"))


def trusted_run_one(program: str, timeout: float = 5.0, *, limits: dict | None = None,
                    mp_mode: str | None = None, silence_output: bool = True) -> tuple[bool, str]:
    """Opt-in HOST execution of a reviewed fixture; callers assert it is trusted."""
    mode = mp_mode or ("fork" if sys.platform != "win32" else "spawn")
    ctx = mp.get_context(mode)
    effective = None
    if limits is not False:
        effective = dict(DEFAULT_LIMITS)
        effective["cpu_seconds"] = max(1, int(timeout * 2))
        if mode == "spawn":
            effective["memory_bytes"] = SPAWN_MEMORY_BYTES
        if limits:
            effective.update(limits)
    q = ctx.Queue()
    process = ctx.Process(target=_trusted_target, args=(program, q, effective, silence_output))
    process.start()
    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            return False, "timeout"
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            if process.exitcode and process.exitcode < 0:
                return False, f"killed-{signal.Signals(-process.exitcode).name}"
            return False, "no-result"
    finally:
        q.close()
        q.join_thread()
        process.close()


def trusted_run_many(programs: list[str], timeout: float = 5.0, *, max_workers: int | None = None,
                     **kwargs) -> list[tuple[bool, str]]:
    """Sequential legacy fixture execution; never used by a production reward."""
    if max_workers not in (None, 1):
        raise ValueError("trusted fixture execution is sequential; max_workers must be 1")
    return [trusted_run_one(program, timeout, **kwargs) for program in programs]
