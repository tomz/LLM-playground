#!/usr/bin/env python3
"""Safely fast-forward clean main checkouts to one shared origin snapshot.

Runs on macOS/Linux with Python 3.10+ and Git/SSH. This is a maintenance tool,
not a training or test runner. It never commits, pushes, stashes, or resets.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=6", "-o", "ConnectionAttempts=1",
               "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
CHECKOUT_SCRIPT = r'''
set -eu
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=6'
repo=$1
expected=$2
if [ ! -d "$repo" ]; then echo "SKIP: checkout missing: $repo"; exit 3; fi
cd "$repo"
if [ "$(git rev-parse --show-toplevel)" != "$(pwd -P)" ]; then
    echo 'SKIP: path is not a repository root'; exit 3
fi
if [ "$(git branch --show-current)" != main ]; then
    echo 'SKIP: checkout is not on main'; exit 3
fi
for marker in MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD BISECT_LOG rebase-merge rebase-apply; do
    if [ -e "$(git rev-parse --git-path "$marker")" ]; then
        echo "SKIP: Git operation in progress: $marker"; exit 3
    fi
done
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    echo 'SKIP: uncommitted or untracked work'; exit 3
fi
git fetch --quiet --no-tags origin refs/heads/main:refs/remotes/origin/main
if [ "$(git rev-parse refs/remotes/origin/main)" != "$expected" ]; then
    echo 'SKIP: origin differs from the shared snapshot; retry next run'; exit 3
fi
if ! git merge-base --is-ancestor HEAD refs/remotes/origin/main; then
    echo 'SKIP: main is ahead or diverged; publish/reconcile it manually'; exit 3
fi
git merge --ff-only --quiet refs/remotes/origin/main
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    echo 'CHECK: work appeared during sync; leaving it untouched'; exit 3
fi
printf 'OK: main %s\n' "$(git rev-parse HEAD)"
'''


def run(command: list[str], *, cwd: Path | None = None,
        text: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a bounded, noninteractive command without hiding its failure."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
           "GIT_SSH_COMMAND": "ssh " + " ".join(SSH_OPTIONS)}
    return subprocess.run(command, cwd=cwd, input=text, text=True,
                          capture_output=True, timeout=timeout, env=env)


def sync_checkout(repo: str, expected: str, *, peer: str | None = None,
                  timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Update one local checkout, or a checkout relative to a peer's home."""
    if peer is None:
        command = ["sh", "-s", "--", repo, expected]
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", peer):
            raise ValueError("peer must be an SSH host alias")
        path = Path(repo)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("remote path must be relative to the peer's home")
        remote_command = f'sh -s -- "$HOME"/{shlex.quote(repo)} {shlex.quote(expected)}'
        command = ["ssh", *SSH_OPTIONS, peer, remote_command]
    return run(command, text=CHECKOUT_SCRIPT, timeout=timeout)


def main() -> int:
    """Sync the local checkout and explicitly selected peers; report all skips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--peer", action="append", default=[], help="SSH alias; repeatable")
    parser.add_argument("--remote-path", default="dev-macrohard/LLM-playground")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo = args.repo.resolve()
    try:
        lock_result = run(["git", "rev-parse", "--git-path", "peer-sync.lock"], cwd=repo)
        lock_result.check_returncode()
        lock_path = repo / lock_result.stdout.strip()
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                LOG.warning("Another sync is running; skipping this invocation")
                return 1
            source = run(["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"], cwd=repo)
            source.check_returncode()
            expected = source.stdout.split()[0]
            if not re.fullmatch(r"[0-9a-f]{40,64}", expected):
                raise ValueError("origin did not return a valid main commit")
            LOG.info("Shared origin/main snapshot: %s", expected)
            targets = [("local", str(repo), None)] + [
                (peer, args.remote_path, peer) for peer in dict.fromkeys(args.peer)
            ]
            failures = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = {
                    pool.submit(sync_checkout, path, expected, peer=peer): label
                    for label, path, peer in targets
                }
                for future in concurrent.futures.as_completed(futures):
                    label = futures[future]
                    try:
                        result = future.result()
                        message = (result.stdout + result.stderr).strip()
                        if result.returncode:
                            failures += 1
                            LOG.warning("%s: %s", label, message)
                        else:
                            LOG.info("%s: %s", label, message)
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        failures += 1
                        LOG.warning("%s: %s", label, exc)
            return int(failures > 0)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        LOG.error("Sync aborted safely: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
