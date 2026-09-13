"""Exercise sync safety against disposable local Git repositories, not LAN peers."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peer_sync import sync_checkout  # noqa: E402


class PeerSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull,
                                          "GIT_CONFIG_NOSYSTEM": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.origin = root / "origin.git"
        self.source = root / "source"
        self.peer = root / "peer"
        self.git(root, "init", "--bare", "--initial-branch=main", str(self.origin))
        self.git(root, "clone", str(self.origin), str(self.source))
        self.configure(self.source)
        self.commit(self.source, "base.txt", "base")
        self.git(self.source, "push", "origin", "main")
        self.git(root, "clone", str(self.origin), str(self.peer))
        self.configure(self.peer)
        self.before = self.git(self.peer, "rev-parse", "HEAD")
        self.commit(self.source, "new.txt", "upstream")
        self.git(self.source, "push", "origin", "main")
        self.expected = self.git(self.source, "rev-parse", "HEAD")

    def git(self, cwd: Path, *args: str) -> str:
        result = subprocess.run(["git", *args], cwd=cwd, check=True,
                                text=True, capture_output=True)
        return result.stdout.strip()

    def configure(self, repo: Path) -> None:
        self.git(repo, "config", "user.name", "Sync Test")
        self.git(repo, "config", "user.email", "sync@example.invalid")
        self.git(repo, "config", "commit.gpgsign", "false")

    def commit(self, repo: Path, name: str, content: str) -> None:
        (repo / name).write_text(content)
        self.git(repo, "add", "--", name)
        self.git(repo, "commit", "-m", "test fixture")

    def assert_skipped(self, expected_head: str | None = None) -> None:
        result = sync_checkout(str(self.peer), self.expected)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.peer, "rev-parse", "HEAD"), expected_head or self.before)

    def test_clean_fast_forward(self) -> None:
        result = sync_checkout(str(self.peer), self.expected)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.peer, "rev-parse", "HEAD"), self.expected)
        self.assertEqual(self.git(self.peer, "status", "--porcelain"), "")

    def test_already_synced_is_idempotent(self) -> None:
        self.assertEqual(sync_checkout(str(self.peer), self.expected).returncode, 0)
        self.assertEqual(sync_checkout(str(self.peer), self.expected).returncode, 0)

    def test_dirty_tracked_file_is_preserved(self) -> None:
        (self.peer / "base.txt").write_text("uncommitted")
        self.assert_skipped()
        self.assertEqual((self.peer / "base.txt").read_text(), "uncommitted")

    def test_staged_work_is_preserved(self) -> None:
        (self.peer / "base.txt").write_text("staged")
        self.git(self.peer, "add", "base.txt")
        self.assert_skipped()
        self.assertEqual(self.git(self.peer, "show", ":base.txt"), "staged")

    def test_untracked_file_is_preserved(self) -> None:
        (self.peer / "new.txt").write_text("local-only")
        self.assert_skipped()
        self.assertEqual((self.peer / "new.txt").read_text(), "local-only")

    def test_other_branch_is_untouched(self) -> None:
        self.git(self.peer, "switch", "-c", "feature")
        self.assert_skipped()
        self.assertEqual(self.git(self.peer, "branch", "--show-current"), "feature")

    def test_detached_head_is_untouched(self) -> None:
        self.git(self.peer, "checkout", "--detach")
        self.assert_skipped()

    def test_diverged_history_is_untouched(self) -> None:
        self.commit(self.peer, "local.txt", "private")
        self.assert_skipped(self.git(self.peer, "rev-parse", "HEAD"))

    def test_ahead_history_is_not_pushed(self) -> None:
        self.assertEqual(sync_checkout(str(self.peer), self.expected).returncode, 0)
        self.commit(self.peer, "local.txt", "private")
        self.assert_skipped(self.git(self.peer, "rev-parse", "HEAD"))
        self.assertEqual(self.git(self.source, "ls-remote", "origin", "refs/heads/main").split()[0], self.expected)

    def test_changed_origin_snapshot_is_retried(self) -> None:
        result = sync_checkout(str(self.peer), self.before)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git(self.peer, "rev-parse", "HEAD"), self.before)

    def test_in_progress_git_operation_is_preserved(self) -> None:
        marker = self.peer / ".git" / "CHERRY_PICK_HEAD"
        marker.write_text(self.before)
        self.assert_skipped()
        self.assertTrue(marker.exists())

    def test_missing_checkout_is_not_created(self) -> None:
        missing = self.peer / "missing"
        result = sync_checkout(str(missing), self.expected)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(missing.exists())

    def test_repository_subdirectory_is_refused(self) -> None:
        nested = self.peer / "nested"
        nested.mkdir()
        result = sync_checkout(str(nested), self.expected)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git(self.peer, "rev-parse", "HEAD"), self.before)

    def test_unsafe_peer_or_path_is_rejected_before_ssh(self) -> None:
        with self.assertRaises(ValueError):
            sync_checkout("dev/repo", self.expected, peer="-oProxyCommand=bad")
        with self.assertRaises(ValueError):
            sync_checkout("../other-repo", self.expected, peer="example")


if __name__ == "__main__":
    unittest.main()
