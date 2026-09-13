# Repository maintenance tools

These utilities are independent of the five training projects. They are not a
repository-wide build or test runner.

## Safe peer synchronization

`peer_sync.py` keeps existing, clean `main` checkouts at the same commit as the
coordinator's live `origin/main`. It requires Python 3.10+, Git, and OpenSSH on
the coordinator, and Git plus a POSIX shell on each peer. No Python packages
need to be installed.

```bash
python3 tools/peer_sync.py \
  --peer i714700k --peer i78700 --peer phenomx6
```

The coordinator checkout defaults to this repository. Each peer defaults to
`~/dev-macrohard/LLM-playground`; override with `--remote-path`. Peer aliases use
your existing SSH configuration and **must already have trusted host keys**.
With no `--peer` arguments, only the coordinator checkout is updated.

Safety rules:

- Fetch one live origin snapshot and require every peer to agree before updating.
- Fast-forward only: skip ahead/diverged histories, non-`main` branches, detached
  heads, active Git operations, and any staged, unstaged, or untracked work.
- Never commit, push, stash, reset, delete files, or create missing checkouts.
- Leave ignored machine-local environments and outputs alone.
- Use bounded, noninteractive SSH/Git commands; one offline peer does not prevent
  other peers from updating. A process lock prevents overlapping coordinator runs.
- Return nonzero and log the reason if any checkout cannot be updated. A later
  invocation retries offline hosts and skipped checkouts.

**Commit and push reviewed changes to `origin/main` yourself.** The scheduled
job distributes published commits; it does not publish work on your behalf.
If a checkout is dirty or diverged, reconcile it manually rather than resetting
it. Do not point this tool at an unrelated project.

### Scheduling on macOS

A user LaunchAgent can run the command every 900 seconds with `RunAtLoad` and
`StartInterval`. Use absolute paths for the Python interpreter, script, and
repository; preserve the user's existing scheduled jobs. The coordinator must
be powered on and logged in, and SSH keys must be usable without prompting.
Sleeping/offline peers are retried on later runs; a missing checkout must be
created intentionally before it can participate.

For this LAN, the coordinator is `mbpm5`. Its local job definition, latest run
log, and recovery metadata live under `.git/peer-sync/` and
`.git/peer-sync-audit/`. The job label is `com.tomz.llm-playground.peer-sync`;
its LaunchAgents entry points to the local job definition. These are local
operational files, not credentials or shared repository configuration.

```bash
cat .git/peer-sync/latest.log
launchctl print gui/$(id -u)/com.tomz.llm-playground.peer-sync
# Stop scheduled updates in the current login session:
launchctl bootout gui/$(id -u)/com.tomz.llm-playground.peer-sync
# To also disable it at future logins, remove only its LaunchAgents symlink:
# ~/Library/LaunchAgents/com.tomz.llm-playground.peer-sync.plist
```

### Tests

Run maintenance-tool tests separately from each project's tests. They use
only disposable local repositories; they do not contact GitHub or LAN hosts.

```bash
cd tools
python3 -m unittest discover -s tests -v
```

Other existing helpers: `orchestrate.py` launches the numbered GPU examples;
`set_example_gpu.sh` adjusts their GPU selection.
