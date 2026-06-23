"""Keep/revert driver — the autonomous research loop.

One iteration:
  1. run harness.py on the current candidate.py (fixed budget, GPU)
  2. gate: if not ok → REVERT candidate.py and log status=crash
  3. compare val_bpb against the running best in ledger.tsv:
       improved → KEEP candidate.py (new revert target) and log status=keep
       worse    → REVERT candidate.py                    and log status=discard
  4. append a row to ledger.tsv and refresh progress.png

An agent edits candidate.py *between* iterations; you run `python loop.py` after
each edit (or wrap it). `--auto N` runs N iterations applying a built-in random
knob mutation each time, so the loop is demonstrably autonomous without an LLM
in the seat (handy for CI/smoke + a baseline to beat).

Keep/revert substrate: **git** is the cross-process source of truth (the
autokernel/autoresearch mechanic). The agent edits `candidate.py` and runs this
script as a *fresh process*, so the "last good" state must live somewhere
durable: kept edits are real commits, and a discard/crash is
`git checkout -- candidate.py` back to the last commit. The one exception is
`--auto --no-git`, which runs all iterations inside a single process and so can
hold the last-good snapshot in memory. The ledger is the human-readable audit
trail the chart reads.
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "candidate.py"
LEDGER = HERE / "ledger.tsv"
FIELDS = ["experiment", "val_bpb", "tok_per_s", "vram_mb", "params_m",
          "tokens", "gen_gap", "status", "description"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                          text=True, check=False).stdout.strip()


def _candidate_tracked() -> bool:
    """True iff candidate.py is tracked by git (so checkout-revert can work).
    `git ls-files <path>` echoes the path when tracked, nothing otherwise."""
    return bool(_git("ls-files", str(CANDIDATE.relative_to(HERE))))


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def running_best(rows: list[dict]) -> float:
    best = float("inf")
    for r in rows:
        if r.get("status") == "keep":
            try:
                best = min(best, float(r["val_bpb"]))
            except (ValueError, KeyError):
                pass
    return best


def append_row(row: dict) -> None:
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def candidate_description() -> str:
    """Pull DESCRIPTION out of candidate.py without importing torch."""
    import ast
    tree = ast.parse(CANDIDATE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DESCRIPTION":
                    return ast.literal_eval(node.value)
    return "(no description)"


def run_once(*, tokens: int, minutes: float | None, seed: int, keep_git: bool) -> dict:
    """Run the harness on the current candidate and return its Result dict."""
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        out_json = tf.name
    cmd = [sys.executable, str(HERE / "harness.py"),
           "--candidate", str(CANDIDATE), "--seed", str(seed),
           "--json-out", out_json]
    if minutes is not None:
        cmd += ["--minutes", str(minutes)]
    else:
        cmd += ["--tokens", str(tokens)]
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # A crash (bad patch, OOM, syntax error in candidate) → not ok.
        return {"ok": False, "reason": f"harness crashed (rc={proc.returncode})",
                "stderr": proc.stderr[-800:]}
    return json.loads(Path(out_json).read_text())


def keep(desc: str, *, use_git: bool, snapshot: dict) -> None:
    """Record a kept edit as the new revert target.

    * git mode (default): commit candidate.py so a later fresh-process run can
      `git checkout` back to it. Requires the file be tracked.
    * in-memory fallback (--no-git, single process): stash the text in
      `snapshot` for `restore()` to write back.
    """
    snapshot["text"] = CANDIDATE.read_text()
    if use_git:
        _git("add", str(CANDIDATE.relative_to(HERE)))
        _git("commit", "-q", "-m", f"autoresearch keep: {desc}")


def restore(*, use_git: bool, snapshot: dict) -> None:
    """Revert candidate.py to the last KEPT state (a discard/crash).

    git mode rolls the file back to HEAD (the last kept commit); the in-memory
    fallback rewrites the stashed text. Both are correct because `keep()`
    advanced the target only on an improvement."""
    if use_git:
        _git("checkout", "--", str(CANDIDATE.relative_to(HERE)))
    elif snapshot.get("text") is not None:
        CANDIDATE.write_text(snapshot["text"])


def refresh_plot() -> None:
    subprocess.run([sys.executable, str(HERE / "plot.py")], cwd=HERE,
                   capture_output=True, text=True, check=False)


def _random_mutation(seed: int) -> str:
    """Apply a small, reversible knob tweak to candidate.py — a stand-in 'agent'
    so the loop is autonomous in CI without an LLM. Edits the source text so the
    change is a real git diff the keep/revert path acts on."""
    rng = random.Random(seed)
    src = CANDIDATE.read_text()
    choices = [
        ('"dropout": 0.1', '"dropout": 0.0', "dropout 0.1→0.0"),
        ('# "qk_norm": True', '"qk_norm": True', "enable qk_norm"),
        ('# "zero_init_proj": True', '"zero_init_proj": True', "enable zero_init_proj"),
        ('"optimizer": "adamw"', '"optimizer": "muon"', "AdamW→Muon"),
        ('"lr": 1.0e-3', '"lr": 1.5e-3', "lr 1e-3→1.5e-3"),
    ]
    rng.shuffle(choices)
    for old, new, label in choices:
        if old in src and new not in src:
            src = src.replace(old, new, 1)
            src = src.replace(f'DESCRIPTION = "{candidate_description()}"',
                              f'DESCRIPTION = "{label}"', 1)
            CANDIDATE.write_text(src)
            return label
    return "(no mutation available)"


def loop(args) -> None:
    rows = read_ledger()
    n = len(rows)
    iters = args.auto if args.auto else 1
    # Decide the revert substrate. git is the durable, cross-process default; we
    # fall back to an in-memory snapshot for --no-git (single-process) runs, or
    # automatically if candidate.py isn't tracked yet (git-checkout would no-op).
    use_git = not args.no_git
    if use_git and not _candidate_tracked():
        print("[loop] note: candidate.py is untracked — using in-memory revert "
              "for this run (commit research/ to enable durable git keep/revert).")
        use_git = False
    snapshot: dict = {"text": CANDIDATE.read_text()}  # in-memory fallback target
    for i in range(iters):
        if args.auto and i > 0:
            label = _random_mutation(args.seed + i)
            print(f"\n=== auto-mutation {i + 1}/{iters}: {label} ===")
        elif args.auto:
            print(f"\n=== baseline (no mutation) {i + 1}/{iters} ===")
        desc = candidate_description()
        res = run_once(tokens=args.tokens, minutes=args.minutes,
                       seed=args.seed, keep_git=use_git)
        n += 1
        best = running_best(rows)
        if not res.get("ok", False):
            status = "crash"
            print(f"[loop] exp {n}: CRASH/GATE-FAIL — {res.get('reason')}")
            restore(use_git=use_git, snapshot=snapshot)
        else:
            bpb = float(res["val_bpb"])
            if bpb < best - 1e-6:
                status = "keep"
                print(f"[loop] exp {n}: KEEP  val_bpb {bpb:.5f} < best "
                      f"{best if best < float('inf') else float('nan'):.5f}  ({desc})")
                keep(desc, use_git=use_git, snapshot=snapshot)
            else:
                status = "discard"
                print(f"[loop] exp {n}: discard  val_bpb {bpb:.5f} ≥ best {best:.5f}  ({desc})")
                restore(use_git=use_git, snapshot=snapshot)
        row = {"experiment": n, "description": desc, "status": status,
               **{k: res.get(k, "") for k in ("val_bpb", "tok_per_s", "vram_mb",
                                              "params_m", "tokens", "gen_gap")}}
        append_row(row)
        rows.append({**row, "status": status})
    refresh_plot()
    print(f"\n[loop] done — {len([r for r in rows if r['status']=='keep'])} kept of {len(rows)}; "
          f"best val_bpb {running_best(rows):.5f}; chart → progress.png")


def main():
    ap = argparse.ArgumentParser(description="autoresearch keep/revert loop")
    ap.add_argument("--tokens", type=int, default=2_000_000, help="per-experiment token budget")
    ap.add_argument("--minutes", type=float, default=None, help="wall-clock budget instead of tokens")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--auto", type=int, default=0,
                    help="run N iterations applying a built-in knob mutation each (no LLM needed)")
    ap.add_argument("--no-git", action="store_true",
                    help="don't git-commit kept edits (still reverts discards in-tree)")
    args = ap.parse_args()
    loop(args)


if __name__ == "__main__":
    main()
