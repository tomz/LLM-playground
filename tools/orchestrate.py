#!/usr/bin/env python3
"""Orchestrator: run pytest + ruff across every subproject and summarize.

Usage:
    python tools/orchestrate.py            # run everything
    python tools/orchestrate.py --tests    # tests only
    python tools/orchestrate.py --lint     # lint only
    python tools/orchestrate.py -p midgpt  # one project

Each subproject is a self-contained venv-using directory; this script does not
install anything, it just shells out to `<proj>/.venv/bin/python -m pytest`.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ["nanogpt-edu", "midgpt", "distgpt", "coder-finetune", "frontier-platform"]


def _run(cmd: list[str], cwd: Path, timeout: int = 600) -> tuple[int, str, str, float]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def _venv_python(proj_dir: Path) -> Path | None:
    p = proj_dir / ".venv" / "bin" / "python"
    return p if p.exists() else None


def run_pytest(proj: str) -> dict:
    pdir = ROOT / proj
    py = _venv_python(pdir) or Path(sys.executable)
    rc, out, err, dt = _run([str(py), "-m", "pytest", "-q", "--tb=short"], pdir)
    # Parse the summary line: "X passed, Y failed in Z.Zs"
    summary = ""
    for line in (out or err).splitlines()[::-1]:
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            break
    return {"project": proj, "rc": rc, "summary": summary, "elapsed_s": round(dt, 2)}


def run_ruff() -> dict:
    py = _venv_python(ROOT / "nanogpt-edu") or Path(sys.executable)
    rc, out, err, dt = _run([str(py), "-m", "ruff", "check", "."], ROOT)
    if rc != 0 and ("No module named" in err or "No module named" in out):
        # Fall back to system ruff.
        rc, out, err, dt = _run(["ruff", "check", "."], ROOT)
    issues = sum(1 for l in (out + err).splitlines() if " error " in l or " warning " in l)
    summary = "clean" if rc == 0 else f"{issues} issues"
    return {"project": "<root>", "rc": rc, "summary": f"ruff: {summary}", "elapsed_s": round(dt, 2),
            "stdout": out, "stderr": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true", help="run pytest only")
    ap.add_argument("--lint", action="store_true", help="run ruff only")
    ap.add_argument("-p", "--project", action="append", default=None,
                    help="only this project (repeatable)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = ap.parse_args()

    do_tests = args.tests or not args.lint
    do_lint = args.lint or not args.tests
    projects = args.project or PROJECTS

    results = []
    if do_tests:
        for proj in projects:
            if not (ROOT / proj).is_dir():
                print(f"  ✗ {proj}: missing", file=sys.stderr)
                continue
            print(f"→ pytest {proj}")
            r = run_pytest(proj)
            results.append(r)
            mark = "✓" if r["rc"] == 0 else "✗"
            print(f"  {mark} {proj:20s}  {r['summary']}  ({r['elapsed_s']}s)")

    if do_lint:
        print("→ ruff (root)")
        r = run_ruff()
        # Strip large output to keep summary terse; full output stays in r.
        summary = {k: v for k, v in r.items() if k not in ("stdout", "stderr")}
        results.append(summary)
        mark = "✓" if r["rc"] == 0 else "✗"
        print(f"  {mark} {'<root>':20s}  {r['summary']}  ({r['elapsed_s']}s)")
        if r["rc"] != 0:
            # Print first 30 lines of issues to stdout for inspection.
            lines = (r["stdout"] + r["stderr"]).splitlines()
            for line in lines[:30]:
                print(f"    {line}")
            if len(lines) > 30:
                print(f"    ... ({len(lines) - 30} more)")

    if args.json:
        print(json.dumps(results, indent=2))

    failed = [r for r in results if r["rc"] != 0]
    if failed:
        print(f"\n{len(failed)} of {len(results)} steps failed", file=sys.stderr)
        sys.exit(1)
    print(f"\nall {len(results)} steps passed ✓")


if __name__ == "__main__":
    main()
