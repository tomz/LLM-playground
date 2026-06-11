#!/usr/bin/env python3
"""Strip duplicate / rewind-loop entries from a distgpt log.jsonl.

The 416M FineWeb-Edu run was paused once (after the SpikeMonitor
rewind-loop bug), the stability bug was fixed, and the run was resumed
from step 1500. The unfixed first phase logged the same steps (1510-1890)
up to ~8 times with a degrading LR as the buggy RewindController halved
the learning rate on every false-positive spike (0.5**6 ~= 0.016x of the
cosine value by the end of the storm).

De-dup rule: **keep the LAST occurrence of each step.** In a
pause-fix-resume log the resume is authoritative -- it re-wrote the
affected steps at the correct cosine LR, and those rows come last. The
old version of this script kept the *first* occurrence, which is correct
for steps before the spike but wrong for the steps caught mid-storm
(their first-written row was already rewound), leaving a spurious dip in
the plotted LR schedule from step ~1510 to ~1890. Keeping the last
occurrence reproduces the analytic cosine to 0.00% across every step.

The script reads from the pristine ``<log>.raw`` backup when it exists
(so re-running is idempotent and can always recover the good rows) and
never overwrites that backup. Output is sorted by step so the plotter
draws monotonic lines.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "out/gpt_416m_fweb_5060ti/log.jsonl")
backup = path.with_suffix(".jsonl.raw")

# Always clean from the pristine raw log when we have it; otherwise this is
# the first run, so the current log IS the raw source -- back it up first.
if backup.exists():
    source = backup
    print(f"reading pristine raw log <- {backup}")
else:
    source = path
    backup.write_text(path.read_text())
    print(f"backed up raw log -> {backup}")

rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]

# Keep the LAST occurrence of each step, separately for train rows (have
# "loss") and eval rows (have "eval_loss") so both series survive a step
# that logged both. dict preserves last-write-wins by reassignment.
last_train: dict[int, dict] = {}
last_eval: dict[int, dict] = {}
for r in rows:
    step = int(r["step"])
    if "eval_loss" in r:
        last_eval[step] = r
    else:
        last_train[step] = r

# Merge and sort by (step, is_eval) so train precedes eval at the same step
# and the plotted lines advance monotonically in step.
merged = list(last_train.values()) + list(last_eval.values())
merged.sort(key=lambda r: (int(r["step"]), "eval_loss" in r))

path.write_text("\n".join(json.dumps(r) for r in merged) + "\n")
print(f"wrote {len(merged)} unique-step entries to {path}  "
      f"(from {len(rows)} raw rows, dropped {len(rows) - len(merged)} "
      f"duplicate/rewound rows)")
