#!/usr/bin/env python3
"""Strip duplicate / rewind-loop entries from a distgpt log.jsonl.

The training run was paused once (after the SpikeMonitor rewind-loop bug),
fixed, and resumed from step 1500. The unfixed first phase logged the same
steps (1500-1800) up to 14 times with degrading LR. Drop those duplicates,
keeping only the cleanest first-pass entries (LR matching the cosine
schedule, no rewind scaling) plus everything from the resume onwards.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "out/gpt_416m_fweb_5060ti/log.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()]

# Keep the first occurrence of each step (the pre-rewind one is the clean
# cosine-schedule entry), plus all eval_loss entries (they're idempotent
# so we'll just keep the first per step too).
seen_loss: set[int] = set()
seen_eval: set[int] = set()
out: list[dict] = []
for r in rows:
    step = int(r["step"])
    if "eval_loss" in r:
        if step in seen_eval:
            continue
        seen_eval.add(step)
    else:
        if step in seen_loss:
            continue
        seen_loss.add(step)
    out.append(r)

backup = path.with_suffix(".jsonl.raw")
if not backup.exists():
    backup.write_text(path.read_text())
    print(f"backed up raw log -> {backup}")

path.write_text("\n".join(json.dumps(r) for r in out) + "\n")
print(f"wrote {len(out)} unique-step entries to {path}  "
      f"(dropped {len(rows) - len(out)} duplicates)")
