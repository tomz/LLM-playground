"""One-off: seed ledger.tsv with a curated, fully-measured improvement ladder.

Not part of the loop — a bootstrap so `progress.png` shows real structure on a
fresh checkout. Each entry is a genuine candidate config trained by harness.py
on GPU; keep/discard follows the same running-best rule loop.py uses. Run once:

    python research/seed_ledger.py

Every number written is measured, not synthetic — this just curates the search
(a stand-in for what the agent explores) so the chart has an honest staircase
and a populated quality–cost frontier.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from loop import append_row, running_best  # noqa: E402

LEDGER = HERE / "ledger.tsv"
CANDIDATE = HERE / "candidate.py"
PRISTINE = CANDIDATE.read_text()

# A curated ladder: (description, KNOB overrides). Ordered so the agent's
# "journey" reads sensibly; keep/discard is decided by *measured* val_bpb.
LADDER: list[tuple[str, dict]] = [
    ("baseline: 6L d256, AdamW, dropout 0.1", {}),
    ("qk_norm on", {"qk_norm": True}),
    ("qk_norm + zero_init_proj", {"qk_norm": True, "zero_init_proj": True}),
    ("dropout 0.1→0.05", {"qk_norm": True, "zero_init_proj": True, "dropout": 0.05}),
    ("wider d_ffn 768→1024", {"qk_norm": True, "zero_init_proj": True, "d_ffn": 1024}),
    ("deeper 6→8 layers", {"qk_norm": True, "zero_init_proj": True, "n_layer": 8}),
    ("untie embeddings", {"qk_norm": True, "zero_init_proj": True, "n_layer": 8,
                          "tie_embeddings": False}),
    ("wider d_model 256→384", {"qk_norm": True, "zero_init_proj": True, "n_layer": 8,
                               "d_model": 384, "d_ffn": 1024}),
    ("lr 1e-3→8e-4 (big model)", {"qk_norm": True, "zero_init_proj": True, "n_layer": 8,
                                  "d_model": 384, "d_ffn": 1024, "lr": 8e-4}),
    ("MTP x2 aux heads", {"qk_norm": True, "zero_init_proj": True, "n_layer": 8,
                          "d_model": 384, "d_ffn": 1024, "lr": 8e-4, "mtp_tokens": 2}),
]


def write_candidate(desc: str, overrides: dict) -> None:
    src = PRISTINE
    src = src.replace('DESCRIPTION = "baseline: 6L d256, AdamW, dropout 0.1"',
                      f'DESCRIPTION = "{desc}"', 1)
    for key, val in overrides.items():
        # flip commented knobs on, or replace existing values
        commented = f'# "{key}":'
        if commented in src:
            # uncomment + set the chosen value
            import re
            src = re.sub(rf'#\s*"{key}":[^,\n]*,',
                         f'"{key}": {val!r},', src, count=1)
        else:
            import re
            src = re.sub(rf'("{key}":\s*)[^,\n]+(,)',
                         rf'\g<1>{val!r}\g<2>', src, count=1)
    CANDIDATE.write_text(src)


def run() -> dict:
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        out = tf.name
    cmd = [sys.executable, str(HERE / "harness.py"), "--candidate", str(CANDIDATE),
           "--tokens", "2000000", "--json-out", out]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return {"ok": False, "reason": f"rc={p.returncode}", "stderr": p.stderr[-400:]}
    return json.loads(Path(out).read_text())


def main():
    if LEDGER.exists():
        LEDGER.unlink()
    rows: list[dict] = []
    try:
        for n, (desc, ov) in enumerate(LADDER, 1):
            write_candidate(desc, ov)
            res = run()
            best = running_best(rows)
            if not res.get("ok", False):
                status = "crash"
            elif float(res["val_bpb"]) < best - 1e-6:
                status = "keep"
            else:
                status = "discard"
            row = {"experiment": n, "description": desc, "status": status,
                   **{k: res.get(k, "") for k in ("val_bpb", "tok_per_s", "vram_mb",
                                                  "params_m", "tokens", "gen_gap")}}
            append_row(row)
            rows.append({**row})
            print(f"exp {n:2d} [{status:7s}] val_bpb={res.get('val_bpb','?')} "
                  f"tok/s={res.get('tok_per_s','?')} params={res.get('params_m','?')}M  {desc}")
    finally:
        CANDIDATE.write_text(PRISTINE)  # always leave the baseline candidate in place
    print(f"\nseeded {len(rows)} rows → {LEDGER.name}; best val_bpb {running_best(rows):.5f}")


if __name__ == "__main__":
    main()
