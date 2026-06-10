"""Tier 9 integration test: the REAL launcher -> ``cf_dist`` contract.

``test_dist_env.py`` sets ``os.environ`` in a SINGLE process and reads it back.
That pins the parsing logic but not the thing that actually matters under DDP:
when a real launcher (``torchrun`` / ``accelerate``, which wraps it) spawns TWO
genuine processes, each one must read its OWN distinct, correct topology from
``cf_dist`` -- distinct ranks, distinct local ranks, distinct QLoRA device
pinning -- and the two processes must actually coordinate.

This drives ``python -m torch.distributed.run --nproc_per_node 2`` over the
**gloo (CPU) backend with CUDA hidden**, so it consumes zero GPU memory and can
run alongside a live GPU training job. It closes the "proven by env-simulated
unit tests only" gap: two real OS processes, a real rendezvous, a real
collective.

The sharpest assertion here: with ``CUDA_VISIBLE_DEVICES=""`` the workers see
``torch.cuda.device_count() == 0``, yet ``cf_dist`` reports ``world_size == 2``
-- so this test passes *only* if the topology comes from ``WORLD_SIZE`` and not
from the device count (the exact bug cf_dist fixes).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = pathlib.Path(__file__).resolve().parent / "_ddp_worker.py"

# torch must be importable to launch the distributed runner at all.
pytest.importorskip("torch")


def _have_gloo() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_gloo_available()
    except Exception:
        return False


@pytest.mark.skipif(not _have_gloo(), reason="torch.distributed gloo backend unavailable")
def test_real_two_process_launch_reads_distinct_topology(tmp_path):
    """Spawn two genuine processes via the real launcher and assert each reads
    its own correct topology from cf_dist, with zero GPU contact."""
    env = dict(os.environ)
    env["CFDIST_OUT"] = str(tmp_path)
    # Hide CUDA: zero GPU contact (won't touch a concurrent GPU run) AND forces
    # device_count()==0 so world_size MUST come from WORLD_SIZE to be correct.
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Start from a clean topology so the launcher is the only thing that sets it.
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(WORKER),
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, (
        f"launcher exited {proc.returncode}\n"
        f"--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}"
    )

    # Both ranks must have written their snapshot.
    files = sorted(tmp_path.glob("rank_*.json"))
    assert len(files) == 2, f"expected 2 rank files, got {[f.name for f in files]}"
    results = {}
    for f in files:
        r = json.loads(f.read_text())
        results[r["rank"]] = r
    assert set(results) == {0, 1}, f"ranks seen: {sorted(results)}"

    r0, r1 = results[0], results[1]

    # Each process read world_size=2 from cf_dist -- NOT from device_count(),
    # which is 0 here because CUDA is hidden. This is the crux of the fix.
    assert r0["world_size"] == 2 and r1["world_size"] == 2

    # Distinct local ranks => distinct per-rank QLoRA device pinning.
    assert {r0["local_rank"], r1["local_rank"]} == {0, 1}
    assert r0["placement"] == {"": r0["local_rank"]}
    assert r1["placement"] == {"": r1["local_rank"]}

    # is_main true on exactly one process (rank 0) => rank0_print de-dupes.
    assert r0["is_main"] is True and r1["is_main"] is False
    assert r0["is_distributed"] is True and r1["is_distributed"] is True

    # cf_dist matched the launcher's published env exactly.
    assert r0["raw_env"]["WORLD_SIZE"] == "2"
    assert r0["raw_env"]["RANK"] == "0"
    assert r1["raw_env"]["RANK"] == "1"

    # The two processes genuinely coordinated (gloo all_gather saw both ranks)...
    assert r0["gathered_ranks"] == [0, 1]
    assert r1["gathered_ranks"] == [0, 1]
    # ...and are genuinely different OS processes.
    assert r0["pid"] != r1["pid"]
