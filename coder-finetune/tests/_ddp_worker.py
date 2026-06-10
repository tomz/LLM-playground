"""Worker spawned under ``torch.distributed.run`` to exercise the REAL
two-process launcher -> ``cf_dist`` contract on the gloo (CPU) backend.

This is **not** a pytest module (leading underscore => not collected). It is
launched as a subprocess by ``tests/test_dist_launch.py`` via::

    python -m torch.distributed.run --standalone --nproc_per_node 2 _ddp_worker.py

It deliberately uses ``backend="gloo"`` with CUDA hidden
(``CUDA_VISIBLE_DEVICES=""``) so it consumes **zero GPU memory** and never
collides with a concurrent GPU training run. Each rank:

  1. snapshots ``cf_dist.dist_env()`` *before* forming any process group
     (cf_dist only reads env vars the launcher published);
  2. forms a genuine gloo process group and ``all_gather``s its rank, proving
     two real coordinating processes exist (not one process with mutated env);
  3. writes its snapshot to ``$CFDIST_OUT/rank_<rank>.json`` for the parent
     test to assert on.

With CUDA hidden, ``torch.cuda.device_count()`` is 0 in every worker, yet
``dist_env().world_size`` is 2 -- the cleanest possible demonstration that the
fix reads ``WORLD_SIZE`` and not the device count.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch.distributed as dist

from cf_dist import dist_env, placement_device_map


def main() -> int:
    out_dir = pathlib.Path(os.environ["CFDIST_OUT"])

    # 1. cf_dist reads what the launcher published -- before any process group.
    env = dist_env()

    # 2. Form a genuine process group to prove two real coordinating processes
    #    exist. gloo => CPU-only, zero GPU contact.
    dist.init_process_group(backend="gloo")
    try:
        pg_world = dist.get_world_size()  # capture while the group is alive
        gathered: list[object] = [None] * pg_world
        dist.all_gather_object(gathered, env.rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()

    # 3. Self-checks (defensive: a non-zero exit is also caught by the parent).
    #    cf_dist's env-derived world_size must equal the live process group's.
    assert env.world_size == pg_world
    gathered_ranks = sorted(int(g) for g in gathered)

    result = {
        "rank": env.rank,
        "local_rank": env.local_rank,
        "world_size": env.world_size,
        "is_main": env.is_main,
        "is_distributed": env.is_distributed,
        "placement": placement_device_map(env),
        "gathered_ranks": gathered_ranks,
        # Cross-check cf_dist against the raw env the launcher actually set.
        "raw_env": {
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        },
        "pid": os.getpid(),
    }
    (out_dir / f"rank_{env.rank}.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
