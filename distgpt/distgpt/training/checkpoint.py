"""Distributed checkpointing via DCP — sharded, async, reshardable.

Each DP rank's loader cursor is recorded in `meta_rank{rank}.json`; the rank-0
process additionally writes `meta.json` with the latest step and any extras.

Optimizer storage
-----------------
``save`` / ``load`` accept either a single optimizer or a list of optimizers.
The on-disk layout is:

  * single optimizer → key ``"optim"`` (legacy; what existing checkpoints have)
  * list of optimizers → keys ``"optim_0"``, ``"optim_1"``, ... (new in
    distgpt-with-Muon, because Muon + AdamW = two optimizers).

``load`` auto-detects: if the saved state has ``"optim"`` it routes that to
the first optimizer in the list (or the lone optimizer in the legacy case);
otherwise it looks for ``optim_N``.
"""
from __future__ import annotations
import os
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def _as_list(opts):
    """Normalize a single-optimizer or list-of-optimizers argument."""
    if isinstance(opts, (list, tuple)):
        return list(opts)
    return [opts]


class CheckpointManager:
    def __init__(self, root: str, run_id: str, keep_last: int = 5):
        self.root = os.path.join(root, run_id, "ckpts")
        self.keep_last = keep_last
        os.makedirs(self.root, exist_ok=True)

    def _path(self, step: int) -> str:
        return os.path.join(self.root, f"step_{step:09d}")

    def save(self, model, optimizer, loader, step: int, extra: dict | None = None) -> str:
        path = self._path(step)
        os.makedirs(path, exist_ok=True)
        opts = _as_list(optimizer)
        state: dict = {"model": model.state_dict()}
        if len(opts) == 1:
            # Preserve the legacy on-disk key so older checkpoints stay
            # readable. Only multi-optimizer (Muon+AdamW) saves use the
            # numbered keys.
            state["optim"] = opts[0].state_dict()
        else:
            for i, o in enumerate(opts):
                state[f"optim_{i}"] = o.state_dict()
        dcp.save(state, checkpoint_id=path)
        # Each rank writes its own loader-state file (cursor differs per DP rank).
        from ..utils.dist import is_master
        rank = dist.get_rank() if dist.is_initialized() else 0
        import json
        with open(os.path.join(path, f"meta_rank{rank}.json"), "w") as f:
            json.dump({"step": step, "loader": loader.state_dict(),
                       "n_optim": len(opts)}, f)
        if is_master():
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump({"step": step, "n_optim": len(opts),
                           **(extra or {})}, f)
        # Synchronize so GC can run safely on rank 0 only.
        if dist.is_initialized():
            dist.barrier()
        if is_master():
            self._gc()
        if dist.is_initialized():
            dist.barrier()
        return path

    def load(self, model, optimizer, loader, step: int | str = "latest") -> int:
        if step == "latest":
            step = self.latest()
            if step is None:
                return 0
        path = self._path(int(step))
        opts = _as_list(optimizer)
        state: dict = {"model": model.state_dict()}
        if len(opts) == 1:
            state["optim"] = opts[0].state_dict()
        else:
            for i, o in enumerate(opts):
                state[f"optim_{i}"] = o.state_dict()
        dcp.load(state, checkpoint_id=path)
        model.load_state_dict(state["model"])
        # Back-compat: load "optim" into opts[0] if that's what was saved;
        # otherwise route the numbered keys.
        if "optim" in state and len(opts) >= 1:
            opts[0].load_state_dict(state["optim"])
            for i in range(1, len(opts)):
                # Numbered keys exist only on multi-optim saves; if the
                # checkpoint was single-optim and the user reloads into a
                # multi-optim config, the extra optimizers start fresh.
                key = f"optim_{i}"
                if key in state:
                    opts[i].load_state_dict(state[key])
        else:
            for i, o in enumerate(opts):
                key = f"optim_{i}"
                if key in state:
                    o.load_state_dict(state[key])
        rank = dist.get_rank() if dist.is_initialized() else 0
        import json
        rank_meta = os.path.join(path, f"meta_rank{rank}.json")
        # Fall back to legacy single meta.json (older checkpoints).
        meta_path = rank_meta if os.path.exists(rank_meta) else os.path.join(path, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        loader.load_state_dict(meta["loader"])
        return int(meta["step"]) + 1

    def latest(self) -> int | None:
        if not os.path.isdir(self.root):
            return None
        steps = []
        for d in os.listdir(self.root):
            if d.startswith("step_"):
                try:
                    steps.append(int(d.split("_")[1]))
                except ValueError:
                    pass
        return max(steps) if steps else None

    def _gc(self) -> None:
        if not os.path.isdir(self.root):
            return
        steps = sorted(
            int(d.split("_")[1]) for d in os.listdir(self.root) if d.startswith("step_")
        )
        for s in steps[: -self.keep_last]:
            import shutil
            shutil.rmtree(self._path(s), ignore_errors=True)
