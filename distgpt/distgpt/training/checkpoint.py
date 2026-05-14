"""Distributed checkpointing via DCP — sharded, async, reshardable."""
from __future__ import annotations
import os
import torch.distributed.checkpoint as dcp


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
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        }
        dcp.save(state, checkpoint_id=path)
        # loader state + meta on rank 0 only (small json)
        from ..utils.dist import is_master
        if is_master():
            import json
            with open(os.path.join(path, "meta.json"), "w") as f:
                json.dump({"step": step, "loader": loader.state_dict(), **(extra or {})}, f)
            self._gc()
        return path

    def load(self, model, optimizer, loader, step: int | str = "latest") -> int:
        if step == "latest":
            step = self.latest()
            if step is None:
                return 0
        path = self._path(int(step))
        state = {"model": model.state_dict(), "optim": optimizer.state_dict()}
        dcp.load(state, checkpoint_id=path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        import json
        with open(os.path.join(path, "meta.json")) as f:
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
