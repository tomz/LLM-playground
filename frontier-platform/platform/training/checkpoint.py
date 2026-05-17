"""Checkpointing. Single-process path uses plain `torch.save`; the multi-rank
path documents how to switch to `torch.distributed.checkpoint` (DCP), which
handles sharding+resharding transparently."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckpointMeta:
    run_id: str
    step: int
    model_config_sha: str = ""
    parallel_layout: dict = field(default_factory=dict)
    shard_uris: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointManager:
    def __init__(
        self,
        root_uri: str,
        run_id: str,
        keep_last: int = 10,
        milestone_every: int = 10_000,
    ):
        self.root = Path(root_uri) / run_id / "ckpts"
        self.run_id = run_id
        self.keep_last = int(keep_last)
        self.milestone_every = int(milestone_every)
        self.root.mkdir(parents=True, exist_ok=True)

    def _step_dir(self, step: int) -> Path:
        return self.root / f"step_{step:09d}"

    def save_async(self, engine, dataloader, step: int) -> CheckpointMeta:
        """Synchronous in the single-process path. In a multi-rank setup you'd
        use `torch.distributed.checkpoint.save({...}, checkpoint_id=path)`
        instead; DCP handles sharding+async semantics natively."""
        import torch

        path = self._step_dir(step)
        path.mkdir(parents=True, exist_ok=True)
        # Engine's `.model` may be DDP-wrapped; unwrap.
        model = getattr(engine.model, "module", engine.model)
        state = {
            "model": model.state_dict(),
            "optim": engine.optimizer.state_dict(),
            "loader": dataloader.state_dict() if dataloader is not None and hasattr(dataloader, "state_dict") else None,
            "step": int(step),
        }
        state_path = path / "state.pt"
        torch.save(state, state_path)
        meta = CheckpointMeta(
            run_id=self.run_id,
            step=int(step),
            shard_uris=[str(state_path)],
            sha256={state_path.name: _sha256(state_path)},
        )
        with open(path / "meta.json", "w") as f:
            json.dump(dataclasses.asdict(meta), f)
        self._gc()
        return meta

    def load(self, step: int | str = "latest", target_layout: dict | None = None) -> CheckpointMeta:
        if step == "latest":
            s = self.latest()
            if s is None:
                raise FileNotFoundError(f"no checkpoints in {self.root}")
            step = s
        path = self._step_dir(int(step))
        with open(path / "meta.json") as f:
            meta_dict = json.load(f)
        # NOTE: resharding to `target_layout` is delegated to DCP in real
        # multi-rank training; here world_size==1 so no resharding is needed.
        return CheckpointMeta(**meta_dict)

    def load_into(self, engine, dataloader, step: int | str = "latest") -> CheckpointMeta:
        """Companion to `save_async` for the single-process path."""
        import torch

        meta = self.load(step)
        path = self._step_dir(meta.step) / "state.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = getattr(engine.model, "module", engine.model)
        model.load_state_dict(state["model"])
        engine.optimizer.load_state_dict(state["optim"])
        if dataloader is not None and state.get("loader") is not None and hasattr(dataloader, "load_state_dict"):
            dataloader.load_state_dict(state["loader"])
        return meta

    def latest(self) -> int | None:
        if not self.root.is_dir():
            return None
        steps = []
        for d in os.listdir(self.root):
            if d.startswith("step_"):
                try:
                    steps.append(int(d.split("_", 1)[1]))
                except ValueError:
                    pass
        return max(steps) if steps else None

    def _gc(self) -> None:
        if not self.root.is_dir():
            return
        steps = sorted(
            int(d.split("_", 1)[1])
            for d in os.listdir(self.root)
            if d.startswith("step_")
        )
        for s in steps[: -self.keep_last]:
            shutil.rmtree(self._step_dir(s), ignore_errors=True)
