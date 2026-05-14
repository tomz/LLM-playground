"""JSONL + optional W&B logger. Only rank 0 emits."""
from __future__ import annotations
import json, os, time
from .dist import is_master


class Logger:
    def __init__(self, jsonl_path: str | None, wandb_project: str | None, config: dict):
        self.master = is_master()
        self.t0 = time.time()
        self.fh = None
        self.wb = None
        if not self.master:
            return
        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            self.fh = open(jsonl_path, "a")
        if wandb_project:
            try:
                import wandb
                self.wb = wandb.init(project=wandb_project, config=config)
            except ImportError:
                print("[logger] wandb not installed; skipping")

    def log(self, step: int, **kw) -> None:
        if not self.master:
            return
        kw["step"] = step
        kw["wall_s"] = round(time.time() - self.t0, 3)
        if self.fh:
            self.fh.write(json.dumps(kw) + "\n")
            self.fh.flush()
        if self.wb:
            self.wb.log(kw, step=step)

    def close(self) -> None:
        if self.fh:
            self.fh.close()
        if self.wb:
            self.wb.finish()
