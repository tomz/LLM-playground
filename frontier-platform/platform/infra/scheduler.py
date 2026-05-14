"""Gang scheduler interface. Wraps Slurm or Kueue/Volcano."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class JobSpec:
    name: str
    image: str
    command: list[str]
    nodes: int
    gpus_per_node: int
    priority: int = 100
    max_runtime_h: int = 168
    requeue_on_failure: bool = True
    env: dict[str, str] | None = None


class Scheduler:
    def submit(self, spec: JobSpec) -> str: raise NotImplementedError
    def status(self, job_id: str) -> str: raise NotImplementedError
    def cancel(self, job_id: str) -> None: raise NotImplementedError
