"""Gang scheduler interface. Wraps Slurm or Kueue/Volcano."""
from __future__ import annotations
import subprocess
import uuid
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


class LocalScheduler(Scheduler):
    """Runs JobSpec.command in a subprocess. Useful for tests + dev."""

    def __init__(self) -> None:
        self._jobs: dict[str, subprocess.Popen] = {}

    def submit(self, spec: JobSpec) -> str:
        env = None
        if spec.env:
            import os
            env = {**os.environ, **spec.env}
        proc = subprocess.Popen(
            spec.command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        self._jobs[job_id] = proc
        return job_id

    def status(self, job_id: str) -> str:
        proc = self._jobs.get(job_id)
        if proc is None:
            return "UNKNOWN"
        rc = proc.poll()
        if rc is None:
            return "RUNNING"
        if rc == 0:
            return "COMPLETED"
        if rc < 0:
            return "CANCELLED"
        return "FAILED"

    def cancel(self, job_id: str) -> None:
        proc = self._jobs.get(job_id)
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class SlurmScheduler(Scheduler):
    """Skeleton wrapper around sbatch/squeue/scancel.

    Only `submit` and `cancel` are trivially shell-able; full status parsing
    (job arrays, requeue counts, etc.) is left to a real deployment.
    """

    def submit(self, spec: JobSpec) -> str:
        cmd = [
            "sbatch", "--parsable",
            f"--job-name={spec.name}",
            f"--nodes={spec.nodes}",
            f"--gpus-per-node={spec.gpus_per_node}",
            f"--time={spec.max_runtime_h}:00:00",
            f"--priority={spec.priority}",
            "--wrap", " ".join(spec.command),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        # sbatch --parsable returns "<jobid>[;<cluster>]"
        return out.split(";")[0]

    def status(self, job_id: str) -> str:
        # Full state parsing varies by site; leave for the real deployment.
        raise NotImplementedError("parse squeue / sacct output per site policy")

    def cancel(self, job_id: str) -> None:
        subprocess.run(["scancel", job_id], check=False)
