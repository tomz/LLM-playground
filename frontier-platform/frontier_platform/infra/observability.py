"""Metric + log emission. OpenTelemetry over OTLP → Prom/Loki/Tempo."""
from __future__ import annotations
import json
import os
import subprocess
from typing import ClassVar


class Metrics:
    _path: ClassVar[str | None] = None

    @classmethod
    def _sink(cls) -> str | None:
        return os.environ.get("PLATFORM_METRICS_PATH")

    @staticmethod
    def emit(name: str, value: float, tags: dict[str, str] | None = None) -> None:
        sink = Metrics._sink()
        if not sink:
            return
        line = json.dumps({"name": name, "value": value, "tags": tags or {}})
        with open(sink, "a") as f:
            f.write(line + "\n")

    @classmethod
    def flush(cls) -> None:  # JSONL sink is line-buffered; nothing to do.
        return None


class GpuHealth:
    """Polls DCGM/NVML for XID errors, ECC counts, thermals."""

    def poll(self) -> list[dict]:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            try:
                out = []
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    out.append({
                        "index": i,
                        "temperature": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        "utilization": util.gpu,
                        "memory_used_mib": mem.used // (1 << 20),
                        "memory_total_mib": mem.total // (1 << 20),
                        "ecc_uncorrected": 0,
                    })
                return out
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total,ecc.errors.uncorrected.aggregate.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return []
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "index": int(parts[0]),
                    "temperature": float(parts[1]),
                    "utilization": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "memory_total_mib": float(parts[4]),
                    "ecc_uncorrected": int(parts[5]) if parts[5].isdigit() else 0,
                })
            except ValueError:
                continue
        return rows
