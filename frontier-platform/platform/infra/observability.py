"""Metric + log emission. OpenTelemetry over OTLP → Prom/Loki/Tempo."""
from __future__ import annotations


class Metrics:
    @staticmethod
    def emit(name: str, value: float, tags: dict[str, str] | None = None) -> None:
        # real: otel meter or statsd client
        pass


class GpuHealth:
    """Polls DCGM for XID errors, ECC counts, thermals; reports unhealthy GPUs."""
    def poll(self) -> list[dict]:
        raise NotImplementedError
