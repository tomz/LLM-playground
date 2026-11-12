"""Virtual clock. Resolution = seconds. All subsystems advance the same clock."""
from __future__ import annotations


class Clock:
    def __init__(self, t0: float = 0.0):
        self.t = t0   # seconds since program start

    def advance(self, dt_seconds: float) -> None:
        assert dt_seconds >= 0
        self.t += dt_seconds

    @property
    def days(self) -> float:
        return self.t / 86400.0

    @property
    def hours(self) -> float:
        return self.t / 3600.0

    def fmt(self) -> str:
        d = int(self.days)
        h = int((self.t - d * 86400) / 3600)
        m = int((self.t - d * 86400 - h * 3600) / 60)
        return f"day {d:3d} {h:02d}:{m:02d}"
