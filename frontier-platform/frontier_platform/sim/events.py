"""In-process event bus + JSONL logger. Every subsystem emits structured events."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EventBus:
    handlers: list[Callable[[dict], None]] = field(default_factory=list)
    out_path: str | None = None
    _fh: object = None

    def __post_init__(self):
        if self.out_path:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            self._fh = open(self.out_path, "w")

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self.handlers.append(fn)

    def emit(self, kind: str, **kw) -> None:
        ev = {"kind": kind, **kw}
        if self._fh:
            self._fh.write(json.dumps(ev) + "\n")
            self._fh.flush()
        for h in self.handlers:
            h(ev)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
