"""Rolling cost & power accounting."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CostBook:
    by_phase: dict[str, float] = field(default_factory=dict)
    by_resource: dict[str, float] = field(default_factory=dict)

    def charge(self, phase: str, resource: str, dollars: float) -> None:
        self.by_phase[phase] = self.by_phase.get(phase, 0.0) + dollars
        self.by_resource[resource] = self.by_resource.get(resource, 0.0) + dollars

    @property
    def total(self) -> float:
        return sum(self.by_phase.values())

    def report(self) -> str:
        lines = [f"  TOTAL: ${self.total:,.0f}"]
        lines.append("  by phase:")
        for k, v in sorted(self.by_phase.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:24s} ${v:>14,.0f}")
        lines.append("  by resource:")
        for k, v in sorted(self.by_resource.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:24s} ${v:>14,.0f}")
        return "\n".join(lines)
