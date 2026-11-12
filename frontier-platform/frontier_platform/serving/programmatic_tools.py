"""Bounded programmatic tool plans with provenance and output filtering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

OutputFilter = Callable[[str], str]


class ToolRunner(Protocol):
    """Structural boundary implemented by ``rl.agentic.ToolEnv`` and backends."""

    def run_tool(self, payload: dict) -> tuple[str, bool]: ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


@dataclass(frozen=True)
class ToolPlan:
    calls: list[ToolCall]


@dataclass(frozen=True)
class ToolEvent:
    index: int
    name: str
    ok: bool
    raw_chars: int
    retained_chars: int


@dataclass
class PlanResult:
    outputs: list[str] = field(default_factory=list)
    events: list[ToolEvent] = field(default_factory=list)
    stopped_reason: str = "complete"


def execute_plan(
    env: ToolRunner,
    plan: ToolPlan,
    *,
    max_calls: int = 8,
    max_retained_chars: int = 16_000,
    output_filter: OutputFilter | None = None,
) -> PlanResult:
    """Execute deterministic model-written control flow under hard budgets."""
    if max_calls <= 0 or max_retained_chars < 0:
        raise ValueError("invalid tool budget")
    result = PlanResult()
    for index, call in enumerate(plan.calls):
        if index >= max_calls:
            result.stopped_reason = "call_budget"
            break
        output, ok = env.run_tool({"name": call.name, "args": call.args})
        filtered = output_filter(output) if output_filter else output
        remaining = max_retained_chars - sum(len(item) for item in result.outputs)
        retained = filtered[:max(0, remaining)]
        result.outputs.append(retained)
        result.events.append(ToolEvent(index, call.name, ok, len(output), len(retained)))
        if not ok:
            result.stopped_reason = "tool_error"
            break
        if len(retained) < len(filtered):
            result.stopped_reason = "output_budget"
            break
    return result
