"""Agentic / tool-use training environment for long-horizon RL (gap #4,
docs/14-gap-analysis-vs-frontier.md §5).

The 2025-2026 flagships are sold as *agents*: SWE-bench, computer/browser use,
multi-tool orchestration, Deep-Research workflows. Training for this needs a
different regime from single-turn SFT/DPO — **multi-turn tool-use trajectories**
and RL over **long-horizon, sparse-reward** tasks where the reward lands only at
the end (did the task complete?).

This module provides a toy-functional, dependency-free harness that captures the
*structure* of agentic RL so the rest of the platform (RLVR loop, sim, eval) can
reason about it:

  - `Tool`: a named callable the agent can invoke (deterministic, sandboxable).
  - `ToolEnv`: a multi-turn environment — the agent emits ``<|tool_call|>`` JSON,
    the env runs the tool and returns a ``<|tool_result|>``, until the agent
    emits a final answer or a step budget is hit.
  - `rollout_episode`: runs one agent↔env episode with a *policy function*
    (any callable: prompt history -> next action string), returning a
    `Trajectory` with per-step transitions and a terminal verifiable reward.

The policy is abstracted so tests can drive it with a scripted agent; in
production the policy is the LLM's generate() and the trajectory feeds GRPO with
the *final* task-completion reward broadcast over the agent's tokens.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

# Re-use the tokenizer's reserved agentic special tokens (see tokenizer/bytes.py).
TOOL_CALL = "<|tool_call|>"
TOOL_RESULT = "<|tool_result|>"
FINAL = "<|final|>"

Tool = Callable[[dict], str]
# A policy maps the running transcript -> the agent's next utterance.
Policy = Callable[[str], str]


@dataclass
class ToolSpec:
    name: str
    fn: Tool
    description: str = ""


_FINAL_RE = re.compile(re.escape(FINAL) + r"\s*(.*)", re.DOTALL)


def _extract_balanced_json(s: str, start: int) -> str | None:
    """Extract a balanced ``{...}`` substring starting at the first ``{`` at or
    after ``start``. Handles nested braces (the regex non-greedy approach can't).
    """
    i = s.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i : j + 1]
    return None


def parse_action(utterance: str) -> tuple[str, dict | str]:
    """Parse an agent utterance into ('tool', {name,args}) | ('final', text) |
    ('noop', raw). Tool calls are ``<|tool_call|>{"name":..,"args":{..}}``."""
    m = _FINAL_RE.search(utterance)
    if m:
        return "final", m.group(1).strip()
    idx = utterance.find(TOOL_CALL)
    if idx >= 0:
        blob = _extract_balanced_json(utterance, idx + len(TOOL_CALL))
        if blob is not None:
            try:
                return "tool", json.loads(blob)
            except json.JSONDecodeError:
                return "noop", utterance
    return "noop", utterance


@dataclass
class Transition:
    step: int
    agent: str          # the agent's utterance
    kind: str           # 'tool' | 'final' | 'noop'
    observation: str    # the env's response (tool result / empty)


@dataclass
class Trajectory:
    task: str
    transitions: list[Transition] = field(default_factory=list)
    final_answer: str | None = None
    reward: float = 0.0
    success: bool = False
    steps: int = 0
    # 'malformed' tool calls (reward-hacking / format failures) for monitoring.
    malformed: int = 0

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for t in self.transitions if t.kind == "tool")


class ToolEnv:
    """A multi-turn tool-use environment.

    Holds a registry of tools and a per-task verifier (final-answer check). The
    agent and env alternate until ``<|final|>`` or ``max_steps``.
    """

    def __init__(self, tools: list[ToolSpec], *, max_steps: int = 8):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def tool_manifest(self) -> str:
        """Human/agent-readable list of available tools (goes in the prompt)."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def run_tool(self, payload: dict) -> tuple[str, bool]:
        """Execute a parsed tool call. Returns (observation, ok)."""
        name = payload.get("name")
        args = payload.get("args", {}) or {}
        spec = self.tools.get(name)
        if spec is None:
            return f"error: unknown tool {name!r}", False
        try:
            return str(spec.fn(args)), True
        except Exception as e:  # tools must be sandboxed in production
            return f"error: {type(e).__name__}: {e}", False


def rollout_episode(
    env: ToolEnv,
    task: str,
    policy: Policy,
    verifier: Callable[[str, str], float],
    *,
    system_prompt: str | None = None,
) -> Trajectory:
    """Run one agent↔env episode and score the terminal answer.

    ``policy(transcript) -> utterance``. ``verifier(task, final_answer) ->
    reward``. The reward is *terminal and sparse* (the hallmark of agentic RL):
    intermediate tool steps get no reward; only task completion does.
    """
    sys = system_prompt or (
        "You are a tool-using agent. Call tools with "
        f'{TOOL_CALL}{{"name": <tool>, "args": {{...}}}} and finish with '
        f"{FINAL}<answer>.\nTools:\n" + env.tool_manifest()
    )
    transcript = f"{sys}\n\nTask: {task}\n"
    traj = Trajectory(task=task)

    for step in range(env.max_steps):
        utterance = policy(transcript)
        kind, parsed = parse_action(utterance)
        if kind == "final":
            traj.final_answer = parsed if isinstance(parsed, str) else ""
            traj.transitions.append(Transition(step, utterance, "final", ""))
            transcript += f"{utterance}\n"
            break
        elif kind == "tool":
            obs, ok = env.run_tool(parsed)  # type: ignore[arg-type]
            if not ok:
                traj.malformed += 1
            traj.transitions.append(Transition(step, utterance, "tool", obs))
            transcript += f"{utterance}\n{TOOL_RESULT}{obs}\n"
        else:
            traj.malformed += 1
            traj.transitions.append(Transition(step, utterance, "noop", ""))
            transcript += f"{utterance}\n"

    traj.steps = len(traj.transitions)
    if traj.final_answer is not None:
        traj.reward = verifier(task, traj.final_answer)
        traj.success = traj.reward > 0
    return traj


# --- a couple of built-in deterministic tools (safe, sandbox-free) ---

def make_calculator() -> ToolSpec:
    """A tiny arithmetic tool. Evaluates ``{"expr": "2+2*3"}`` with a safe parser."""
    def _calc(args: dict) -> str:
        expr = str(args.get("expr", ""))
        if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
            raise ValueError("unsafe expression")
        # eval restricted to arithmetic chars only (validated above).
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    return ToolSpec("calculator", _calc, "evaluate an arithmetic expression {expr}")


def make_keyvalue_store(data: dict[str, str]) -> ToolSpec:
    """A read-only lookup tool over a fixed dict — stands in for search/retrieval."""
    def _lookup(args: dict) -> str:
        key = str(args.get("key", ""))
        return data.get(key, "<not found>")
    return ToolSpec("lookup", _lookup, "look up a key {key} in the knowledge store")
