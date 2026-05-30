"""Agentic / tool-use environment tests (platform/rl/agentic.py, gap #4)."""
from __future__ import annotations

from platform.rl.agentic import (
    FINAL,
    TOOL_CALL,
    ToolEnv,
    make_calculator,
    make_keyvalue_store,
    parse_action,
    rollout_episode,
)


def test_parse_action_tool_final_noop():
    kind, payload = parse_action(TOOL_CALL + '{"name": "calculator", "args": {"expr": "2+2"}}')
    assert kind == "tool" and payload["name"] == "calculator"
    kind, payload = parse_action(FINAL + " the answer is 42")
    assert kind == "final" and "42" in payload
    kind, payload = parse_action("just thinking out loud")
    assert kind == "noop"
    # malformed JSON -> noop, not a crash
    kind, _ = parse_action(TOOL_CALL + "{not json}")
    assert kind == "noop"


def test_calculator_tool_runs_and_rejects_unsafe():
    calc = make_calculator()
    assert calc.fn({"expr": "2+2*3"}) == "8"
    import pytest
    with pytest.raises(ValueError):
        calc.fn({"expr": "__import__('os')"})


def test_tool_env_unknown_tool_is_graceful():
    env = ToolEnv([make_calculator()])
    obs, ok = env.run_tool({"name": "nope", "args": {}})
    assert not ok and "unknown tool" in obs


def test_rollout_episode_multi_turn_success():
    """A scripted agent: call the calculator, then answer with the result.
    Verifier checks the final answer contains the right number."""
    env = ToolEnv([make_calculator()], max_steps=4)

    script = iter([
        TOOL_CALL + '{"name": "calculator", "args": {"expr": "21*2"}}',
        FINAL + " 42",
    ])

    def policy(_transcript: str) -> str:
        return next(script)

    def verifier(_task: str, answer: str) -> float:
        return 1.0 if "42" in answer else 0.0

    traj = rollout_episode(env, "what is 21*2?", policy, verifier)
    assert traj.success and traj.reward == 1.0
    assert traj.n_tool_calls == 1
    assert traj.final_answer.strip() == "42"
    # tool observation was fed back into the transcript
    assert any(t.kind == "tool" and t.observation == "42" for t in traj.transitions)


def test_rollout_episode_sparse_reward_on_failure():
    env = ToolEnv([make_keyvalue_store({"capital_of_france": "Paris"})], max_steps=3)

    script = iter([
        TOOL_CALL + '{"name": "lookup", "args": {"key": "capital_of_france"}}',
        FINAL + " London",
    ])
    policy = lambda _t: next(script)
    verifier = lambda _task, ans: 1.0 if "Paris" in ans else 0.0

    traj = rollout_episode(env, "capital of France?", policy, verifier)
    # Wrong final answer -> terminal reward 0 even though the tool succeeded.
    assert not traj.success and traj.reward == 0.0
    assert traj.n_tool_calls == 1


def test_rollout_episode_step_budget_and_malformed():
    env = ToolEnv([make_calculator()], max_steps=3)
    # Agent never finalizes and emits malformed actions.
    policy = lambda _t: "I am confused"
    verifier = lambda _task, ans: 1.0
    traj = rollout_episode(env, "do something", policy, verifier)
    assert traj.final_answer is None  # never finalized
    assert traj.steps == 3            # hit the step budget
    assert traj.malformed == 3
    assert traj.reward == 0.0
