"""Token identity, completion, snapshot, and sampling-distribution regressions."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from frontier_platform.rl.async_rollout import AsyncRolloutConfig, AsyncRolloutEngine
from frontier_platform.rl.grpo import GRPOConfig, grpo_step
from frontier_platform.rl.rollout import GroupRollout, sample_group
from frontier_platform.serving.engine import finished_event


class TokenText:
    pad_id = 0
    eos_id = 15
    vocab_size = 16

    def decode(self, ids: list[int]) -> str:
        return ",".join(map(str, ids))


class TokenPolicy(torch.nn.Module):
    """A nonuniform policy with dropout to expose train/eval distribution drift."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(vocab_size=16, max_seq_len=8)
        self.bias = torch.nn.Parameter(torch.linspace(-2, 1, 16))
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.dropout(self.bias).expand(*ids.shape, -1), None

    def choose(self, token: int) -> None:
        with torch.no_grad():
            self.bias.fill_(-20)
            self.bias[token] = 20


def actor_for(policy: TokenPolicy | None = None, **kwargs) -> AsyncRolloutEngine:
    options = dict(group_size=1, max_new_tokens=3, seq_len=5, device="cpu")
    options.update(kwargs)
    return AsyncRolloutEngine(policy or TokenPolicy(), TokenText(), AsyncRolloutConfig(**options))


def generate(kind: str, model: TokenPolicy, prompts: list[list[int]], **kwargs) -> GroupRollout:
    options = dict(group_size=1, max_new_tokens=3, seq_len=5, temperature=1.0)
    options.update(kwargs)
    if kind == "async":
        return actor_for(model, **options).generate_group(prompts)
    return sample_group(model, prompts, tokenizer=TokenText(), seed=4, **options)


class ScriptedEngine:
    def __init__(self, events) -> None:
        self.events = events
        self.calls = []
        self.closed = 0

    async def generate(self, req):
        self.calls.append(req)
        try:
            for event in self.events(req):
                yield event
        finally:
            self.closed += 1


def test_f7_caps_before_dispatch_and_rewards_exact_retained_ids():
    actor = actor_for()

    def events(req):
        assert req.max_new_tokens == 1
        yield {"token_id": 5, "logprob": -0.5, "done": False}
        yield finished_event(req, 1, "length", logprob_kind="behavior")

    actor.engine = ScriptedEngine(events)
    roll = actor.generate_group([[1, 2, 3, 4]])
    assert roll.ids.tolist() == [[1, 2, 3, 4, 5]]
    assert roll.resp_mask.tolist() == [[0, 0, 0, 0, 1]]
    assert roll.response_text == ["5"]
    assert roll.provenance.response_ids == ((5,),)
    assert roll.provenance.generation_limits == (1,)


@pytest.mark.parametrize("kind", ["sync", "async"])
def test_ragged_prompts_keep_independent_generation_budgets(kind):
    model = TokenPolicy()
    model.choose(5)
    roll = generate(kind, model, [[1, 2, 3, 4], [1]], group_size=2, temperature=0)
    assert roll.resp_mask.sum(1).tolist() == [1, 1, 3, 3]
    assert roll.prompt_lens.tolist() == [4, 4, 1, 1]
    assert roll.group_index.tolist() == [0, 0, 1, 1]
    for i, text in enumerate(roll.response_text):
        ids = roll.ids[i][roll.resp_mask[i].bool()].tolist()
        assert TokenText().decode(ids) == text
        assert tuple(ids) == roll.provenance.response_ids[i]


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("prompt", [[], [1] * 6, [-1], [True], [16]])
def test_invalid_prompts_fail_without_cropping(kind, prompt):
    with pytest.raises(ValueError):
        generate(kind, TokenPolicy(), [[1], prompt])


@pytest.mark.parametrize("kind", ["sync", "async"])
def test_model_context_limit_also_caps_generation(kind):
    roll = generate(kind, TokenPolicy(), [[1] * 7], seq_len=20)
    assert roll.ids.shape[1] <= 8
    assert roll.provenance.generation_limits == (1,)
    with pytest.raises(ValueError):
        generate(kind, TokenPolicy(), [[1] * 9], seq_len=20)


@pytest.mark.parametrize("kind", ["sync", "async"])
def test_empty_batch_and_zero_output_are_explicit(kind):
    empty = generate(kind, TokenPolicy(), [])
    assert empty.ids.shape == (0, 0)
    assert empty.response_text == []
    roll = generate(kind, TokenPolicy(), [[1] * 5], max_new_tokens=0)
    assert roll.ids.tolist() == [[1] * 5]
    assert roll.resp_mask.sum() == 0
    assert roll.response_text == [""]
    assert roll.provenance.finish_reasons == ("length",)


@pytest.mark.parametrize("kind", ["sync", "async"])
def test_eos_is_retained_once_for_mask_and_reward(kind):
    model = TokenPolicy()
    model.choose(TokenText.eos_id)
    roll = generate(kind, model, [[1, 2]], temperature=0)
    assert roll.response_text == ["15"]
    assert roll.resp_mask.sum() == 1
    assert roll.provenance.response_ids == ((15,),)
    assert roll.provenance.finish_reasons == ("eos",)
    assert roll.provenance.stop_token_ids == (15,)
    assert roll.behavior_logp[roll.resp_mask.bool()].tolist() == [0.0]


@pytest.mark.parametrize("lp", [None, float("nan"), float("inf"), -float("inf"), 0.1, True])
def test_async_rejects_missing_or_invalid_behavior_logprobs(lp):
    actor = actor_for(max_new_tokens=1)
    actor.engine = ScriptedEngine(lambda req: [
        {"token_id": 5, "logprob": lp, "done": False},
        finished_event(req, 1, "length", logprob_kind="behavior"),
    ])
    with pytest.raises(ValueError, match="log.prob"):
        actor.generate_group([[1]])
    assert actor.engine.closed == 1


@pytest.mark.parametrize("fault", [
    "missing_logprob", "no_finish", "duplicate_finish", "after_finish", "overrun",
    "bad_usage", "bool_usage", "missing_reason", "early_length", "raw_model",
    "missing_kind", "bad_token", "after_eos", "false_eos", "missing_stop_id",
    "cancelled_finish", "missing_done",
])
def test_async_rejects_invalid_or_incomplete_streams(fault):
    actor = actor_for(max_new_tokens=2)

    def events(req):
        token = {"token_id": 5, "logprob": -0.5, "done": False}
        terminal = finished_event(req, 2, "length", logprob_kind="behavior")
        stream = [dict(token), dict(token), terminal]
        if fault == "missing_logprob":
            stream[0].pop("logprob")
        elif fault == "no_finish":
            stream.pop()
        elif fault == "duplicate_finish":
            stream.append(terminal)
        elif fault == "after_finish":
            stream.append(token)
        elif fault == "overrun":
            stream.insert(0, token)
        elif fault == "bad_usage":
            terminal["usage"]["prompt_tokens"] = 99
        elif fault == "bool_usage":
            terminal["usage"]["prompt_tokens"] = True
        elif fault == "missing_reason":
            terminal.pop("finish_reason")
        elif fault == "early_length":
            stream.pop(0)
            terminal["usage"]["completion_tokens"] = 1
        elif fault == "raw_model":
            terminal["logprob_kind"] = "raw_model"
        elif fault == "missing_kind":
            terminal.pop("logprob_kind")
        elif fault == "bad_token":
            stream[0]["token_id"] = 16
        elif fault == "after_eos":
            stream[0]["token_id"] = 15
        elif fault == "false_eos":
            terminal.update(finish_reason="eos", stop_token_id=15)
        elif fault == "missing_stop_id":
            stream[1]["token_id"] = 15
            terminal["finish_reason"] = "eos"
        elif fault == "cancelled_finish":
            terminal["finish_reason"] = "cancelled"
        elif fault == "missing_done":
            stream[0].pop("done")
        return stream

    actor.engine = ScriptedEngine(events)
    with pytest.raises(ValueError):
        actor.generate_group([[1]])
    assert actor.engine.closed == 1
    actor.sync_weights()  # failed groups release the sync guard


def test_actor_snapshots_and_explicit_sync_preserve_provenance():
    learner = TokenPolicy()
    learner.choose(5)
    actor = actor_for(learner, max_new_tokens=1, temperature=0)
    assert learner.training
    assert actor.model is not learner
    assert actor.model.bias.data_ptr() != learner.bias.data_ptr()
    assert not actor.model.bias.requires_grad
    before = actor.generate_group([[1]])
    learner.choose(7)
    assert actor.generate_group([[1]]).response_text == ["5"]
    actor.sync_weights()
    after = actor.generate_group([[1]])
    assert after.response_text == ["7"]
    assert before.provenance.weight_version == 0
    assert after.provenance.weight_version == 1
    assert learner.training


@pytest.mark.parametrize("failure", ["cancel", "bad_logprob"])
def test_inflight_sync_is_rejected_and_all_siblings_are_drained(failure):
    async def run():
        actor = actor_for(group_size=2, max_new_tokens=1)
        started = asyncio.Event()
        release = asyncio.Event()
        active, closed = 0, 0

        class BlockingEngine:
            async def generate(self, req):
                nonlocal active, closed
                active += 1
                number = active
                if active == 2:
                    started.set()
                try:
                    await release.wait()
                    if number == 1:
                        yield {"token_id": 5, "done": False}  # invalid probability
                    else:
                        await asyncio.Event().wait()
                finally:
                    closed += 1

        actor.engine = BlockingEngine()
        task = asyncio.create_task(actor.generate_group_async([[1]]))
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(RuntimeError, match="flight"):
            actor.sync_weights()
        if failure == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            with pytest.raises(ValueError):
                await asyncio.wait_for(task, 2)
        assert closed == 2
        assert actor.weight_version == 0
        actor.sync_weights()
        assert actor.weight_version == 1

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("temperature", [0.4, 1.0, 1.8])
def test_grpo_matches_behavior_temperature_and_disables_dropout(kind, temperature):
    policy = TokenPolicy()
    ref = TokenPolicy()
    roll = generate(kind, policy, [[1], [2]], temperature=temperature, group_size=2)
    cfg = GRPOConfig(temperature=temperature, entropy_coef=0.1)
    metrics = grpo_step(policy, ref, roll, torch.tensor([0., 1., 1., 0.]), cfg)
    expected_logp = (policy.bias.detach() / temperature).log_softmax(-1)
    expected_entropy = -(expected_logp.exp() * expected_logp).sum().item()
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["entropy"] == pytest.approx(expected_entropy, abs=1e-6)
    assert policy.training and ref.training


@pytest.mark.parametrize("temperature,top_p,cfg_temperature", [(0., 1., 1.), (1., .7, 1.), (.5, 1., 1.)])
def test_grpo_rejects_unsupported_or_mismatched_sampling(temperature, top_p, cfg_temperature):
    policy = TokenPolicy()
    roll = generate("async", policy, [[1]], temperature=temperature, top_p=top_p)
    with pytest.raises(ValueError):
        grpo_step(policy, TokenPolicy(), roll, torch.ones(1), GRPOConfig(temperature=cfg_temperature))


@pytest.mark.parametrize("fault", ["missing", "nan", "inf", "positive", "shape", "raw"])
def test_grpo_rejects_unusable_behavior_probabilities(fault):
    policy = TokenPolicy()
    roll = generate("sync", policy, [[1]])
    if fault == "missing":
        roll.behavior_logp = None
    elif fault == "shape":
        roll.behavior_logp = roll.behavior_logp[:, :-1]
    elif fault == "raw":
        roll.provenance = replace(roll.provenance, logprob_kind="raw_model")
    else:
        value = {"nan": float("nan"), "inf": float("inf"), "positive": .1}[fault]
        roll.behavior_logp[roll.resp_mask.bool()] = value
    with pytest.raises(ValueError):
        grpo_step(policy, TokenPolicy(), roll, torch.ones(1), GRPOConfig())


def test_legacy_record_constructor_still_works_but_missing_probabilities_do_not():
    policy = TokenPolicy()
    roll = generate("sync", policy, [[1]])
    legacy = GroupRollout(roll.ids, roll.resp_mask, roll.group_index, roll.prompt_lens, roll.response_text)
    with pytest.raises(ValueError, match="behavior"):
        grpo_step(policy, TokenPolicy(), legacy, torch.ones(1), GRPOConfig())
    legacy.behavior_logp = roll.behavior_logp
    assert grpo_step(policy, TokenPolicy(), legacy, torch.ones(1), GRPOConfig())["ratio_mean"] == pytest.approx(1.)


def test_skew_uses_observed_probabilities_instead_of_fabricating_ratio_one():
    policy = TokenPolicy()
    roll = generate("sync", policy, [[1]], group_size=4, temperature=.7)
    with torch.no_grad():
        policy.bias[::2] += 1.5
    selected = roll.resp_mask.bool()
    current = (policy.bias.detach() / .7).log_softmax(-1)[roll.ids[selected]]
    expected = (current - roll.behavior_logp[selected]).exp().mean().item()
    metrics = grpo_step(policy, TokenPolicy(), roll, torch.arange(4.), GRPOConfig(temperature=.7))
    assert metrics["ratio_mean"] == pytest.approx(expected, abs=1e-6)
    assert abs(expected - 1.) > .01
