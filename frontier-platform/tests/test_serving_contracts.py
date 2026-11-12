"""Numerical fixtures for the repaired sampler and CPU cache/stop contract."""
from __future__ import annotations

import asyncio
import copy

import pytest
import torch

from frontier_platform.model.config import ModelConfig
from frontier_platform.model.transformer import Transformer
from frontier_platform.serving.engine import Engine, EngineConfig, GenRequest
from frontier_platform.serving.sampling import sampling_log_probs


def drain(stream):
    async def run():
        return [event async for event in stream]
    return asyncio.run(run())


def tiny():
    torch.manual_seed(7)
    return Transformer(ModelConfig(vocab_size=16, n_layer=2, n_head=2, n_kv_head=1,
                                   d_model=16, d_ffn=32, max_seq_len=16))


def test_f7_nucleus_keeps_the_crossing_token():
    logits = torch.tensor([.60, .25, .15]).log()
    probs = sampling_log_probs(logits, top_p=.7).exp()
    torch.testing.assert_close(probs, torch.tensor([.60 / .85, .25 / .85, 0.]))
    exact = sampling_log_probs(logits, top_p=.6).exp()
    torch.testing.assert_close(exact, torch.tensor([1., 0., 0.]))


@pytest.mark.parametrize("temperature,top_p", [(0., 1.), (.6, 1.), (1.3, .7)])
def test_cached_and_full_forward_tokens_and_behavior_logps_match(temperature, top_p):
    class FullForward(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model, self.cfg = model, model.cfg

        def forward(self, ids):
            return self.model(ids)

    model = tiny()
    req = GenRequest(prompt_ids=[1, 2, 3], max_new_tokens=5, temperature=temperature, top_p=top_p)
    results = []
    for backend_model in (model, FullForward(copy.deepcopy(model))):
        engine = Engine(EngineConfig(device="cpu"), model=backend_model)
        torch.manual_seed(9)
        results.append(drain(engine.generate(req)))
    cached, full = results
    assert [c["token_id"] for c in cached[:-1]] == [c["token_id"] for c in full[:-1]]
    assert [c["logprob"] for c in cached[:-1]] == pytest.approx([c["logprob"] for c in full[:-1]], abs=1e-6)
    assert cached[-1] == full[-1]
    assert cached[-1]["finish_reason"] == "length"
    assert cached[-1]["logprob_kind"] == "behavior"


@pytest.mark.parametrize("finish", ["eos", "stop"])
def test_stop_token_is_emitted_once_and_finish_reason_is_exact(finish):
    engine = Engine(EngineConfig(device="cpu"), model=tiny())
    first = drain(engine.generate(GenRequest([1, 2], max_new_tokens=1, temperature=0)))[0]["token_id"]
    kwargs = {"eos_id": first} if finish == "eos" else {"stop": [first]}
    events = drain(engine.generate(GenRequest([1, 2], max_new_tokens=5, temperature=0, **kwargs)))
    assert len(events) == 2
    assert events[0]["token_id"] == first
    assert events[0]["logprob"] == 0.
    assert events[-1]["finish_reason"] == finish
    assert events[-1]["stop_token_id"] == first
    assert events[-1]["usage"]["completion_tokens"] == 1


def test_zero_output_does_not_forward_and_grad_mode_does_not_leak():
    engine = Engine(EngineConfig(device="cpu"), model=tiny())
    events = drain(engine.generate(GenRequest([1], max_new_tokens=0)))
    assert len(events) == 1
    assert events[0]["finish_reason"] == "length"
    assert events[0]["usage"]["completion_tokens"] == 0

    async def run():
        async for event in engine.generate(GenRequest([1], max_new_tokens=2)):
            assert torch.is_grad_enabled()
    asyncio.run(run())


@pytest.mark.parametrize("kwargs", [
    {"prompt_ids": []}, {"prompt_ids": [1] * 17}, {"prompt_ids": [16]},
    {"max_new_tokens": -1}, {"max_new_tokens": True}, {"temperature": float("nan")},
    {"top_p": 0}, {"stop": [-1]},
])
def test_torch_invalid_requests_fail_before_forward(kwargs):
    engine = Engine(EngineConfig(device="cpu"), model=tiny())
    options = dict(prompt_ids=[1], max_new_tokens=1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        drain(engine.generate(GenRequest(**options)))
