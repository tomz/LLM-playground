"""Tests for the vLLM serving backend adapter.

The adapter is meant to be a drop-in replacement for :class:`TorchEngine` in
the rollout / serving paths. Since we don't want to pull vLLM (and a GPU) into
the unit-test environment, we exercise the adapter against a fake ``vllm.LLM``
that returns canned token streams. The schema (``{token_id, logprob, done,
usage}``) must match ``TorchEngine`` exactly.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Fake vLLM module
# ---------------------------------------------------------------------------


class _FakeLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob


class _FakeCompletionOutput:
    def __init__(self, token_ids, logprobs, text=""):
        self.token_ids = list(token_ids)
        self.logprobs = list(logprobs)
        self.text = text


class _FakeRequestOutput:
    def __init__(self, outputs):
        self.outputs = list(outputs)


class _FakeLLM:
    """Mimics vllm.LLM.generate signature for the tests."""

    def __init__(self, *args, canned=None, **kwargs):
        self.constructed_with = kwargs
        # The token stream every .generate() call will replay.
        self.canned = canned or [10, 11, 12, 13]
        self.call_log: list[dict] = []
        # Test hook: weight-loading path is monkeypatched on demand.

    def generate(self, prompts, sampling_params):
        prompt = prompts[0]
        self.call_log.append({
            "prompt_token_ids": list(prompt.get("prompt_token_ids", [])),
            "max_tokens": getattr(sampling_params, "max_tokens", None),
            "temperature": getattr(sampling_params, "temperature", None),
            "stop_token_ids": getattr(sampling_params, "stop_token_ids", None),
            "logprobs": getattr(sampling_params, "logprobs", None),
        })
        n = int(getattr(sampling_params, "max_tokens", len(self.canned)) or len(self.canned))
        toks = list(self.canned)[:n]
        # vLLM-style logprobs: list of dict[token_id, Logprob] aligned to tokens.
        logprobs = [{int(t): _FakeLogprob(-0.5 - 0.01 * i)} for i, t in enumerate(toks)]
        return [_FakeRequestOutput([_FakeCompletionOutput(toks, logprobs, text="hello world")])]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _install_fake_vllm(monkeypatch, llm_cls=_FakeLLM, sp_cls=_FakeSamplingParams):
    fake = types.ModuleType("vllm")
    fake.LLM = llm_cls
    fake.SamplingParams = sp_cls
    monkeypatch.setitem(sys.modules, "vllm", fake)
    # Force re-import so vllm_engine picks the fake.
    if "platform.serving.vllm_engine" in sys.modules:
        importlib.reload(sys.modules["platform.serving.vllm_engine"])
    return fake


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain(agen):
    async def _go():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return out
    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_engine_routes_vllm_backend(monkeypatch):
    """Engine(cfg=…backend=vllm) no longer raises NotImplementedError."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model"))
    assert eng._impl.__class__.__name__ == "VLLMEngine"


def test_vllm_engine_generate_yields_torch_engine_schema(monkeypatch):
    """Each yielded chunk must look like TorchEngine's chunk; final has usage."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig, GenRequest

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model"))
    req = GenRequest(prompt_ids=[1, 2, 3], max_new_tokens=4)
    chunks = _drain(eng.generate(req))

    # 4 token chunks + 1 done chunk.
    assert len(chunks) == 5
    token_chunks = [c for c in chunks if not c.get("done")]
    done = chunks[-1]
    assert len(token_chunks) == 4
    for c in token_chunks:
        assert set(c) == {"token_id", "logprob", "text", "done"}
        assert isinstance(c["token_id"], int)
        assert isinstance(c["logprob"], float)
        assert c["done"] is False
    assert done["done"] is True
    assert done["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}


def test_vllm_engine_passes_sampling_params(monkeypatch):
    """SamplingParams kwargs match what callers expect (temperature, top_p, stop)."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig, GenRequest

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model"))
    req = GenRequest(
        prompt_ids=[5, 6, 7], max_new_tokens=2, temperature=0.3,
        top_p=0.7, stop=[42],
    )
    _drain(eng.generate(req))

    log = eng._impl._llm.call_log[-1]
    assert log["max_tokens"] == 2
    assert abs(log["temperature"] - 0.3) < 1e-9
    assert log["stop_token_ids"] == [42]
    # We always request at least 1 logprob so RL importance ratios are correct.
    assert (log["logprobs"] or 0) >= 1


def test_vllm_engine_logprob_threads_through(monkeypatch):
    """The chunk's logprob must match the vLLM Logprob entry for the chosen token."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig, GenRequest

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model"))
    chunks = _drain(eng.generate(GenRequest(prompt_ids=[1], max_new_tokens=3)))
    token_chunks = [c for c in chunks if not c.get("done")]
    # Fake produced -0.5, -0.51, -0.52 for the three tokens (see _FakeLLM).
    assert abs(token_chunks[0]["logprob"] - (-0.5)) < 1e-9
    assert abs(token_chunks[1]["logprob"] - (-0.51)) < 1e-9
    assert abs(token_chunks[2]["logprob"] - (-0.52)) < 1e-9


def test_vllm_engine_requires_ckpt(monkeypatch):
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig

    with pytest.raises(ValueError):
        Engine(EngineConfig(backend="vllm", ckpt=""))


def test_vllm_engine_rejects_in_memory_model(monkeypatch):
    """vLLM can't wrap a live torch.nn.Module; surface that clearly."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig

    with pytest.raises(NotImplementedError):
        Engine(EngineConfig(backend="vllm", ckpt="fake/model"), model=object())


def test_vllm_engine_drops_unknown_llm_kwargs(monkeypatch):
    """If the installed vLLM doesn't recognise an option we requested, we must
    quietly drop it instead of crashing on construction."""
    class _PickyLLM(_FakeLLM):
        def __init__(self, *args, **kwargs):
            for forbidden in ("enable_prefix_caching", "enable_chunked_prefill"):
                if forbidden in kwargs:
                    raise TypeError(f"__init__() got an unexpected keyword argument '{forbidden}'")
            super().__init__(*args, **kwargs)

    _install_fake_vllm(monkeypatch, llm_cls=_PickyLLM)
    from platform.serving.engine import Engine, EngineConfig

    # Should not raise even though the picky LLM rejects prefix-caching / chunked-prefill.
    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model"))
    assert eng._impl._llm.__class__.__name__ == "_PickyLLM"
    assert "enable_prefix_caching" not in eng._impl._llm.constructed_with


def test_vllm_engine_passes_tp_pp_and_dtype(monkeypatch):
    """tp/pp/dtype from EngineConfig must reach vLLM."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model", tp=4, pp=2, dtype="bf16"))
    kwargs = eng._impl._llm.constructed_with
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["pipeline_parallel_size"] == 2
    assert kwargs["dtype"] == "bfloat16"


def test_vllm_engine_speculative_draft_forwarded(monkeypatch):
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model",
                              speculative_draft="some/draft"))
    assert eng._impl._llm.constructed_with["speculative_model"] == "some/draft"


def test_vllm_engine_update_weights_round_trips(monkeypatch):
    """update_weights walks the vLLM internals; we stub the path and confirm
    the state_dict reaches the leaf model."""
    _install_fake_vllm(monkeypatch)
    from platform.serving.engine import Engine, EngineConfig

    eng = Engine(EngineConfig(backend="vllm", ckpt="fake/model"))

    received: list[dict] = []

    class _Leaf:
        def load_state_dict(self, sd):
            received.append(dict(sd))

    chain = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            model_executor=types.SimpleNamespace(
                driver_worker=types.SimpleNamespace(
                    model_runner=types.SimpleNamespace(model=_Leaf())
                )
            )
        )
    )
    # Patch the internals on the running LLM instance.
    eng._impl._llm.llm_engine = chain.llm_engine

    eng.update_weights({"layer.weight": "tensor-like"})
    assert received and received[-1] == {"layer.weight": "tensor-like"}


def test_engine_update_weights_unsupported_backend_raises(monkeypatch):
    """TorchEngine has no update_weights (it shares the object); the wrapper
    must report that clearly rather than silently no-op."""
    pytest.importorskip("torch")
    import torch
    from platform.model.config import ModelConfig
    from platform.model.transformer import Transformer
    from platform.serving.engine import Engine, EngineConfig

    torch.manual_seed(0)
    model = Transformer(ModelConfig(
        vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
        d_model=32, d_ffn=64, max_seq_len=64,
    ))
    eng = Engine(EngineConfig(backend="torch", device="cpu"), model=model)
    with pytest.raises(NotImplementedError):
        eng.update_weights({})


def test_vllm_engine_missing_package_raises_importerror(monkeypatch):
    """If vllm isn't importable, the user gets an actionable ImportError —
    not the old ``NotImplementedError`` masking-the-real-problem."""
    monkeypatch.setitem(sys.modules, "vllm", None)  # force ImportError
    # Reload the module so the import statement re-runs against the stub.
    if "platform.serving.vllm_engine" in sys.modules:
        importlib.reload(sys.modules["platform.serving.vllm_engine"])
    from platform.serving.engine import Engine, EngineConfig

    with pytest.raises(ImportError):
        Engine(EngineConfig(backend="vllm", ckpt="fake/model"))
