"""vLLM serving backend.

Replaces the ``NotImplementedError`` branch in :class:`Engine` with a real
adapter that talks to the installed ``vllm`` package. Same API as
:class:`TorchEngine`: ``generate(req) -> AsyncIterator[dict]`` yielding chunks
matching ``{"token_id", "logprob", "text", "done"}`` and a final ``{"done":
True, "usage": ...}``.

Two things this unlocks:

1. **Scaled RLVR rollouts.** :class:`AsyncRolloutEngine` already orchestrates
   many concurrent rollouts over an :class:`Engine`; pointing it at the vLLM
   backend gives the throughput a frontier-class reasoning RL phase needs.
2. **Product serving.** Continuous batching, paged KV, prefix cache, and
   chunked prefill come for free from vLLM. Speculative decoding via
   ``EngineConfig.speculative_draft``.

This module supports modern vLLM (``LLM`` + ``SamplingParams``). We import
vLLM lazily inside ``__init__`` so plain pip installs of frontier-platform
don't pull in the (large) vLLM dependency. The fake-vllm test
(``test_vllm_engine_fake_backend``) verifies the adapter against a stub
without requiring the real package.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .engine import EngineConfig, GenRequest


class VLLMEngine:
    """Adapter around ``vllm.LLM`` with the same chunk schema as TorchEngine.

    Parameters
    ----------
    cfg : EngineConfig
        ``cfg.backend`` should be ``'vllm'``. ``cfg.ckpt`` is the HF model id
        or local path. ``cfg.tp`` maps to vLLM's ``tensor_parallel_size``,
        ``cfg.pp`` to ``pipeline_parallel_size``. ``cfg.dtype`` is passed
        through (vLLM understands ``'auto' | 'fp16' | 'bf16' | 'fp32'``).
    tokenizer : optional
        Forwarded to vLLM if provided, else vLLM loads the model's own tokenizer.
    llm : optional
        Pre-constructed ``vllm.LLM`` (used by tests to inject a fake). When
        provided, ``cfg.ckpt`` is not required.
    """

    def __init__(self, cfg: EngineConfig, *, tokenizer=None, llm=None, model=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        if model is not None:
            # vLLM doesn't accept an arbitrary in-memory nn.Module; tell the
            # caller clearly rather than silently ignoring.
            raise NotImplementedError(
                "VLLMEngine cannot wrap an in-memory torch model; pass `cfg.ckpt` "
                "(an HF model id or local path) instead."
            )
        if llm is not None:
            self._llm = llm
            self._sampling_cls = _resolve_sampling_params_cls(llm)
            return

        try:
            import vllm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vllm backend requires `pip install vllm` (and an Ampere+ GPU "
                "for full performance; CPU build exists but is slow)."
            ) from e

        from vllm import LLM, SamplingParams  # type: ignore

        if not cfg.ckpt:
            raise ValueError("VLLMEngine requires `cfg.ckpt` (HF id or local path)")

        llm_kwargs: dict = dict(
            model=cfg.ckpt,
            tensor_parallel_size=int(cfg.tp),
            pipeline_parallel_size=int(cfg.pp),
            dtype=_normalise_dtype(cfg.dtype),
            max_model_len=int(cfg.max_model_len),
            enable_prefix_caching=bool(cfg.enable_prefix_cache),
            enable_chunked_prefill=bool(cfg.enable_chunked_prefill),
        )
        if cfg.speculative_draft:
            # vLLM versions differ on the speculative-decoding API; pass the
            # draft model in the form most modern releases accept and let vLLM
            # raise a clear error if the installed version disagrees.
            llm_kwargs["speculative_model"] = cfg.speculative_draft
        # Drop kwargs the installed vLLM doesn't recognise rather than failing
        # at construction time — older vLLMs lack some of these.
        self._llm = _construct_llm(LLM, llm_kwargs)
        self._sampling_cls = SamplingParams

    # ------------------------------------------------------------------
    # Weight sync hook for out-of-process RL actors
    # ------------------------------------------------------------------

    def update_weights(self, state_dict: dict) -> None:
        """Push fresh weights into the engine's running model (RLVR actor path).

        Modern vLLM exposes ``llm.llm_engine.model_executor.driver_worker.
        model_runner.model.load_state_dict``. Where unavailable, raises a
        ``RuntimeError`` with the version requirement so the rollout layer can
        fall back to a slow restart-on-sync path.
        """
        path = (
            "llm_engine",
            "model_executor",
            "driver_worker",
            "model_runner",
            "model",
        )
        obj = self._llm
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                raise RuntimeError(
                    "installed vLLM does not expose the weight-loading path; "
                    "upgrade to a release that exposes "
                    "`LLM.llm_engine.model_executor.driver_worker.model_runner.model`."
                )
        obj.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        """Yield the same chunk schema TorchEngine produces.

        vLLM's batch-oriented ``LLM.generate`` returns *all* outputs at once, so
        we run the call in a worker thread (so the asyncio loop isn't blocked)
        and replay the resulting tokens as a chunk stream. This keeps the
        interface symmetric with TorchEngine; callers that need true
        token-by-token streaming can swap in ``AsyncLLMEngine`` later without
        changing the upstream RL / serving code."""
        params = self._build_sampling_params(req)
        prompt = {"prompt_token_ids": list(req.prompt_ids)}
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._llm.generate([prompt], params),
        )
        # vLLM returns list[RequestOutput]; one per prompt. We only sent one.
        out = outputs[0]
        completion = out.outputs[0] if out.outputs else None
        if completion is None:
            yield {"done": True, "usage": {
                "prompt_tokens": len(req.prompt_ids),
                "completion_tokens": 0,
            }}
            return

        token_ids = list(getattr(completion, "token_ids", []))
        text = getattr(completion, "text", "")
        logprobs = getattr(completion, "logprobs", None) or []
        for i, tid in enumerate(token_ids):
            lp = 0.0
            if logprobs and i < len(logprobs) and logprobs[i] is not None:
                # vLLM logprobs are a dict[token_id, Logprob]; pull the chosen one.
                entry = logprobs[i].get(int(tid)) if hasattr(logprobs[i], "get") else None
                if entry is not None:
                    lp = float(getattr(entry, "logprob", 0.0))
            yield {
                "token_id": int(tid),
                "logprob": lp,
                "text": None,
                "done": False,
            }
        yield {
            "done": True,
            "usage": {
                "prompt_tokens": len(req.prompt_ids),
                "completion_tokens": len(token_ids),
            },
            "text": text,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sampling_params(self, req: GenRequest):
        kwargs = dict(
            max_tokens=int(req.max_new_tokens),
            temperature=float(req.temperature),
            top_p=float(req.top_p),
        )
        if req.stop:
            kwargs["stop_token_ids"] = list(req.stop)
        if req.logprobs and req.logprobs > 0:
            kwargs["logprobs"] = int(req.logprobs)
        else:
            # Always request the chosen-token logprob so RL importance ratios
            # stay correct — at the rollout layer we read it back per chunk.
            kwargs["logprobs"] = 1
        return self._sampling_cls(**kwargs)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _normalise_dtype(d: str) -> str:
    """Map our short dtype strings to what vLLM accepts."""
    m = {
        "fp32": "float32", "float32": "float32",
        "fp16": "float16", "float16": "float16",
        "bf16": "bfloat16", "bfloat16": "bfloat16",
        "auto": "auto",
    }
    return m.get(d, d)


def _construct_llm(LLM, kwargs: dict):
    """Construct ``vllm.LLM`` dropping kwargs the installed version doesn't take.

    vLLM's ``LLM`` signature has grown over time; we want a single adapter that
    works against a range of versions rather than pinning the user."""
    try:
        return LLM(**kwargs)
    except TypeError as e:
        msg = str(e)
        for key in (
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "speculative_model",
            "pipeline_parallel_size",
        ):
            if key in msg and key in kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k != key}
                return _construct_llm(LLM, kwargs)
        raise


def _resolve_sampling_params_cls(llm):
    """Best-effort import of ``SamplingParams`` for a pre-built ``llm``."""
    try:
        from vllm import SamplingParams  # type: ignore
        return SamplingParams
    except Exception:  # pragma: no cover - test injects its own
        cls = getattr(llm, "SamplingParams", None)
        if cls is not None:
            return cls
        raise RuntimeError(
            "could not locate vllm.SamplingParams; pass a stub on the llm object"
        )
