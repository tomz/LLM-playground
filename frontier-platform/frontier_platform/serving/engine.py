"""Inference backend dispatch and validated, token-exact generation requests."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import AsyncIterator

from .sampling import validate_sampling


def require_int(name: str, value: int, *, minimum: int = 0) -> None:
    """Reject bool, fractional counts, and values below the declared minimum."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass
class EngineConfig:
    backend: str = "torch"     # 'torch' | 'vllm'; other backends remain interfaces
    ckpt: str = ""
    tp: int = 1
    pp: int = 1
    dtype: str = "fp32"
    device: str = "auto"
    max_model_len: int = 32_768
    max_num_seqs: int = 256
    enable_prefix_cache: bool = True
    enable_chunked_prefill: bool = True
    speculative_draft: str | None = None

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "max_model_len", "max_num_seqs"):
            require_int(name, getattr(self, name), minimum=1)
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if self.dtype not in {"auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}:
            raise ValueError("unsupported dtype")


@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[int] | None = None
    logprobs: int = 0
    request_id: str = ""
    eos_id: int | None = None


def validated_request(
    req: GenRequest, *, max_model_len: int, vocab_size: int | None = None,
    eos_id: int | None = None,
) -> GenRequest:
    """Snapshot inputs and reject invalid/overlong requests before any work.

    Empty prompts are rejected (callers may explicitly supply BOS). Zero output
    tokens is valid and produces only a length-finished usage event. A request
    that exceeds context is rejected, never silently cropped. Rollout callers
    must cap generation *before* dispatch to their learner context length.
    """
    require_int("max_model_len", max_model_len, minimum=1)
    require_int("max_new_tokens", req.max_new_tokens)
    require_int("logprobs", req.logprobs)
    validate_sampling(req.temperature, req.top_p)
    if not isinstance(req.prompt_ids, (list, tuple)) or not req.prompt_ids:
        raise ValueError("prompt_ids must be a nonempty token sequence")
    if req.stop is not None and not isinstance(req.stop, (list, tuple)):
        raise ValueError("stop must be a sequence of token IDs")
    effective_eos = req.eos_id if req.eos_id is not None else eos_id
    for name, ids in (("prompt_ids", req.prompt_ids), ("stop", req.stop or ()),
                      ("eos_id", () if effective_eos is None else (effective_eos,))):
        for tid in ids:
            require_int(name, tid)
            if vocab_size is not None and tid >= vocab_size:
                raise ValueError(f"{name} contains a token outside vocab_size={vocab_size}")
    if len(req.prompt_ids) + req.max_new_tokens > max_model_len:
        raise ValueError("prompt plus max_new_tokens exceeds max_model_len")
    return replace(req, prompt_ids=list(req.prompt_ids), stop=list(req.stop or ()), eos_id=effective_eos)


def finished_event(
    req: GenRequest, count: int, reason: str, *, logprob_kind: str,
    stop_token_id: int | None = None,
) -> dict:
    """One terminal event for a successfully completed, fully consumed request."""
    return {
        "done": True,
        "finish_reason": reason,
        "stop_token_id": stop_token_id,
        "logprob_kind": logprob_kind,
        "usage": {"prompt_tokens": len(req.prompt_ids), "completion_tokens": count},
    }


class Engine:
    """Dispatch to a CPU reference or an optional, batch-replay vLLM adapter."""

    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None, **kwargs):
        self.cfg = cfg
        if cfg.backend == "torch":
            from .torch_engine import TorchEngine
            self._impl = TorchEngine(cfg, model=model, tokenizer=tokenizer)
        elif cfg.backend == "vllm":
            from .vllm_engine import VLLMEngine
            self._impl = VLLMEngine(cfg, tokenizer=tokenizer, model=model, **kwargs)
        else:
            raise NotImplementedError(
                f"backend {cfg.backend!r} not wired; use 'torch' or the optional 'vllm' adapter"
            )

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        stream = self._impl.generate(req)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.aclose()

    def update_weights(self, state_dict: dict) -> None:
        """Forward an explicit backend weight update; never silently no-op."""
        impl_update = getattr(self._impl, "update_weights", None)
        if impl_update is None:
            raise NotImplementedError(
                f"{type(self._impl).__name__} does not support update_weights; "
                "construct a new TorchEngine with an isolated model snapshot"
            )
        impl_update(state_dict)
