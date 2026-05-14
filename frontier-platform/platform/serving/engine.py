"""Inference engine abstraction. Backends: vllm | trtllm | sglang."""
from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class EngineConfig:
    backend: str = "vllm"
    ckpt: str = ""
    tp: int = 1
    pp: int = 1
    dtype: str = "bf16"        # 'bf16' | 'fp8' | 'int4'
    max_model_len: int = 32_768
    max_num_seqs: int = 256
    enable_prefix_cache: bool = True
    enable_chunked_prefill: bool = True
    speculative_draft: str | None = None


@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: list[int] | None = None
    logprobs: int = 0
    request_id: str = ""


class Engine:
    def __init__(self, cfg: EngineConfig): ...
    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        """Yields incremental tokens + (final) logprobs/usage."""
        raise NotImplementedError
        yield  # pragma: no cover
