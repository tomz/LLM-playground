"""Inference engine abstraction. Dispatches to a concrete backend."""
from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class EngineConfig:
    backend: str = "torch"     # 'torch' | 'vllm' | 'trtllm' | 'sglang'
    ckpt: str = ""
    tp: int = 1
    pp: int = 1
    dtype: str = "fp32"        # bf16 not used on Pascal; fp16 or fp32
    device: str = "auto"       # 'auto' | 'cpu' | 'cuda' — override for tests/old GPUs
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
    """Dispatch wrapper. The actual code lives in `torch_engine.TorchEngine`."""

    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None):
        self.cfg = cfg
        if cfg.backend == "torch":
            from .torch_engine import TorchEngine
            self._impl = TorchEngine(cfg, model=model, tokenizer=tokenizer)
        elif cfg.backend == "vllm":
            try:
                import vllm  # noqa: F401
            except ImportError as e:
                raise NotImplementedError(
                    "vllm backend requires `pip install vllm` (and an Ampere+ GPU)"
                ) from e
            raise NotImplementedError("vllm wrapper not wired in this blueprint")
        else:
            raise NotImplementedError(
                f"backend {cfg.backend!r} not wired; use 'torch' (in-process) or 'vllm'"
            )

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        async for chunk in self._impl.generate(req):
            yield chunk
