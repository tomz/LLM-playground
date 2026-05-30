"""In-process PyTorch serving backend.

Uses the model's incremental KV-cache decode path
(`Transformer.forward_with_cache` + `platform.model.kv_cache.KVCache`) when the
model supports it: prefill the prompt once, then decode one token per step over
the cache — O(T) per token instead of re-encoding the whole prefix. Falls back to
full re-encode for models without `forward_with_cache`.
"""
from __future__ import annotations
from typing import AsyncIterator

import torch

from .engine import EngineConfig, GenRequest


class TorchEngine:
    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        if cfg.device == "cpu":
            self.device = torch.device("cpu")
        elif cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            if not cfg.ckpt:
                raise ValueError("TorchEngine requires `model=` or `cfg.ckpt` set")
            model = torch.load(cfg.ckpt, map_location=self.device, weights_only=False)
        self.model = model.to(self.device).eval()
        self._use_cache = hasattr(self.model, "forward_with_cache")

    def _sample(self, next_logits: torch.Tensor, req: GenRequest) -> int:
        next_logits = next_logits.float()
        if req.temperature <= 0.0:
            return int(next_logits.argmax(dim=-1).item())
        probs = (next_logits / req.temperature).softmax(dim=-1)
        if 0.0 < req.top_p < 1.0:
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cum = sorted_probs.cumsum(dim=-1)
            mask = cum > req.top_p
            mask[..., 0] = False
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            pick = torch.multinomial(sorted_probs, 1)
            return int(sorted_idx.gather(-1, pick).item())
        return int(torch.multinomial(probs, 1).item())

    def _decode_text(self, token_id: int):
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode([token_id])
            except Exception:
                return None
        return None

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        stop = set(req.stop or [])
        prompt = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        with torch.no_grad():
            if self._use_cache:
                from platform.model.kv_cache import KVCache
                cache = KVCache(self.model.cfg.n_layer)
                # Prefill the whole prompt in one pass.
                logits = self.model.forward_with_cache(prompt, cache)
                next_logits = logits[:, -1, :]
                for _ in range(req.max_new_tokens):
                    next_id = self._sample(next_logits, req)
                    generated.append(next_id)
                    yield {"token_id": next_id, "text": self._decode_text(next_id), "done": False}
                    if next_id in stop:
                        break
                    step = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    logits = self.model.forward_with_cache(step, cache)
                    next_logits = logits[:, -1, :]
            else:
                for _ in range(req.max_new_tokens):
                    logits, _ = self.model(prompt)
                    next_id = self._sample(logits[:, -1, :], req)
                    generated.append(next_id)
                    yield {"token_id": next_id, "text": self._decode_text(next_id), "done": False}
                    if next_id in stop:
                        break
                    prompt = torch.cat(
                        [prompt, torch.tensor([[next_id]], dtype=torch.long, device=self.device)],
                        dim=1,
                    )
        yield {
            "done": True,
            "usage": {
                "prompt_tokens": len(req.prompt_ids),
                "completion_tokens": len(generated),
            },
        }
