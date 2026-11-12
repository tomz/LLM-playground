"""In-process PyTorch reference serving, with incremental KV-cache decoding.

Token events are produced incrementally, but CPU forward calls block the event
loop. This is not a continuous-batching server or a concurrent GPU actor. The
caller owns model lifetime; RL uses a separate frozen copy of learner weights.
"""
from __future__ import annotations

from typing import AsyncIterator

import torch

from .engine import EngineConfig, GenRequest, finished_event, validated_request
from .sampling import sample_token


class TorchEngine:
    """Token-exact reference; log-probs are from the actual sampling distribution."""

    logprob_kind = "behavior"
    streaming_mode = "incremental_blocking"

    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        if cfg.tp != 1 or cfg.pp != 1 or cfg.speculative_draft is not None:
            raise NotImplementedError("TorchEngine supports only TP=PP=1 without speculative decoding")
        self.device = torch.device(
            "cuda" if cfg.device == "auto" and torch.cuda.is_available()
            else "cpu" if cfg.device == "auto" else cfg.device
        )
        if model is None:
            if not cfg.ckpt:
                raise ValueError("TorchEngine requires `model=` or `cfg.ckpt` set")
            # This path accepts only caller-trusted serialized model objects.
            model = torch.load(cfg.ckpt, map_location=self.device, weights_only=False)
        self.model = model.to(self.device).eval()
        self._use_cache = hasattr(self.model, "forward_with_cache")
        model_cfg = getattr(model, "cfg", None)
        self.max_model_len = min(cfg.max_model_len, getattr(model_cfg, "max_seq_len", cfg.max_model_len))
        self.vocab_size = getattr(model_cfg, "vocab_size", getattr(tokenizer, "vocab_size", None))

    def _sample(self, next_logits: torch.Tensor, req: GenRequest) -> tuple[int, float]:
        return sample_token(next_logits, temperature=req.temperature, top_p=req.top_p)

    def _decode_text(self, token_id: int) -> str | None:
        """A per-token convenience fragment, not an exact concatenated decoding.

        Byte/BPE boundaries can span tokens. Reward consumers must decode the
        complete returned token sequence once, never join these fragments.
        """
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode([token_id])
        return None

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        req = validated_request(
            req, max_model_len=self.max_model_len, vocab_size=self.vocab_size,
            eos_id=getattr(self.tokenizer, "eos_id", None),
        )
        stop = set(req.stop or ())
        prompt = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
        count, reason, stop_id = 0, "length", None
        if req.max_new_tokens:
            cache = None
            if self._use_cache:
                from ..model.kv_cache import KVCache
                cache = KVCache(self.model.cfg.n_layer)
            for i in range(req.max_new_tokens):
                # Never suspend an async generator inside no_grad(): doing so
                # changes grad mode in the consumer and other event-loop tasks.
                with torch.no_grad():
                    if cache is not None:
                        logits = self.model.forward_with_cache(prompt, cache)
                    else:
                        logits, _ = self.model(prompt)
                    next_id, logp = self._sample(logits[:, -1, :], req)
                count += 1
                yield {"token_id": next_id, "logprob": logp, "text": self._decode_text(next_id), "done": False}
                if next_id == req.eos_id:
                    reason, stop_id = "eos", next_id
                    break
                if next_id in stop:
                    reason, stop_id = "stop", next_id
                    break
                if i + 1 < req.max_new_tokens:
                    step = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    prompt = step if cache is not None else torch.cat([prompt, step], dim=1)
        yield finished_event(req, count, reason, logprob_kind=self.logprob_kind, stop_token_id=stop_id)
