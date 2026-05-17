"""In-process PyTorch serving backend.

Toy implementation: re-encodes the full prefix each step (no KV cache). At
the toy scale we exercise in tests this is plenty fast; the KV-cache version
lives in `distgpt`'s sampler if you want to port it later.
"""
from __future__ import annotations
from typing import AsyncIterator

import torch

from .engine import EngineConfig, GenRequest


class TorchEngine:
    def __init__(self, cfg: EngineConfig, model=None, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            if not cfg.ckpt:
                raise ValueError("TorchEngine requires `model=` or `cfg.ckpt` set")
            model = torch.load(cfg.ckpt, map_location=self.device, weights_only=False)
        self.model = model.to(self.device).eval()

    async def generate(self, req: GenRequest) -> AsyncIterator[dict]:
        stop = set(req.stop or [])
        prompt = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        with torch.no_grad():
            for _ in range(req.max_new_tokens):
                logits, _ = self.model(prompt)
                next_logits = logits[:, -1, :].float()
                if req.temperature <= 0.0:
                    next_id = int(next_logits.argmax(dim=-1).item())
                else:
                    probs = (next_logits / req.temperature).softmax(dim=-1)
                    if 0.0 < req.top_p < 1.0:
                        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                        cum = sorted_probs.cumsum(dim=-1)
                        mask = cum > req.top_p
                        # keep at least one token
                        mask[..., 0] = False
                        sorted_probs[mask] = 0.0
                        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                        pick = torch.multinomial(sorted_probs, 1)
                        next_id = int(sorted_idx.gather(-1, pick).item())
                    else:
                        next_id = int(torch.multinomial(probs, 1).item())
                generated.append(next_id)
                text = None
                if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
                    try:
                        text = self.tokenizer.decode([next_id])
                    except Exception:
                        text = None
                yield {"token_id": next_id, "text": text, "done": False}
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
