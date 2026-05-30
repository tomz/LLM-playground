"""Async actor–learner rollout for GRPO (see docs/15-reasoning-rl-rlvr.md).

The synchronous `sample_group` in `rollout.py` is a correctness reference. Real
RLVR decouples **generation** (many long rollouts, throughput-bound) from the
**learner** (gradient updates), running them as an async actor–learner loop with
periodic weight sync. This module provides that structure on top of the existing
serving `Engine` (which now has a real KV-cache decode path):

  AsyncRolloutEngine  — wraps a serving Engine; generates G samples/prompt
                        concurrently via asyncio; weight-sync from the learner.
  RolloutBuffer       — async queue carrying (prompt, samples, rewards) groups
                        between actors and the learner.
  generate_group_async — the async analogue of sample_group, returning the same
                        GroupRollout so the GRPO learner is unchanged.

Swapping to vLLM/SGLang is a backend change inside `Engine` (EngineConfig.backend
== 'vllm'); this orchestration layer stays the same. On CPU/today it runs the
in-process TorchEngine, proving the async path end-to-end.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import torch

from ..serving.engine import Engine, EngineConfig, GenRequest
from .rollout import GroupRollout


@dataclass
class AsyncRolloutConfig:
    group_size: int = 4
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    seq_len: int = 256
    device: str = "cpu"
    max_concurrency: int = 16          # cap simultaneous in-flight rollouts


class AsyncRolloutEngine:
    """Async generation actor backed by a serving Engine.

    The learner owns the policy weights; the actor holds an Engine wrapping the
    same model object. ``sync_weights`` re-points the actor at the learner's
    latest state (in-process here; an out-of-process vLLM actor would load a
    weight broadcast). Because the Engine shares the model object, in-process
    sync is automatic, but we keep the explicit hook for the distributed case.
    """

    def __init__(self, model, tokenizer, cfg: AsyncRolloutConfig):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        self.engine = Engine(
            EngineConfig(backend="torch", device=cfg.device), model=model, tokenizer=tokenizer
        )
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self.weight_version = 0

    def sync_weights(self, model=None) -> None:
        """Point the actor at the latest learner weights. In-process this is a
        no-op (shared object); out-of-process this would reload a broadcast."""
        if model is not None and model is not self.model:
            self.model = model
            self.engine = Engine(
                EngineConfig(backend="torch", device=self.cfg.device),
                model=model, tokenizer=self.tokenizer,
            )
        self.weight_version += 1

    async def _one_rollout(self, prompt_ids: list[int]) -> tuple[list[int], list[float]]:
        async with self._sem:
            req = GenRequest(
                prompt_ids=list(prompt_ids),
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
            gen: list[int] = []
            logps: list[float] = []
            async for chunk in self.engine.generate(req):
                if not chunk.get("done"):
                    gen.append(chunk["token_id"])
                    logps.append(float(chunk.get("logprob", 0.0)))
            return gen, logps

    async def generate_group_async(self, prompts_ids: list[list[int]]) -> GroupRollout:
        """Generate ``group_size`` samples for each prompt concurrently and pack
        them into a :class:`GroupRollout` (same shape the GRPO learner expects)."""
        cfg = self.cfg
        tasks = []
        group_index: list[int] = []
        rows: list[list[int]] = []
        for gi, p in enumerate(prompts_ids):
            for _ in range(cfg.group_size):
                tasks.append(self._one_rollout(p))
                group_index.append(gi)
                rows.append(list(p))

        results = await asyncio.gather(*tasks)
        gen_lists = [g for g, _ in results]
        logp_lists = [lp for _, lp in results]

        # Pack into padded [N, T] tensors with a response mask (mirrors sample_group).
        pad_id, eos_id = self.tokenizer.pad_id, self.tokenizer.eos_id
        N = len(rows)
        prompt_lens = [len(p) for p in rows]
        full_rows = []
        for p, gen in zip(rows, gen_lists):
            full_rows.append(list(p) + list(gen))
        T_total = min(cfg.seq_len, max(len(r) for r in full_rows)) if full_rows else 0

        ids = torch.full((N, T_total), pad_id, dtype=torch.long)
        resp_mask = torch.zeros((N, T_total), dtype=torch.float32)
        behavior_logp = torch.zeros((N, T_total), dtype=torch.float32)
        response_text: list[str] = []
        for i, (p, gen, lps) in enumerate(zip(rows, gen_lists, logp_lists)):
            seq = (list(p) + list(gen))[:T_total]
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            gen_start = min(len(p), T_total)
            gen_end = min(len(p) + len(gen), T_total)
            resp_mask[i, gen_start:gen_end] = 1.0
            n_gen = gen_end - gen_start
            if n_gen > 0:
                behavior_logp[i, gen_start:gen_end] = torch.tensor(
                    lps[:n_gen], dtype=torch.float32
                )
            response_text.append(self.tokenizer.decode(list(gen)))
            _ = eos_id  # eos handled by the engine's stop logic upstream

        return GroupRollout(
            ids=ids,
            resp_mask=resp_mask,
            group_index=torch.tensor(group_index, dtype=torch.long),
            prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
            response_text=response_text,
            behavior_logp=behavior_logp,
        )

    def generate_group(self, prompts_ids: list[list[int]]) -> GroupRollout:
        """Synchronous convenience wrapper around :meth:`generate_group_async`."""
        return asyncio.run(self.generate_group_async(prompts_ids))


@dataclass
class RolloutBuffer:
    """Async queue of scored rollout groups between actors and the learner."""

    maxsize: int = 8
    _q: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())

    def __post_init__(self):
        self._q = asyncio.Queue(maxsize=self.maxsize)

    async def put(self, item) -> None:
        await self._q.put(item)

    async def get(self):
        return await self._q.get()

    def qsize(self) -> int:
        return self._q.qsize()
