"""Isolated in-process rollout actors with explicit weight synchronization.

The asyncio interface bounds in-flight streams; TorchEngine forwards still block
its event loop. This is a CPU correctness reference, not distributed/concurrent
GPU execution. See docs/20-sota-repair-contracts.md for the supported contract.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field, replace

from ..serving.engine import Engine, EngineConfig, GenRequest, require_int
from ..serving.sampling import validate_sampling
from .rollout import (
    GroupRollout, _behavior_logprob, _pack_rollouts, _prepare_requests,
    _snapshot_policy, _validate_finish,
)


@dataclass(frozen=True)
class AsyncRolloutConfig:
    group_size: int = 4
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    seq_len: int = 256
    device: str = "cpu"
    max_concurrency: int = 16

    def __post_init__(self) -> None:
        for name in ("group_size", "seq_len", "max_concurrency"):
            require_int(name, getattr(self, name), minimum=1)
        require_int("max_new_tokens", self.max_new_tokens)
        validate_sampling(self.temperature, self.top_p)


class AsyncRolloutEngine:
    """Own one frozen snapshot; sync and generation groups cannot overlap.

    ``sync_weights()`` copies the source learner explicitly, never repoints at a
    live model. The caller must not mutate the source during that synchronous
    copy. Multiple streams within one group share an immutable actor/version;
    overlapping groups or sync attempts fail rather than mixing versions.
    """

    def __init__(self, model, tokenizer, cfg: AsyncRolloutConfig):
        self.cfg = replace(cfg)
        self.tokenizer = tokenizer
        self._learner = model
        self._guard = threading.Lock()
        self._weight_version = 0
        self.model, self.engine = self._build_snapshot(model)

    def _build_snapshot(self, learner) -> tuple:
        model = _snapshot_policy(learner)
        engine = Engine(
            EngineConfig(backend="torch", device=self.cfg.device, max_model_len=self.cfg.seq_len),
            model=model, tokenizer=self.tokenizer,
        )
        return model, engine

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def sync_weights(self, model=None) -> None:
        """Atomically replace the actor snapshot after all streams have drained."""
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("cannot sync actor weights while a group/sync is in flight")
        try:
            learner = self._learner if model is None else model
            snapshot, engine = self._build_snapshot(learner)
            self.model, self.engine, self._learner = snapshot, engine, learner
            self._weight_version += 1
        finally:
            self._guard.release()

    async def _one_rollout(
        self, req: GenRequest, engine: Engine, sem: asyncio.Semaphore,
    ) -> tuple[list[int], list[float], dict]:
        async with sem:
            gen, logps = [], []
            terminal = None
            vocab_size = getattr(getattr(self.model, "cfg", None), "vocab_size", None)
            stream = engine.generate(req)
            try:
                async for chunk in stream:
                    if not isinstance(chunk, dict) or type(chunk.get("done")) is not bool:
                        raise ValueError("rollout event must declare done as a boolean")
                    if terminal is not None:
                        raise ValueError("rollout emitted an event after completion")
                    if chunk["done"]:
                        terminal = chunk
                        continue
                    if len(gen) >= req.max_new_tokens:
                        raise ValueError("rollout exceeded its admitted generation budget")
                    if gen and (gen[-1] == req.eos_id or gen[-1] in (req.stop or ())):
                        raise ValueError("rollout emitted a token after EOS/stop")
                    token = chunk.get("token_id")
                    require_int("token_id", token)
                    if vocab_size is not None and token >= vocab_size:
                        raise ValueError("rollout token is outside the model vocabulary")
                    if chunk.get("logprob_kind", "behavior") != "behavior":
                        raise ValueError("rollout requires behavior log-probabilities")
                    logps.append(_behavior_logprob(chunk.get("logprob")))
                    gen.append(token)
                _validate_finish(req, gen, terminal)
                return gen, logps, terminal
            finally:
                await stream.aclose()

    async def generate_group_async(self, prompts_ids: list[list[int]]) -> GroupRollout:
        """Admit, fully consume, and pack one group without post-hoc cropping."""
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("actor already has a group/sync in flight")
        try:
            cfg, engine, version = self.cfg, self.engine, self.weight_version
            requests, groups = _prepare_requests(
                self.model, self.tokenizer, prompts_ids, group_size=cfg.group_size,
                max_new_tokens=cfg.max_new_tokens, seq_len=cfg.seq_len,
                temperature=cfg.temperature, top_p=cfg.top_p,
            )
            # Loop-local primitives also allow repeated synchronous wrappers,
            # each of which creates its own asyncio event loop.
            sem = asyncio.Semaphore(cfg.max_concurrency)
            tasks = [asyncio.create_task(self._one_rollout(req, engine, sem)) for req in requests]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                # gather does not cancel siblings on failure. Drain them before
                # releasing the snapshot guard, including caller cancellation.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return _pack_rollouts(
                requests, groups, results, tokenizer=self.tokenizer,
                temperature=cfg.temperature, top_p=cfg.top_p, weight_version=version,
            )
        finally:
            self._guard.release()

    def generate_group(self, prompts_ids: list[list[int]]) -> GroupRollout:
        """Synchronous wrapper; async callers must await generate_group_async."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate_group_async(prompts_ids))
        raise RuntimeError("use await generate_group_async inside a running event loop")


@dataclass
class RolloutBuffer:
    """Async queue of scored rollout groups between actors and the learner."""

    maxsize: int = 8
    _q: asyncio.Queue = field(default_factory=asyncio.Queue)

    def __post_init__(self) -> None:
        self._q = asyncio.Queue(maxsize=self.maxsize)

    async def put(self, item) -> None:
        await self._q.put(item)

    async def get(self):
        return await self._q.get()

    def qsize(self) -> int:
        return self._q.qsize()
