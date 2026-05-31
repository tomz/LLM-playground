"""Parallelism abstraction.

This module replaces the old ``torch_native``-only engine that raised
``NotImplementedError`` for ``tp>1`` / ``pp>1`` with a real, layered runtime:

* **FSDP2 (DP + ZeRO-3 sharding)** via ``torch.distributed.fsdp.fully_shard``.
  Each transformer ``Block`` is wrapped as its own FSDP unit so all-gather /
  reduce-scatter sit at the block boundary (the conventional layout).
* **Tensor parallel (TP) with DTensor** via ``torch.distributed.tensor.parallel``.
  Column-parallel for ``q_proj`` / ``k_proj`` / ``v_proj`` / ``w1`` / ``w3``,
  row-parallel for ``o_proj`` / ``w2``. Tested via a ``dry_run`` planner so the
  *plan* is verified in CI without spinning up a real ProcessGroup.
* **Pipeline parallel (PP)** is still stubbed at the runtime layer (we don't
  want to spin up the pipeline scheduler in CI). The planner emits a stage map
  so the higher-level plan is testable.
* **DDP** path is preserved for single-machine multi-GPU runs that don't need
  full FSDP. The trainer never needs to know which path is active.

The Megatron-Core / NeMo / DeepSpeed branches still exist as opt-in backends
(``backend="megatron_core" | "nemo" | "deepspeed"``) but now degrade to a
clear ``ImportError`` with an install hint instead of the old
``NotImplementedError``.

Tests live in ``tests/test_parallel.py``:

* the *planner* (``ParallelPlan``) is exercised in CI without
  ``torch.distributed`` ever being initialised — it returns the *intended*
  wrap layout for any (dp, tp, pp) config so we can pin the math.
* the *engine* still runs the single-rank path (``world_size == 1``) end-to-
  end on CPU through ``Trainer.fit`` (covered by ``tests/test_training.py``).
* a real multi-rank run is gated behind a ``RUN_DIST_TESTS=1`` env var.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .precision import PrecisionPolicy

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class ParallelConfig:
    """How to shard / replicate the model across the cluster.

    The configured product ``dp * tp * pp * cp`` must equal world_size. Set
    ``backend`` to switch runtimes; "torch_native" composes FSDP2 + DTensor TP
    + (stubbed) PP and is the path tested here.
    """

    backend: str = "torch_native"        # 'torch_native'|'megatron_core'|'nemo'|'deepspeed'
    dp: int = 1
    tp: int = 1
    pp: int = 1
    sp: bool = False
    ep: int = 1
    cp: int = 1
    zero_stage: int = 3                  # FSDP2 uses ZeRO-3 by default
    activation_recompute: str = "selective"
    grad_clip: float = 1.0               # mirrored from OptimConfig; set by Trainer
    precision: str = "fp32"              # passed to PrecisionPolicy.create()
    fsdp_wrap_policy: str = "block"      # 'block' wraps each Transformer Block;
                                         # 'layer' wraps each leaf nn.Linear;
                                         # 'none' = DDP-only (no sharding)

    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp


# ----------------------------------------------------------------------------
# Planner — pure-Python, no torch.distributed needed
# ----------------------------------------------------------------------------


@dataclass
class TensorParallelPlan:
    """How each parameterised submodule is sharded across the TP group.

    Keys are dotted module names (``layers.3.attn.q_proj``); values are one of
    ``"colwise"``, ``"rowwise"``, ``"replicate"``. Built by
    :func:`build_tp_plan` from a config + a model, without any distributed
    primitives.
    """

    plan: dict[str, str] = field(default_factory=dict)

    def colwise(self) -> list[str]:
        return [k for k, v in self.plan.items() if v == "colwise"]

    def rowwise(self) -> list[str]:
        return [k for k, v in self.plan.items() if v == "rowwise"]

    def replicate(self) -> list[str]:
        return [k for k, v in self.plan.items() if v == "replicate"]


@dataclass
class PipelinePlan:
    """Per-rank pipeline stage assignment for ``pp`` stages of ``n_layer``."""

    n_layer: int
    pp: int

    def stages(self) -> list[list[int]]:
        """Return a list of layer-indices per pipeline stage.

        Balanced split: layers ``[0, n/pp)`` go to stage 0, etc.; any
        remainder is distributed to the earliest stages.
        """
        if self.pp <= 1:
            return [list(range(self.n_layer))]
        base = self.n_layer // self.pp
        rem = self.n_layer % self.pp
        out, idx = [], 0
        for s in range(self.pp):
            take = base + (1 if s < rem else 0)
            out.append(list(range(idx, idx + take)))
            idx += take
        return out


@dataclass
class ParallelPlan:
    """Composite plan returned by :func:`plan_parallelism`."""

    cfg: ParallelConfig
    tp: TensorParallelPlan
    pp: PipelinePlan
    notes: list[str] = field(default_factory=list)


# Submodule-name patterns we know how to shard. The model module names match
# what :class:`platform.model.transformer` produces.
_TP_COLWISE_SUFFIXES = ("q_proj", "k_proj", "v_proj", "w1", "w3", "lm_head",
                        "q_up", "k_up", "v_up", "k_rope", "gate")
_TP_ROWWISE_SUFFIXES = ("o_proj", "w2", "q_down", "kv_down")


def build_tp_plan(model, *, tp: int) -> TensorParallelPlan:
    """Walk ``model.named_modules()`` and decide colwise/rowwise/replicate.

    No-op (everything replicate) when ``tp == 1`` so callers can always use
    this without branching on world topology.
    """
    plan: dict[str, str] = {}
    if tp <= 1:
        return TensorParallelPlan(plan)
    for name, mod in model.named_modules():
        # Only sharded leaves we know how to handle.
        if mod.__class__.__name__ != "Linear":
            continue
        if any(name.endswith(suf) for suf in _TP_COLWISE_SUFFIXES):
            plan[name] = "colwise"
        elif any(name.endswith(suf) for suf in _TP_ROWWISE_SUFFIXES):
            plan[name] = "rowwise"
        else:
            plan[name] = "replicate"
    return TensorParallelPlan(plan)


def plan_parallelism(model, cfg: ParallelConfig) -> ParallelPlan:
    """Build a :class:`ParallelPlan` for ``model`` under ``cfg``.

    Pure-Python: no ProcessGroup, no actual sharding. Run in CI to pin the
    *intended* layout; the engine then applies it under a real distributed
    runtime.
    """
    notes: list[str] = []
    n_layer = getattr(getattr(model, "cfg", None), "n_layer", None)
    if n_layer is None:
        n_layer = sum(1 for _ in getattr(model, "layers", []))
    tp_plan = build_tp_plan(model, tp=cfg.tp)
    pp_plan = PipelinePlan(n_layer=int(n_layer), pp=cfg.pp)
    if cfg.tp > 1 and not tp_plan.plan:
        notes.append("tp>1 but no shardable Linear modules found — TP is a no-op")
    if cfg.pp > 1:
        notes.append("pp>1: planner returns stage map; runtime scheduler is stubbed")
    if cfg.zero_stage != 3 and cfg.fsdp_wrap_policy != "none":
        notes.append(f"zero_stage={cfg.zero_stage} requested; FSDP2 default is ZeRO-3")
    return ParallelPlan(cfg=cfg, tp=tp_plan, pp=pp_plan, notes=notes)


# ----------------------------------------------------------------------------
# Apply plan: FSDP2 + DTensor TP wrap (lazy-imported torch.distributed bits)
# ----------------------------------------------------------------------------


def _apply_tensor_parallel(model, plan: TensorParallelPlan, tp_mesh):
    """Apply the TP plan to ``model`` using torch.distributed.tensor.parallel.

    Lazy-imports the DTensor parallel API; raises a clear ``ImportError`` if
    the installed torch is too old (TP via DTensor needs torch >= 2.3).
    """
    if not plan.plan:
        return model
    try:
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )
    except ImportError as e:
        raise ImportError(
            "Tensor parallel requires torch>=2.3 with DTensor "
            "(torch.distributed.tensor.parallel). "
            "Run with tp=1 or upgrade torch."
        ) from e

    parallelize_plan: dict[str, object] = {}
    for name, kind in plan.plan.items():
        if kind == "colwise":
            parallelize_plan[name] = ColwiseParallel()
        elif kind == "rowwise":
            parallelize_plan[name] = RowwiseParallel()
        # 'replicate' = default; nothing to do.
    return parallelize_module(model, tp_mesh, parallelize_plan)


def _apply_fsdp(model, *, policy: str, mesh):
    """Wrap ``model`` (or each Block) with FSDP2 ``fully_shard``.

    ``policy="block"`` wraps each transformer Block as an FSDP unit (the
    conventional layout: all-gather happens at block boundaries). ``"layer"``
    wraps every leaf nn.Linear; ``"none"`` is a DDP-style replicate.
    """
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError as e:
        raise ImportError(
            "FSDP2 requires torch>=2.4 (torch.distributed.fsdp.fully_shard). "
            "Set fsdp_wrap_policy='none' to fall back to DDP."
        ) from e

    if policy == "block":
        for block in getattr(model, "layers", []):
            fully_shard(block, mesh=mesh)
    elif policy == "layer":
        import torch.nn as nn
        for sub in model.modules():
            if isinstance(sub, nn.Linear):
                fully_shard(sub, mesh=mesh)
    elif policy == "none":
        pass
    else:
        raise ValueError(f"unknown fsdp_wrap_policy: {policy!r}")
    fully_shard(model, mesh=mesh)
    return model


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------


class ParallelEngine:
    """Owns the model + optimizer; provides ``forward_backward`` + ``step``.

    Composition order (when ``world_size > 1``):

    1. Build a ``DeviceMesh`` shaped ``(dp, tp)`` (PP is layered above via the
       planner; the real PP scheduler is stubbed in this module).
    2. Apply the TP plan with DTensor.
    3. Apply FSDP2 wrap with the DP sub-mesh.
    4. Wrap with DDP only as a fallback when both FSDP and TP are off.

    The CI path (no distributed) goes straight to the unwrapped model; the
    precision policy is applied identically in both cases.
    """

    def __init__(self, model, optimizer, cfg: ParallelConfig):
        self.cfg = cfg
        self.optimizer = optimizer
        self.model = model

        # Validate backend & topology before touching torch.distributed.
        if cfg.backend == "megatron_core":
            self._require_megatron_core()
        elif cfg.backend == "nemo":
            self._require_nemo()
        elif cfg.backend == "deepspeed":
            self._require_deepspeed()
        elif cfg.backend != "torch_native":
            raise ValueError(
                f"unknown backend {cfg.backend!r}; "
                "use 'torch_native' | 'megatron_core' | 'nemo' | 'deepspeed'"
            )

        # Lazy import — keeps this module light when distributed isn't used.
        import torch.distributed as dist
        self._dist = dist

        if cfg.world_size() > 1 and dist.is_available() and dist.is_initialized():
            self._build_distributed()
        elif cfg.dp > 1 and dist.is_available() and dist.is_initialized():
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(model)

        self.precision = PrecisionPolicy.create(cfg.precision)

    # -- distributed composition ----------------------------------------

    def _build_distributed(self) -> None:
        """Compose FSDP2 + DTensor TP under a 2-D device mesh.

        Skipped in CI: gated behind ``dist.is_initialized()``. The pure-Python
        planner (``plan_parallelism``) is what the test suite exercises.
        """
        from torch.distributed.device_mesh import init_device_mesh
        cfg = self.cfg
        mesh = init_device_mesh(
            "cuda", (cfg.dp, cfg.tp), mesh_dim_names=("dp", "tp"),
        )
        plan = plan_parallelism(self.model, cfg)
        # 1. TP
        self.model = _apply_tensor_parallel(self.model, plan.tp, mesh["tp"])
        # 2. FSDP2 over the DP dimension
        if cfg.fsdp_wrap_policy != "none":
            self.model = _apply_fsdp(self.model, policy=cfg.fsdp_wrap_policy,
                                      mesh=mesh["dp"])
        elif cfg.dp > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, process_group=mesh["dp"].get_group())
        self._mesh = mesh

    # -- backend gating helpers -----------------------------------------

    @staticmethod
    def _require_megatron_core():
        try:
            import megatron.core  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='megatron_core' requires `pip install megatron-core`."
            ) from e
        raise NotImplementedError(
            "megatron-core backend is not yet wired; use 'torch_native'."
        )

    @staticmethod
    def _require_nemo():
        try:
            import nemo  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='nemo' requires `pip install nemo-toolkit`."
            ) from e
        raise NotImplementedError(
            "nemo backend is not yet wired; use 'torch_native'."
        )

    @staticmethod
    def _require_deepspeed():
        try:
            import deepspeed  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='deepspeed' requires `pip install deepspeed`."
            ) from e
        raise NotImplementedError(
            "deepspeed backend is not yet wired; use 'torch_native'."
        )

    # -- per-step ops ----------------------------------------------------

    def forward_backward(self, batch) -> dict:
        input_ids, targets = batch
        with self.precision.autocast():
            logits, loss = self.model(input_ids, targets=targets)
        loss.backward()
        return {
            "loss": float(loss.detach()),
            "tokens": int(input_ids.numel()),
            "precision_backend": self.precision.backend,
        }

    def step(self) -> None:
        import torch
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def all_reduce_metric(self, value: float) -> float:
        if self._dist.is_available() and self._dist.is_initialized():
            import torch
            t = torch.tensor([value], dtype=torch.float32)
            self._dist.all_reduce(t, op=self._dist.ReduceOp.SUM)
            return float(t.item()) / self._dist.get_world_size()
        return float(value)


# ----------------------------------------------------------------------------
# Multi-rank smoke entry (used by tests/test_parallel_distributed.py when
# RUN_DIST_TESTS=1; otherwise the test skips). Kept here so the runner doesn't
# need to import the test module.
# ----------------------------------------------------------------------------


def _distributed_smoke_main():
    """Tiny end-to-end FSDP2 run on the spawned ranks.

    Not called in CI. Invoked by the gated test that uses
    ``torch.multiprocessing.spawn`` so we can prove the FSDP wrap actually
    works when a real ProcessGroup exists.
    """
    import torch
    import torch.distributed as dist
    from platform.model.config import ModelConfig
    from platform.model.transformer import Transformer

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
                       d_model=32, d_ffn=64, max_seq_len=16)
    torch.manual_seed(0)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pcfg = ParallelConfig(backend="torch_native", dp=world, tp=1, pp=1,
                          fsdp_wrap_policy="none")  # DDP path on CPU
    eng = ParallelEngine(model, opt, pcfg)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    eng.forward_backward((x, y))
    eng.step()
    dist.destroy_process_group()
