"""Tests for the parallelism planner + engine wrap.

The *planner* (pure-Python, no torch.distributed) is exercised in CI to pin
the intended sharding layout for every (dp, tp, pp) config. The *engine*'s
single-rank path (world_size==1) is exercised via the existing trainer
tests; here we add construction tests for the various opt-in backends to
prove the error messages are actionable instead of the old
``NotImplementedError`` swallow.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.training.parallel import (
    ParallelConfig,
    ParallelEngine,
    ParallelPlan,
    PipelinePlan,
    TensorParallelPlan,
    build_tp_plan,
    plan_parallelism,
)


def _mk_model(*, mla: bool = False, moe: bool = False) -> Transformer:
    cfg = ModelConfig(
        vocab_size=64, n_layer=4, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=32,
        attn_kind="mla" if mla else "gqa",
        mla_kv_latent_dim=32, mla_rope_head_dim=8,
        moe_num_experts=4 if moe else 0,
        moe_top_k=2,
        moe_expert_d_ffn=64,
        moe_shared_experts=1 if moe else 0,
    )
    torch.manual_seed(0)
    return Transformer(cfg)


# ----- Planner: TP plan ------------------------------------------------------


def test_build_tp_plan_is_noop_when_tp_is_one():
    model = _mk_model()
    plan = build_tp_plan(model, tp=1)
    assert plan.plan == {}
    assert plan.colwise() == []
    assert plan.rowwise() == []


def test_build_tp_plan_assigns_colwise_and_rowwise_for_gqa():
    model = _mk_model()
    plan = build_tp_plan(model, tp=4)
    # Q/K/V projections are col-parallel; output projection is row-parallel.
    assert "layers.0.attn.q_proj" in plan.colwise()
    assert "layers.0.attn.k_proj" in plan.colwise()
    assert "layers.0.attn.v_proj" in plan.colwise()
    assert "layers.0.attn.o_proj" in plan.rowwise()
    # FFN: w1/w3 col-parallel; w2 row-parallel.
    assert "layers.0.ffn.w1" in plan.colwise()
    assert "layers.0.ffn.w3" in plan.colwise()
    assert "layers.0.ffn.w2" in plan.rowwise()
    # lm_head is col-parallel by convention (vocab is the sharded dim).
    assert "lm_head" in plan.colwise()


def test_build_tp_plan_handles_mla_specifics():
    """MLA has up/down projections; ensure they get the right colwise/rowwise."""
    model = _mk_model(mla=True)
    plan = build_tp_plan(model, tp=2)
    # Up-projections fan out hidden -> shard along output dim (colwise).
    assert "layers.0.attn.q_up" in plan.colwise()
    assert "layers.0.attn.k_up" in plan.colwise()
    assert "layers.0.attn.v_up" in plan.colwise()
    # Down-projections collapse hidden -> shard along input dim (rowwise).
    assert "layers.0.attn.q_down" in plan.rowwise()
    assert "layers.0.attn.kv_down" in plan.rowwise()


def test_build_tp_plan_handles_moe_routing_and_experts():
    """MoE: gate and per-expert w1/w3 are col-parallel; w2 is row-parallel."""
    model = _mk_model(moe=True)
    plan = build_tp_plan(model, tp=2)
    cs = plan.colwise()
    rs = plan.rowwise()
    assert "layers.0.ffn.gate" in cs
    assert any(n.startswith("layers.0.ffn.experts.0") and n.endswith("w1") for n in cs)
    assert any(n.startswith("layers.0.ffn.experts.0") and n.endswith("w2") for n in rs)
    # Shared expert too.
    assert "layers.0.ffn.shared.0.w1" in cs
    assert "layers.0.ffn.shared.0.w2" in rs


# ----- Planner: pipeline plan -----------------------------------------------


def test_pipeline_plan_single_stage_returns_all_layers():
    p = PipelinePlan(n_layer=12, pp=1)
    assert p.stages() == [list(range(12))]


def test_pipeline_plan_balanced_split():
    p = PipelinePlan(n_layer=8, pp=4)
    assert p.stages() == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_pipeline_plan_distributes_remainder_to_earliest_stages():
    p = PipelinePlan(n_layer=10, pp=3)
    # 10 / 3 = 3 base, remainder 1 → first stage gets the extra.
    assert p.stages() == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]


# ----- plan_parallelism composite -------------------------------------------


def test_plan_parallelism_combines_tp_and_pp():
    model = _mk_model()
    cfg = ParallelConfig(dp=2, tp=4, pp=2)
    plan = plan_parallelism(model, cfg)
    assert isinstance(plan, ParallelPlan)
    assert isinstance(plan.tp, TensorParallelPlan)
    assert isinstance(plan.pp, PipelinePlan)
    assert plan.cfg.world_size() == 2 * 4 * 2 * 1
    # The TP plan should be populated (model has shardable Linears).
    assert plan.tp.colwise(), "expected colwise shards"
    # The PP plan splits the model's 4 layers across 2 stages.
    assert plan.pp.stages() == [[0, 1], [2, 3]]


def test_plan_parallelism_warns_on_pp_runtime_stub():
    """PP planner emits a note that the runtime scheduler is stubbed."""
    model = _mk_model()
    plan = plan_parallelism(model, ParallelConfig(pp=2))
    assert any("pp>1" in n for n in plan.notes)


def test_plan_parallelism_handles_models_without_shardable_modules():
    """A trivial model without any nn.Linear shouldn't crash; the planner
    returns an empty TP plan and a note."""
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = type("c", (), {"n_layer": 1})()
            self.layers = torch.nn.ModuleList([torch.nn.ReLU()])

    plan = plan_parallelism(_Tiny(), ParallelConfig(tp=2))
    assert plan.tp.plan == {}
    assert any("no shardable" in n for n in plan.notes)


# ----- Engine construction --------------------------------------------------


def test_engine_single_rank_construction_is_unchanged():
    """The single-rank path still works on CPU end-to-end."""
    model = _mk_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eng = ParallelEngine(model, opt, ParallelConfig())
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    metrics = eng.forward_backward((x, y))
    eng.step()
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["tokens"] == 32


def test_engine_unknown_backend_raises_clear_value_error():
    model = _mk_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError) as e:
        ParallelEngine(model, opt, ParallelConfig(backend="quantum_ddp"))
    assert "quantum_ddp" in str(e.value)


@pytest.mark.parametrize("backend", ["megatron_core", "nemo", "deepspeed"])
def test_engine_opt_in_backends_raise_clear_importerror(monkeypatch, backend):
    """Without the optional dep installed, the engine raises a clear ImportError
    (not the old NotImplementedError that hid the real cause)."""
    for mod in ("megatron", "megatron.core", "nemo", "deepspeed"):
        monkeypatch.setitem(sys.modules, mod, None)
    model = _mk_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ImportError) as e:
        ParallelEngine(model, opt, ParallelConfig(backend=backend))
    assert backend.replace("_core", "-core").split("_")[0] in str(e.value).lower() \
           or backend in str(e.value)


def test_engine_world_size_one_does_not_touch_distributed():
    """When dp=tp=pp=cp=1 the engine must not call into torch.distributed."""
    model = _mk_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eng = ParallelEngine(model, opt, ParallelConfig())
    # _mesh attribute is only set on the distributed path.
    assert not hasattr(eng, "_mesh")


# ----- Multi-rank: gated behind RUN_DIST_TESTS=1 ----------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_DIST_TESTS") != "1",
    reason="distributed smoke gated behind RUN_DIST_TESTS=1",
)
def test_distributed_ddp_smoke_two_ranks():
    """Spawn 2 ranks on gloo and run one forward/backward through the engine.

    This proves the FSDP/DDP wrap path is wired correctly when a real
    ProcessGroup exists. Gated because it spawns processes and takes ~5s."""
    import torch.multiprocessing as mp
    from platform.training.parallel import _distributed_smoke_main

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    def _runner(rank, world):
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        _distributed_smoke_main()

    mp.spawn(_runner, args=(2,), nprocs=2, join=True)
