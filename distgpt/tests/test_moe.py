"""Sparse MoE FFN tests (Tier 5.12).

The MoE block replaces SwiGLU when ``moe_num_experts > 1``. Risk surface:

  * Routing: every token must hit exactly ``top_k`` experts; shared experts
    fire on every token regardless of routing.
  * Aux loss: train-mode forward stashes a finite, non-zero z-loss on each
    block (and a balance term when ``moe_balance == "aux_loss"``); ``GPT``
    sums those into the main loss so the trainer doesn't need MoE awareness.
  * Bias-based balancer: under ``aux_free`` the routing bias is nudged each
    training step so an initially-imbalanced router converges toward uniform.
  * Both dispatch backends (``"loop"``, ``"batched"``) produce the same
    forward output up to floating-point accumulation order.
  * Muon split: the router gate is IO-shaped 2D but routes to AdamW; the
    expert MLPs are 2D hidden weights and route to Muon.
  * Integrations: trainer runs, TP+MoE refuses (NotImplementedError), HF
    export refuses (NotImplementedError).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT, MoEFFN, SwiGLU  # noqa: E402
from distgpt.training.muon import split_muon_params  # noqa: E402


def _moe_cfg(**over):
    base = dict(
        vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
        d_model=32, d_ffn=64, max_seq_len=16,
        moe_num_experts=4, moe_top_k=2, moe_shared_experts=1,
        moe_expert_d_ffn=32,
    )
    base.update(over)
    return ModelConfig(**base)


# ---------- block-level: shape + module type ----------


def test_moe_block_replaces_swiglu_when_enabled():
    """``moe_num_experts > 1`` must swap SwiGLU for MoEFFN in every block,
    and the dense (default) config keeps SwiGLU. Pins the construction path
    in Block.__init__ so a regression to dense-only is loud."""
    dense_cfg = ModelConfig(
        vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
        d_model=32, d_ffn=64, max_seq_len=16,
    )
    m_dense = GPT(dense_cfg)
    assert all(isinstance(blk.ffn, SwiGLU) for blk in m_dense.layers)
    m_moe = GPT(_moe_cfg())
    assert all(isinstance(blk.ffn, MoEFFN) for blk in m_moe.layers)


def test_moe_forward_shape_matches_dense():
    """MoE forward must return the same ``[B, T, d_model]`` shape as the
    dense path so downstream norm + lm_head don't care which FFN ran."""
    cfg = _moe_cfg()
    m = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = m(x, y)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(loss)


# ---------- routing contract ----------


def test_moe_top_k_routing_selects_exactly_k_experts_per_token():
    """Each token's routing table is ``[N, k]`` of expert ids in [0, E). The
    ``last_expert_counts`` buffer accumulates ``top_k`` slots per token (so
    the total equals N * top_k)."""
    cfg = _moe_cfg(moe_num_experts=6, moe_top_k=2)
    m = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    m(x, x)
    N = 2 * 8
    blk = m.layers[0]
    assert isinstance(blk.ffn, MoEFFN)
    counts = blk.ffn.last_expert_counts
    assert counts.shape == (cfg.moe_num_experts,)
    assert int(counts.sum().item()) == N * cfg.moe_top_k


def test_moe_shared_expert_runs_on_every_token():
    """A shared expert is an always-on SwiGLU added to every token's output.
    Disabling routing (zero out all routed experts' down-proj) leaves only
    the shared expert's contribution — that residual must be non-zero, which
    proves the shared path runs even when no router weight reaches it."""
    cfg = _moe_cfg(moe_num_experts=4, moe_top_k=2, moe_shared_experts=1)
    m = GPT(cfg).eval()
    blk0 = m.layers[0]
    assert isinstance(blk0.ffn, MoEFFN)
    moe = blk0.ffn
    # Zero out every routed expert's w2 (their FFN contribution becomes 0).
    with torch.no_grad():
        for e in moe.experts:
            e.w2.weight.zero_()
    x = torch.randn(2, 4, cfg.d_model)
    y = moe(x)
    # Shared expert is the only remaining contributor; if it weren't running
    # we'd get exact zeros back from the MoE block.
    assert y.abs().sum().item() > 0.0


# ---------- aux loss ----------


def test_moe_aux_loss_is_finite_and_nonzero_in_train_mode():
    """Train-mode forward must populate ``last_aux_loss`` with a finite,
    positive scalar on each MoEFFN block (z-loss > 0 by construction), and
    ``GPT.forward`` must add it into the main loss — so the loss with MoE
    is strictly larger than the bare CE on the same logits."""
    torch.manual_seed(0)
    cfg = _moe_cfg(moe_aux_loss_weight=1.0)
    m = GPT(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, total_loss = m(x, x)
    bare_ce = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), x.reshape(-1),
    )
    aux = sum(blk.ffn.last_aux_loss.float().item() for blk in m.layers
               if isinstance(blk.ffn, MoEFFN))
    assert aux > 0.0 and np.isfinite(aux)
    assert total_loss.item() > bare_ce.item() - 1e-6
    # And the addition is by ``moe_aux_loss_weight`` — sanity check at w=1.
    assert abs((total_loss.item() - bare_ce.item()) - aux) < 1e-3


def test_moe_aux_free_balances_router_bias_over_steps():
    """Aux-free balancer must move the routing bias against the load.

    We synthesise an imbalanced router by setting most gate weight at one
    expert, run a handful of training-mode forwards, and check the bias on
    the over-loaded expert went DOWN while at least one under-loaded
    expert's bias went UP. This is the only behavioural contract we get
    from a stochastic bias update — the absolute magnitudes are tied to
    ``bias_update_speed`` which the test sets aggressively."""
    torch.manual_seed(0)
    cfg = _moe_cfg(moe_balance="aux_free", moe_bias_update_speed=0.5,
                    moe_num_experts=4, moe_top_k=1, moe_shared_experts=0)
    m = GPT(cfg).train()
    # Force expert 0 to win the topk in early steps by zeroing the other
    # rows of every block's gate.
    with torch.no_grad():
        for blk in m.layers:
            blk.ffn.gate.weight.zero_()
            blk.ffn.gate.weight[0] = 1.0
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    for _ in range(8):
        m(x, x)
    bias = m.layers[0].ffn.routing_bias.detach().clone()
    # Expert 0 saw all the load → its bias should be the smallest.
    assert int(bias.argmin().item()) == 0, f"bias={bias.tolist()}"
    # And at least one other expert's bias should have risen above zero.
    assert (bias[1:] > 0).any(), f"bias={bias.tolist()}"


def test_moe_aux_loss_balance_mode_adds_load_balance_term():
    """In ``aux_loss`` mode the aux is z_loss + lb_loss; in ``aux_free``
    mode it is z_loss alone. The lb_loss is bounded below by ~1/E (uniform
    routing minimum), so the aux_loss-mode value must be measurably larger
    than the aux_free value on the same inputs and weights."""
    torch.manual_seed(0)
    cfg_free = _moe_cfg(moe_balance="aux_free")
    cfg_loss = _moe_cfg(moe_balance="aux_loss")
    m_free = GPT(cfg_free).train()
    m_loss = GPT(cfg_loss).train()
    # Copy weights so the only difference is the balance mode.
    m_loss.load_state_dict(m_free.state_dict())
    x = torch.randint(0, cfg_free.vocab_size, (2, 8))
    m_free(x, x)
    m_loss(x, x)
    a_free = m_free.layers[0].ffn.last_aux_loss.item()
    a_loss = m_loss.layers[0].ffn.last_aux_loss.item()
    assert a_loss > a_free + 1e-6, (a_loss, a_free)


# ---------- dispatch backends ----------


def test_moe_dispatch_loop_and_batched_match():
    """Both backends are mathematically the same up to FP accumulation
    order. With deterministic-ish inputs the outputs must match to ~1e-5."""
    torch.manual_seed(0)
    cfg_b = _moe_cfg(moe_dispatch="batched")
    cfg_l = _moe_cfg(moe_dispatch="loop")
    m_b = GPT(cfg_b).eval()
    m_l = GPT(cfg_l).eval()
    m_l.load_state_dict(m_b.state_dict())
    x = torch.randint(0, cfg_b.vocab_size, (2, 8))
    with torch.no_grad():
        y_b, _ = m_b(x, x)
        y_l, _ = m_l(x, x)
    assert torch.allclose(y_b, y_l, atol=1e-4, rtol=1e-4)


def test_moe_dispatch_rejects_unknown_backend():
    cfg = _moe_cfg(moe_dispatch="nope")
    with pytest.raises(ValueError, match="moe_dispatch"):
        GPT(cfg)


# ---------- Muon split ----------


def test_split_muon_params_routes_moe_gate_and_bias_to_adamw():
    """The MoE router gate is an IO-like 2D matrix and ``routing_bias`` is
    a small buffer/param; both must land in the AdamW group, not Muon. The
    expert MLPs (w1/w2/w3) are 2D hidden weights and route to Muon."""
    cfg = _moe_cfg()
    m = GPT(cfg)
    muon, adamw = split_muon_params(m)
    name_for = {id(p): n for n, p in m.named_parameters()}
    muon_names = {name_for[id(p)] for p in muon}
    adamw_names = {name_for[id(p)] for p in adamw}
    # Router gate → AdamW.
    assert any("ffn.gate.weight" in n for n in adamw_names)
    assert not any("ffn.gate.weight" in n for n in muon_names)
    # Expert MLPs → Muon.
    assert any("ffn.experts.0.w1.weight" in n for n in muon_names)
    assert any("ffn.experts.0.w2.weight" in n for n in muon_names)
    assert any("ffn.experts.0.w3.weight" in n for n in muon_names)
    # Shared expert MLPs → Muon (still hidden 2D weights, not IO).
    assert any("ffn.shared.0.w1.weight" in n for n in muon_names)


# ---------- integrations ----------


def test_trainer_runs_with_moe_enabled(tmp_path: Path):
    """3-step smoke: trainer must run end-to-end with MoE on and emit a
    JSONL log with finite losses — proves the MoE aux-loss integration
    composes with the optimizer + grad-clip path."""
    import json
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 64, size=8_000, dtype=np.uint16).tofile(
        str(data_dir / "shard_0.bin")
    )
    cfg = {
        "run_id": "smoke_moe",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 64, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
            "moe_num_experts": 4, "moe_top_k": 2,
            "moe_shared_experts": 1, "moe_expert_d_ffn": 32,
        },
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 3,
        },
        "train": {"micro_batch": 2, "grad_accum": 1,
                    "log_every": 1, "eval_every": 99, "ckpt_every": 99},
    }
    train(cfg)
    log = (Path(cfg["out_dir"]) / "log.jsonl").read_text().splitlines()
    losses = [json.loads(line)["loss"] for line in log if '"loss"' in line]
    assert losses and all(np.isfinite(l) for l in losses)


def test_moe_plus_tp_raises_not_implemented():
    """Tier-6 guard: MoE + tensor parallelism (tp > 1) is not yet supported
    by apply_tp. We pin the NotImplementedError so when expert-parallel
    lands the test fails loudly and someone removes this raise."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")
    if dist.is_initialized():
        pytest.skip("dist already initialized in another test")
    # We don't actually need a process group — apply_tp checks tp_mesh.size()
    # first, then the cfg.moe_enabled. Build a fake mesh-like object.
    class _FakeMesh:
        def size(self): return 2
    cfg = _moe_cfg()
    m = GPT(cfg)
    from distgpt.parallel.tensor import apply_tp
    with pytest.raises(NotImplementedError, match="MoE"):
        apply_tp(m, _FakeMesh())


def test_export_to_hf_rejects_moe(tmp_path: Path):
    """HF export currently has no Mixtral/DeepSeek mapping; MoE must raise
    a NotImplementedError rather than silently strip the experts (which
    would produce a one-expert-only ghost model)."""
    from distgpt.eval.export_hf import export_to_hf
    cfg = _moe_cfg()
    m = GPT(cfg)
    with pytest.raises(NotImplementedError, match="MoE"):
        export_to_hf(m, cfg, tmp_path / "should_not_exist")
