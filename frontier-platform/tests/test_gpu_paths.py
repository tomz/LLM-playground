"""Additional GPU smoke tests covering Tier 2 + Tier 3 paths on CUDA.

Each test is small (~5s on a 3050) and asserts only that the code path
executes without NaN — not convergence quality, which is the CPU suite's job.

For DPO/PPO we exercise the inner math (compute_logps, dpo_loss, rollout,
ppo_step) directly rather than the run_* wrappers, since those wrappers
currently load/save via CPU. The math is what we want on the GPU.
"""
from __future__ import annotations
import asyncio
import math

import pytest
import torch

from platform.alignment._common import clone_for_reference, compute_logps
from platform.alignment.dpo import dpo_loss
from platform.alignment.ppo import PPOConfig, ValueHead, ppo_step, rollout
from platform.alignment.reward_model import RewardModel
from platform.model.config import ModelConfig
from platform.model.transformer import MoEFFN, Transformer
from platform.serving.engine import Engine, EngineConfig, GenRequest
from platform.tokenizer.bytes import BytesTokenizer
from platform.training.checkpoint import CheckpointManager
from platform.training.optim import OptimConfig, build_optimizer
from platform.training.parallel import ParallelConfig, ParallelEngine


def _tiny_cfg(vocab: int = 512) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
    )


def _async_collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


# ---------- 1. Serving / TorchEngine on CUDA ----------

@pytest.mark.gpu
def test_torch_engine_on_cuda(gpu_or_skip):
    device = gpu_or_skip
    torch.manual_seed(0)
    model = Transformer(_tiny_cfg(vocab=64)).to(device)
    eng = Engine(EngineConfig(backend="torch"), model=model)
    req = GenRequest(prompt_ids=[1, 2, 3], max_new_tokens=8, temperature=0.0)
    chunks = _async_collect(eng.generate(req))
    assert chunks[-1]["done"] is True
    assert chunks[-1]["usage"]["completion_tokens"] == 8
    tokens = [c["token_id"] for c in chunks if not c.get("done")]
    assert len(tokens) == 8
    assert all(0 <= t < 64 for t in tokens)


# ---------- 2. Checkpoint roundtrip GPU → CPU ----------

@pytest.mark.gpu
def test_checkpoint_roundtrip_cuda(tmp_path, gpu_or_skip):
    device = gpu_or_skip
    cfg = _tiny_cfg(vocab=128)
    torch.manual_seed(0)
    model = Transformer(cfg).to(device)
    opt, _ = build_optimizer(model, OptimConfig(peak_lr=1e-3))
    pcfg = ParallelConfig(); pcfg.grad_clip = 1.0
    eng = ParallelEngine(model, opt, pcfg)

    x = torch.randint(0, 128, (2, 16), device=device)
    y = torch.randint(0, 128, (2, 16), device=device)
    for _ in range(10):
        eng.forward_backward((x, y))
        eng.step()

    mgr = CheckpointManager(str(tmp_path), "gpu_ckpt")
    mgr.save_async(eng, None, step=10)

    with torch.no_grad():
        _, loss_gpu = model(x, targets=y)

    # Reload on CPU and verify same loss within numerical tolerance.
    torch.manual_seed(0)
    model_cpu = Transformer(cfg)
    opt_cpu, _ = build_optimizer(model_cpu, OptimConfig(peak_lr=1e-3))
    eng_cpu = ParallelEngine(model_cpu, opt_cpu, pcfg)
    mgr.load_into(eng_cpu, None, step="latest")
    with torch.no_grad():
        _, loss_cpu = model_cpu(x.cpu(), targets=y.cpu())
    assert abs(float(loss_gpu) - float(loss_cpu)) < 1e-3, (float(loss_gpu), float(loss_cpu))


# ---------- 3. MoE on CUDA ----------

@pytest.mark.gpu
def test_moe_on_cuda(gpu_or_skip):
    device = gpu_or_skip
    cfg = ModelConfig(
        vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
        moe_num_experts=4, moe_top_k=2,
    )
    torch.manual_seed(0)
    m = Transformer(cfg).to(device)
    assert isinstance(m.layers[0].ffn, MoEFFN)
    x = torch.randint(0, 256, (2, 16), device=device)
    y = torch.randint(0, 256, (2, 16), device=device)
    _, loss = m(x, targets=y)
    aux = m.layers[0].ffn.last_aux_loss.detach()
    assert torch.isfinite(loss)
    assert float(aux) > 0.0


# ---------- 4. DPO loss + backward on CUDA ----------

@pytest.mark.gpu
def test_dpo_step_on_cuda(gpu_or_skip):
    device = gpu_or_skip
    torch.manual_seed(0)
    cfg = _tiny_cfg(vocab=64)
    policy = Transformer(cfg).to(device)
    ref = clone_for_reference(policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    # 4 (chosen, rejected) pairs.
    B, T = 4, 16
    x_c = torch.randint(0, 64, (B, T), device=device)
    y_c = torch.randint(0, 64, (B, T), device=device)
    x_r = torch.randint(0, 64, (B, T), device=device)
    y_r = torch.randint(0, 64, (B, T), device=device)
    mask = torch.ones(B, T, device=device)

    losses = []
    for step in range(5):
        opt.zero_grad()
        p_c = compute_logps(policy, x_c, y_c, mask)
        p_r = compute_logps(policy, x_r, y_r, mask)
        with torch.no_grad():
            r_c = compute_logps(ref, x_c, y_c, mask)
            r_r = compute_logps(ref, x_r, y_r, mask)
        loss = dpo_loss(p_c, p_r, r_c, r_r, beta=0.1, variant="sigmoid")
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(x) for x in losses)


# ---------- 5. PPO rollout + step on CUDA ----------

@pytest.mark.gpu
def test_ppo_step_on_cuda(gpu_or_skip):
    device = gpu_or_skip
    torch.manual_seed(0)
    tok = BytesTokenizer()
    cfg = _tiny_cfg(vocab=tok.vocab_size)
    policy = Transformer(cfg).to(device)
    trunk = Transformer(cfg).to(device)
    rm = RewardModel(trunk, pad_id=tok.pad_id).to(device)
    vh = ValueHead(policy.cfg.d_model).to(device)
    opt = torch.optim.AdamW(list(policy.parameters()) + list(vh.parameters()), lr=1e-4)

    prompts = [tok.encode("Q: a"), tok.encode("Q: b")]
    pcfg = PPOConfig(
        policy_ckpt="", rm_ckpt="", rollout_batch=2,
        max_new_tokens=4, seq_len=32, ppo_epochs=1,
    )
    traj = rollout(policy, prompts, pcfg, value_head=vh, rm=rm, tokenizer=tok)
    for k in ("old_logps", "ref_logps", "values", "advantages", "returns"):
        assert torch.isfinite(traj[k]).all(), k
        assert traj[k].device.type == "cuda", (k, traj[k].device)
    metrics = ppo_step(policy, vh, traj, pcfg, optimizer=opt)
    for k in ("kl", "loss", "loss_pi", "loss_v"):
        assert math.isfinite(metrics[k]), (k, metrics[k])
