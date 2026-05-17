"""End-to-end smoke pipeline: data → tokenize → pretrain → SFT → RM → DPO → PPO → eval → gen.

Runs entirely on CPU in float32. Prints ``=== SMOKE PIPELINE PASS ===`` on success.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The local ``platform`` package shadows the stdlib module. Evict any cached
# stdlib-platform import so our package wins, then let our package's
# __getattr__ proxy take care of any stdlib calls the rest of the stack makes.
for _m in [k for k in list(sys.modules) if k == "platform" or k.startswith("platform.")]:
    del sys.modules[_m]

import asyncio
import json
import math
import shutil
import time

import numpy as np
import torch

# Force CPU and float32 throughout.
torch.set_default_dtype(torch.float32)


def banner(msg: str) -> None:
    print(f"\n--- {msg} ---", flush=True)


def main() -> int:
    t_start = time.time()
    work = ROOT / "out" / "smoke"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # ---- 1. synthetic corpus ---------------------------------------------
    banner("[1] write synthetic corpus")
    from platform.data.synthetic import write_corpus

    corpus_dir = write_corpus(work / "corpus", n_files=20, words_per_file=200, seed=0)
    print(f"corpus: {corpus_dir}")

    # ---- 2. tokenize + shard ---------------------------------------------
    banner("[2] tokenize + shard")
    from platform.data.acquire import LocalFilesSource
    from platform.data.shard import tokenize_and_shard
    from platform.tokenizer.bytes import BytesTokenizer

    tok = BytesTokenizer()
    shard_dir = work / "shards"
    uris = tokenize_and_shard(
        LocalFilesSource(corpus_dir).stream(), tok, shard_dir, domain="synth",
        shard_tokens=8192,
    )
    print(f"wrote {len(uris)} shards")

    # ---- 3. pretrain a tiny Transformer ----------------------------------
    banner("[3] pretrain tiny Transformer")
    from platform.model.config import ModelConfig
    from platform.model.transformer import Transformer
    from platform.training.optim import OptimConfig, build_optimizer
    from platform.training.trainer import Trainer, TrainConfig
    from platform.training.parallel import ParallelConfig

    mcfg = ModelConfig(
        vocab_size=512, n_layer=4, n_head=4, n_kv_head=2,
        d_model=128, d_ffn=256, max_seq_len=64,
    )
    torch.manual_seed(0)
    model = Transformer(mcfg)

    class _ShardLoader:
        """Minimal loader: cycle uint32 shards into (x, y) numpy batches."""

        def __init__(self, shard_paths, batch=4, seq=64):
            self.shards = [np.memmap(p, dtype=np.uint32, mode="r") for p in shard_paths]
            self.batch = batch
            self.seq = seq
            self.s = 0
            self.off = 0

        def __iter__(self):
            need = self.batch * (self.seq + 1)
            while True:
                shard = self.shards[self.s]
                if self.off + need > len(shard):
                    self.s = (self.s + 1) % len(self.shards)
                    self.off = 0
                    continue
                chunk = np.asarray(shard[self.off:self.off + need], dtype=np.int64) \
                    .reshape(self.batch, self.seq + 1)
                self.off += self.batch * self.seq
                yield chunk[:, :-1].copy(), chunk[:, 1:].copy()

        def state_dict(self): return {}
        def load_state_dict(self, _): pass

    shard_paths = sorted(str(p) for p in (shard_dir / "synth").glob("*.bin"))
    eval_shard = shard_paths[-1]
    train_shards = shard_paths[:-1] or shard_paths
    loader = _ShardLoader(train_shards, batch=4, seq=64)

    ocfg = OptimConfig(peak_lr=3e-3, warmup_steps=10, total_steps=100, weight_decay=0.0)
    opt, sched = build_optimizer(model, ocfg)
    tcfg = TrainConfig(
        run_id="smoke", seq_len=64, micro_batch=4, total_tokens=10**12,
        log_every=20, eval_every=0, ckpt_every=0,
        optim=ocfg, parallel=ParallelConfig(),
    )
    tr = Trainer(model, loader, None, None, tcfg, optimizer=opt, scheduler=sched)
    tr.fit()
    h = tr.loss_history
    print(f"pretrain loss curve [first→last]: {h[0]:.3f} → {h[-1]:.3f}")
    assert h[-1] < h[0], "pretrain loss did not decrease"

    base_ckpt = work / "base.pt"
    torch.save({"model": model.state_dict(), "model_cfg": mcfg}, base_ckpt)

    # ---- 4. SFT ----------------------------------------------------------
    banner("[4] SFT")
    from platform.alignment.sft import SFTConfig, run_sft

    sft_jsonl = work / "sft.jsonl"
    with open(sft_jsonl, "w") as f:
        for i in range(10):
            f.write(json.dumps({"prompt": f"Q: item {i}", "response": f"A: ans {i}"}) + "\n")
    eval_jsonl = work / "sft_eval.jsonl"
    with open(eval_jsonl, "w") as f:
        for i in range(2):
            f.write(json.dumps({"prompt": f"Q: probe {i}", "response": f"A: p {i}"}) + "\n")

    sft_out = run_sft(SFTConfig(
        base_ckpt=str(base_ckpt), train_set=str(sft_jsonl), eval_set=str(eval_jsonl),
        out_dir=str(work / "sft"), steps=50, batch_size=4, seq_len=48, lr=3e-3,
    ))
    sft_hist = torch.load(sft_out, map_location="cpu", weights_only=False)["loss_history"]
    print(f"sft loss: {sft_hist[0]:.3f} → {sft_hist[-1]:.3f}")

    # ---- 5. Preference data + reward model -------------------------------
    banner("[5] train reward model")
    from platform.alignment.reward_model import RMConfig, train_reward_model

    pref_jsonl = work / "pref.jsonl"
    with open(pref_jsonl, "w") as f:
        for i in range(10):
            f.write(json.dumps({
                "prompt": f"Q: thing {i}",
                "chosen": f"A: good answer {i}",
                "rejected": f"BAD wrong {i}",
            }) + "\n")
    rm_out = train_reward_model(RMConfig(
        base_ckpt=sft_out, pref_set=str(pref_jsonl),
        out_dir=str(work / "rm"), steps=50, batch_size=4, seq_len=48, lr=1e-3,
    ))
    print(f"rm ckpt: {rm_out}")

    # ---- 6. DPO ----------------------------------------------------------
    banner("[6] DPO")
    from platform.alignment.dpo import DPOConfig, run_dpo

    dpo_out = run_dpo(DPOConfig(
        policy_ckpt=sft_out, pref_set=str(pref_jsonl),
        out_dir=str(work / "dpo"), steps=30, batch_size=4, seq_len=48,
        lr=1e-3, beta=0.1,
    ))
    dpo_hist = torch.load(dpo_out, map_location="cpu", weights_only=False)["history"]
    print(f"dpo logp-gap: {dpo_hist[0]['gap']:.3f} → {dpo_hist[-1]['gap']:.3f}")

    # ---- 7. PPO ----------------------------------------------------------
    banner("[7] PPO rollout + step")
    from platform.alignment.ppo import PPOConfig, run_ppo

    ppo_out = run_ppo(PPOConfig(
        policy_ckpt=sft_out, rm_ckpt=rm_out,
        out_dir=str(work / "ppo"), rollout_batch=2, ppo_epochs=1,
        max_new_tokens=4, seq_len=48, lr=1e-4,
    ), prompts=["Q: hello", "Q: world"])
    ppo_state = torch.load(ppo_out, map_location="cpu", weights_only=False)
    print(f"ppo metrics: {ppo_state['history']}")

    # ---- 8. eval perplexity on held-out shard ----------------------------
    banner("[8] eval perplexity on held-out shard")
    arr = np.memmap(eval_shard, dtype=np.uint32, mode="r")
    seq = 64
    n_seqs = min(8, (len(arr) - 1) // seq)
    chunk = np.asarray(arr[: n_seqs * seq + 1], dtype=np.int64)
    x = torch.from_numpy(chunk[: n_seqs * seq].reshape(n_seqs, seq))
    y = torch.from_numpy(chunk[1: n_seqs * seq + 1].reshape(n_seqs, seq))
    sft_model_state = torch.load(sft_out, map_location="cpu", weights_only=False)
    eval_model = Transformer(sft_model_state["model_cfg"])
    eval_model.load_state_dict(sft_model_state["model"])
    eval_model.eval()
    with torch.no_grad():
        _, loss = eval_model(x, targets=y)
    ppl = math.exp(min(20.0, float(loss)))
    print(f"held-out loss={float(loss):.3f} ppl={ppl:.2f}")

    # ---- 9. generate via TorchEngine -------------------------------------
    banner("[9] generate 30 tokens via TorchEngine")
    from platform.serving.engine import EngineConfig, GenRequest
    from platform.serving.torch_engine import TorchEngine

    ppo_model = Transformer(ppo_state["model_cfg"])
    ppo_model.load_state_dict(ppo_state["model"])
    engine = TorchEngine(EngineConfig(backend="torch", dtype="fp32"),
                         model=ppo_model, tokenizer=tok)
    req = GenRequest(prompt_ids=tok.encode("Q: smoke"), max_new_tokens=30, temperature=0.0)

    async def run_gen():
        out_ids = []
        async for chunk in engine.generate(req):
            if not chunk.get("done"):
                out_ids.append(chunk["token_id"])
        return out_ids

    out_ids = asyncio.run(run_gen())
    text = tok.decode(out_ids)
    print(f"generated {len(out_ids)} tokens: {text!r}")
    assert len(out_ids) == 30

    elapsed = time.time() - t_start
    print(f"\n[total {elapsed:.1f}s]")
    print("=== SMOKE PIPELINE PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
