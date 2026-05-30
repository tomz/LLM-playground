"""SFT -> RM -> DPO chain on synthetic Shakespeare-character QA.

Uses example 01's base checkpoint (out/final.pt) and its BPE tokenizer.
Trains everything on a single CUDA GPU with platform.alignment primitives.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import json
import random
import time
from pathlib import Path

import torch

from platform.tokenizer.bpe import Tokenizer
from platform.model.transformer import Transformer
from platform.alignment._common import (
    tokenize_and_pack, compute_logps, clone_for_reference,
)
from platform.alignment.dpo import dpo_loss
from platform.alignment.reward_model import RewardModel, bt_loss
from platform.serving.engine import Engine, EngineConfig, GenRequest

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BASE_CKPT = HERE.parent / "01_pretrain_shakespeare" / "out" / "final.pt"

# Character -> play. Picked from the TinyShakespeare corpus.
CHAR_PLAY = {
    "Romeo": "Romeo and Juliet",
    "Juliet": "Romeo and Juliet",
    "Mercutio": "Romeo and Juliet",
    "Hamlet": "Hamlet",
    "Ophelia": "Hamlet",
    "Macbeth": "Macbeth",
    "Banquo": "Macbeth",
    "Othello": "Othello",
    "Iago": "Othello",
    "Desdemona": "Othello",
    "Coriolanus": "Coriolanus",
    "Menenius": "Coriolanus",
    "Brutus": "Julius Caesar",
    "Cassius": "Julius Caesar",
    "Portia": "The Merchant of Venice",
}
ALL_PLAYS = sorted(set(CHAR_PLAY.values()))


# ============================================================================
# Tokenizer wrapper (BPE has no pad_id; reuse eos as pad — model never sees it
# because we mask, and the RM scoring uses last-nonpad index).
# ============================================================================

class _BpeTokWithPad:
    """Adapter: adds a pad_id (= eos_id) so alignment helpers work."""
    def __init__(self, base: Tokenizer):
        self._t = base
        self.bos_id = base.bos_id
        self.eos_id = base.eos_id
        # tokenizers' bpe.Tokenizer .pad_id may be 0; use eos for safety.
        self.pad_id = base.eos_id
        self.vocab_size = base.vocab_size

    def encode(self, text: str):
        return self._t.encode(text)

    def decode(self, ids):
        return self._t.decode(list(ids))


# ============================================================================
# Synthetic data
# ============================================================================

def make_sft(rng: random.Random):
    items = []
    for char, play in CHAR_PLAY.items():
        for _ in range(14):       # ~14 paraphrases per character -> 210 rows
            prompt = f"Q: Which play features {char}?\nA:"
            response = f" {play}"
            items.append({"prompt": prompt, "response": response})
    rng.shuffle(items)
    return items[:180], items[180:200]


def make_prefs(rng: random.Random, n: int = 100):
    chars = list(CHAR_PLAY.keys())
    prefs = []
    for _ in range(n):
        char = rng.choice(chars)
        right = CHAR_PLAY[char]
        wrong = rng.choice([p for p in ALL_PLAYS if p != right])
        prefs.append({
            "prompt":   f"Q: Which play features {char}?\nA:",
            "chosen":   f" {right}",
            "rejected": f" {wrong}",
        })
    return prefs


def make_test(rng: random.Random, n: int = 10):
    chars = rng.sample(list(CHAR_PLAY.keys()), n)
    return [{"prompt": f"Q: Which play features {c}?\nA:",
             "character": c, "answer": CHAR_PLAY[c]} for c in chars]


def write_jsonl(path: Path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ============================================================================
# Loaders
# ============================================================================

def load_base(ckpt_path: Path) -> tuple[Transformer, Tokenizer, dict]:
    if not ckpt_path.exists():
        raise SystemExit(
            f"Base checkpoint not found at {ckpt_path}\n"
            f"Run examples/01_pretrain_shakespeare/run.sh first."
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["model_cfg"]
    model = Transformer(cfg)
    model.load_state_dict(state["model"])
    tok = Tokenizer(state["tokenizer_path"])
    return model, tok, state


# ============================================================================
# Training loops
# ============================================================================

def sft_loop(model: Transformer, tok, examples, *, steps: int, lr: float,
             batch_size: int, seq_len: int, device) -> list[float]:
    model.train()
    ids, mask = tokenize_and_pack(examples, tok, seq_len, mask_user_tokens=True)
    ids = ids.to(device); mask = mask.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n = ids.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(1)
    history = []
    import torch.nn.functional as F
    for step in range(steps):
        perm = torch.randint(0, n, (batch_size,), generator=rng).to(device)
        bx, bm = ids[perm], mask[perm]
        x, y = bx[:, :-1], bx[:, 1:]
        m = bm[:, 1:]
        logits, _ = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        gathered = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        loss = -(gathered * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
    return history


def rm_loop(model: Transformer, tok, prefs, *, steps: int, lr: float,
            batch_size: int, seq_len: int, device) -> tuple[RewardModel, list[float]]:
    # Fresh trunk copy so RM doesn't perturb SFT weights.
    trunk = copy.deepcopy(model)
    rm = RewardModel(trunk, pad_id=tok.pad_id).to(device)
    rm.train()
    chosen = [{"prompt": r["prompt"], "response": r["chosen"]} for r in prefs]
    rejected = [{"prompt": r["prompt"], "response": r["rejected"]} for r in prefs]
    ids_c, _ = tokenize_and_pack(chosen, tok, seq_len, mask_user_tokens=False)
    ids_r, _ = tokenize_and_pack(rejected, tok, seq_len, mask_user_tokens=False)
    ids_c = ids_c.to(device); ids_r = ids_r.to(device)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.0)
    n = ids_c.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(2)
    history = []
    for step in range(steps):
        perm = torch.randint(0, n, (batch_size,), generator=rng).to(device)
        sc = rm(ids_c[perm])
        sr = rm(ids_r[perm])
        loss = bt_loss(sc, sr, margin=0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
    rm.eval()
    return rm, history


def dpo_loop(policy: Transformer, tok, prefs, *, steps: int, lr: float,
             batch_size: int, seq_len: int, beta: float, device) -> list[float]:
    ref = clone_for_reference(policy).to(device)
    chosen = [{"prompt": r["prompt"], "response": r["chosen"]} for r in prefs]
    rejected = [{"prompt": r["prompt"], "response": r["rejected"]} for r in prefs]
    ids_c, mask_c = tokenize_and_pack(chosen, tok, seq_len, mask_user_tokens=True)
    ids_r, mask_r = tokenize_and_pack(rejected, tok, seq_len, mask_user_tokens=True)
    ids_c, mask_c = ids_c.to(device), mask_c.to(device)
    ids_r, mask_r = ids_r.to(device), mask_r.to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    n = ids_c.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(3)
    history = []
    policy.train()
    for step in range(steps):
        perm = torch.randint(0, n, (batch_size,), generator=rng).to(device)
        xc, yc, mc = ids_c[perm][:, :-1], ids_c[perm][:, 1:], mask_c[perm][:, 1:]
        xr, yr, mr = ids_r[perm][:, :-1], ids_r[perm][:, 1:], mask_r[perm][:, 1:]
        with torch.no_grad():
            rlp_c = compute_logps(ref, xc, yc, mc)
            rlp_r = compute_logps(ref, xr, yr, mr)
        plp_c = compute_logps(policy, xc, yc, mc)
        plp_r = compute_logps(policy, xr, yr, mr)
        loss = dpo_loss(plp_c, plp_r, rlp_c, rlp_r, beta=beta, variant="sigmoid")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
    return history


# ============================================================================
# Generation + scoring
# ============================================================================

def greedy_generate(model, tok, prompts, *, max_new_tokens: int = 20) -> list[str]:
    engine = Engine(EngineConfig(backend="torch", dtype="fp32"),
                    model=model, tokenizer=tok)
    results = []
    for p in prompts:
        ids = [tok.bos_id] + tok.encode(p)
        req = GenRequest(prompt_ids=ids, max_new_tokens=max_new_tokens,
                         temperature=0.0, top_p=1.0)
        async def _gen():
            out = []
            async for ev in engine.generate(req):
                if ev.get("done"):
                    break
                out.append(ev["token_id"])
            return out
        gen_ids = asyncio.run(_gen())
        # stop at newline if present
        text = tok.decode(gen_ids)
        text = text.split("\n")[0]
        results.append(text)
    return results


@torch.no_grad()
def rm_score(rm: RewardModel, tok, prompts, responses, *, seq_len: int = 128, device) -> list[float]:
    examples = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)]
    ids, _ = tokenize_and_pack(examples, tok, seq_len, mask_user_tokens=False)
    ids = ids.to(device)
    return rm(ids).float().cpu().tolist()


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default=str(BASE_CKPT))
    args = ap.parse_args()
    base_ckpt = Path(args.base_ckpt)

    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    dev_name = torch.cuda.get_device_name(0)
    print(f"[cuda] device={dev_name}")
    t_start = time.time()

    rng = random.Random(0)

    print("[base] loading example 01 checkpoint")
    base_model, base_tok_raw, base_state = load_base(base_ckpt)
    base_model = base_model.to(device).eval()
    tok = _BpeTokWithPad(base_tok_raw)
    print(f"[base] vocab={tok.vocab_size}  bos={tok.bos_id} eos={tok.eos_id} pad={tok.pad_id}")

    # ---- data ----
    sft_train, sft_eval = make_sft(rng)
    prefs = make_prefs(rng, n=100)
    test = make_test(rng, n=10)
    write_jsonl(OUT / "sft.jsonl", sft_train)
    write_jsonl(OUT / "eval_sft.jsonl", sft_eval)
    write_jsonl(OUT / "prefs.jsonl", prefs)
    write_jsonl(OUT / "test.jsonl", test)
    print(f"[data] sft_train={len(sft_train)}  sft_eval={len(sft_eval)}  "
          f"prefs={len(prefs)}  test={len(test)}")

    # ---- SFT ----
    print("[sft] training for 200 steps")
    t0 = time.time()
    sft_model = copy.deepcopy(base_model)
    sft_hist = sft_loop(sft_model, tok, sft_train, steps=200, lr=1e-4,
                        batch_size=4, seq_len=128, device=device)
    print(f"[sft] done in {time.time()-t0:.1f}s  "
          f"first10={sum(sft_hist[:10])/10:.3f}  last10={sum(sft_hist[-10:])/10:.3f}")
    torch.save({"model": sft_model.state_dict(),
                "model_cfg": sft_model.cfg,
                "loss_history": sft_hist}, OUT / "sft_model.pt")

    # ---- RM ----
    print("[rm] training for 200 steps")
    t0 = time.time()
    rm, rm_hist = rm_loop(sft_model, tok, prefs, steps=200, lr=5e-5,
                          batch_size=4, seq_len=128, device=device)
    print(f"[rm] done in {time.time()-t0:.1f}s  "
          f"first10={sum(rm_hist[:10])/10:.3f}  last10={sum(rm_hist[-10:])/10:.3f}")
    torch.save({"trunk": rm.trunk.state_dict(),
                "head": rm.head.state_dict(),
                "model_cfg": rm.trunk.cfg,
                "pad_id": tok.pad_id,
                "loss_history": rm_hist}, OUT / "rm.pt")

    # ---- DPO ----
    print("[dpo] training for 200 steps")
    t0 = time.time()
    dpo_model = copy.deepcopy(sft_model)
    dpo_hist = dpo_loop(dpo_model, tok, prefs, steps=200, lr=5e-6,
                        batch_size=4, seq_len=128, beta=0.1, device=device)
    print(f"[dpo] done in {time.time()-t0:.1f}s  "
          f"first10={sum(dpo_hist[:10])/10:.3f}  last10={sum(dpo_hist[-10:])/10:.3f}")
    torch.save({"model": dpo_model.state_dict(),
                "model_cfg": dpo_model.cfg,
                "loss_history": dpo_hist}, OUT / "dpo_model.pt")

    # ---- Comparison ----
    print("[eval] generating responses for 10 test prompts from base/SFT/DPO")
    prompts = [t["prompt"] for t in test]
    gold = [t["answer"] for t in test]

    base_resp = greedy_generate(base_model, base_tok_raw, prompts)
    sft_resp  = greedy_generate(sft_model,  base_tok_raw, prompts)
    dpo_resp  = greedy_generate(dpo_model,  base_tok_raw, prompts)

    base_scores = rm_score(rm, tok, prompts, base_resp, device=device)
    sft_scores  = rm_score(rm, tok, prompts, sft_resp,  device=device)
    dpo_scores  = rm_score(rm, tok, prompts, dpo_resp,  device=device)

    def em(resps):
        return sum(1 for r, g in zip(resps, gold) if g.lower() in r.lower()) / len(gold)

    base_em, sft_em, dpo_em = em(base_resp), em(sft_resp), em(dpo_resp)

    # ---- result.md ----
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    wall = time.time() - t_start

    lines = []
    lines.append("# 02 — SFT → RM → DPO alignment chain: result\n")
    lines.append(f"Recorded on **{dev_name}**, built on the example 01 base checkpoint.\n")
    lines.append("## Summary\n")
    lines.append(f"| stage | first-10 loss | last-10 loss |\n|---|--:|--:|")
    lines.append(f"| SFT (200 steps, lr=1e-4) | {sum(sft_hist[:10])/10:.3f} | {sum(sft_hist[-10:])/10:.3f} |")
    lines.append(f"| RM  (200 steps, lr=5e-5) | {sum(rm_hist[:10])/10:.3f} | {sum(rm_hist[-10:])/10:.3f} |")
    lines.append(f"| DPO (200 steps, lr=5e-6, β=0.1) | {sum(dpo_hist[:10])/10:.3f} | {sum(dpo_hist[-10:])/10:.3f} |")
    lines.append("")
    lines.append("## Held-out test prompts (10 characters)\n")
    lines.append("| char | gold | base | SFT | DPO |")
    lines.append("|---|---|---|---|---|")
    for i, t in enumerate(test):
        def cell(s, ok):
            mark = "✓" if ok else " "
            return f"{mark} `{s.strip()[:30]}`"
        lines.append(f"| {t['character']} | _{t['answer']}_ "
                     f"| {cell(base_resp[i], t['answer'].lower() in base_resp[i].lower())} "
                     f"| {cell(sft_resp[i],  t['answer'].lower() in sft_resp[i].lower())} "
                     f"| {cell(dpo_resp[i],  t['answer'].lower() in dpo_resp[i].lower())} |")
    lines.append("")
    lines.append("## Aggregate\n")
    lines.append("| model | exact-match acc | mean RM score |")
    lines.append("|---|--:|--:|")
    lines.append(f"| base | {base_em:.0%} | {sum(base_scores)/len(base_scores):+.3f} |")
    lines.append(f"| SFT  | {sft_em:.0%}  | {sum(sft_scores)/len(sft_scores):+.3f} |")
    lines.append(f"| DPO  | {dpo_em:.0%}  | {sum(dpo_scores)/len(dpo_scores):+.3f} |")
    lines.append("")
    lines.append(f"_Total wall time: {wall:.1f}s. Peak GPU memory: {peak_gb:.2f} GiB._\n")
    (HERE / "result.md").write_text("\n".join(lines))

    print(f"\n[done] wall={wall:.1f}s  peak_gpu={peak_gb:.2f} GiB")
    print(f"[done] EM base={base_em:.0%} SFT={sft_em:.0%} DPO={dpo_em:.0%}")
    print(f"[done] RM  base={sum(base_scores)/len(base_scores):+.3f}  "
          f"SFT={sum(sft_scores)/len(sft_scores):+.3f}  "
          f"DPO={sum(dpo_scores)/len(dpo_scores):+.3f}")


if __name__ == "__main__":
    main()
