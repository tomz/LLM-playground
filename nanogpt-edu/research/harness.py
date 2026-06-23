"""Fixed research harness — the part the agent must NOT edit.

`autoresearch` for nanogpt-edu, GPU-only. The agent edits `candidate.py` (knobs
+ an optional model patch); this file trains that candidate under a *fixed
budget*, extracts a multi-metric result, and runs correctness/quality gates.
The keep/revert decision lives in `loop.py`.

Design (and how it improves on karpathy/autoresearch + autokernel):

* **GPU-only.** ~100 experiments/night only makes sense on a GPU; a CPU loop is
  useless. We assert CUDA up front and fail loudly (autokernel's `check_cuda`
  spirit) rather than silently running something slow.
* **Token-budget by default** (`--tokens`), wall-clock optional (`--minutes`).
  autoresearch uses a fixed *wall-clock* budget and its author notes results
  then "become not comparable to other people running on other compute". A
  *token* budget makes a 5060 Ti run and an H100 run land on the **same**
  val_bpb curve — reproducible science, the platform-comparability gap fixed.
* **Multi-metric.** Not one scalar: we report `val_bpb` (quality, vocab-size-
  independent so architecture changes compare fairly), plus throughput, peak
  VRAM, and param count — so the *tradeoff frontier* is visible (see plot.py).
* **Gates.** A kept experiment must (a) produce finite, descending loss,
  (b) not win by overfitting (train/val generalization gap bounded), and the
  harness itself preserves nanogpt-edu's determinism contract. These are the
  training analogue of autokernel's `correct: True`.

`val_bpb` (validation **bits per byte**) is the headline metric: cross-entropy
in nats converted to bits and normalised by bytes-per-token, so it is comparable
across vocab sizes / tokenizers. For a char-level (byte-ish) tokenizer
bytes_per_token≈1, but the formula is written so a BPE swap stays fair.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import ShardDataset, load_meta  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from train import cosine_lr, evaluate, make_autocast  # noqa: E402

LN2 = math.log(2.0)


@dataclass
class Result:
    """One experiment's measured outcome (a row of the ledger)."""
    val_bpb: float          # headline: validation bits/byte (lower better)
    val_loss: float         # raw CE in nats
    train_loss: float       # final train CE (for the generalization gate)
    gen_gap: float          # train↔val gap (val_loss − train_loss)
    tok_per_s: float        # throughput
    vram_mb: float          # peak allocated MiB (0 off-GPU)
    params_m: float         # total params, millions
    tokens: int             # tokens actually trained on
    wall_s: float           # train wall-clock (excl. compile/setup)
    ok: bool                # passed all gates
    reason: str             # gate verdict / failure cause

    def as_row(self) -> dict:
        return asdict(self)


def require_cuda() -> str:
    """GPU-only by design. Fail loudly with an actionable message off-GPU."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "research harness is GPU-only (a CPU loop is too slow to be useful).\n"
            "  - need a CUDA device; `nvidia-smi` should work.\n"
            "  - for CPU/MPS experimentation use train.py directly (configs/*.py)."
        )
    return "cuda"


def _load_candidate(path: str):
    """Import the editable candidate module fresh (so loop.py picks up edits)."""
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(knobs: dict, vocab_size: int, device: str, patch_fn=None) -> tuple[GPT, GPTConfig]:
    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=knobs["block_size"],
        n_layer=knobs["n_layer"], n_head=knobs["n_head"], n_kv_head=knobs["n_kv_head"],
        d_model=knobs["d_model"], d_ffn=knobs["d_ffn"],
        dropout=knobs.get("dropout", 0.0), rope_base=knobs.get("rope_base", 10000.0),
        qk_norm=knobs.get("qk_norm", False),
        zero_init_proj=knobs.get("zero_init_proj", False),
        tie_embeddings=knobs.get("tie_embeddings", True),
        mtp_tokens=knobs.get("mtp_tokens", 0),
        mtp_weight=knobs.get("mtp_weight", 0.3),
        attention_backend=knobs.get("attention_backend", "sdpa"),
    )
    model = GPT(cfg).to(device)
    # Optional architecture patch — the agent's most powerful lever. It receives
    # the constructed model and may mutate it in place (e.g. swap an init,
    # re-tie/untie, add a hook). Kept behind a try so a broken patch is a clean
    # crash the loop reverts, not a harness bug.
    if patch_fn is not None:
        patched = patch_fn(model)
        if patched is not None:
            model = patched
    return model, cfg


def build_optimizers(model: GPT, knobs: dict, device: str):
    is_cuda = device.startswith("cuda")
    if knobs.get("optimizer", "adamw") == "muon":
        from muon import Muon, split_muon_params
        muon_params, adamw_params = split_muon_params(model)
        muon = Muon(muon_params, lr=knobs.get("muon_lr", 0.02),
                    momentum=knobs.get("muon_momentum", 0.95))
        aux = torch.optim.AdamW(adamw_params, lr=knobs["lr"], betas=tuple(knobs["betas"]),
                                weight_decay=knobs["weight_decay"], fused=is_cuda)
        opts = [muon, aux]
    else:
        decay, no_decay = [], []
        for _, p in model.named_parameters():
            (no_decay if p.dim() < 2 else decay).append(p)
        opts = [torch.optim.AdamW(
            [{"params": decay, "weight_decay": knobs["weight_decay"]},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=knobs["lr"], betas=tuple(knobs["betas"]), fused=is_cuda)]
    for opt in opts:
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
    return opts


def run_candidate(candidate_path: str, *, data_dir: str, token_budget: int | None,
                  minute_budget: float | None, seed: int, eval_iters: int,
                  bytes_per_token: float, device: str) -> Result:
    """Train the candidate under a fixed budget and return its measured Result.

    Budget: exactly one of ``token_budget`` / ``minute_budget`` is honoured.
    Token budget is the reproducible default (same science on any GPU). A
    fixed-step warmup/decode schedule is derived from the budget so the cosine
    LR still completes within it.
    """
    torch.manual_seed(seed)
    train_gen = torch.Generator()
    train_gen.manual_seed(seed + 1)

    cand = _load_candidate(candidate_path)
    knobs = dict(cand.KNOBS)
    patch_fn = getattr(cand, "patch_model", None)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[knobs.get("dtype", "bfloat16")]
    ctx = make_autocast(device, dtype)
    is_cuda = device.startswith("cuda")

    meta = load_meta(data_dir)
    vocab_size = meta["vocab_size"]
    knobs["vocab_size"] = vocab_size  # for the descent gate's ln(vocab) ceiling
    block = knobs["block_size"]
    train_ds = ShardDataset(data_dir, "train", block, device)
    val_ds = ShardDataset(data_dir, "val", block, device)

    model, _ = build_model(knobs, vocab_size, device, patch_fn=patch_fn)
    params_m = model.num_params(non_embedding=False) / 1e6
    opts = build_optimizers(model, knobs, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and is_cuda))

    bs, ga = knobs["batch_size"], knobs["grad_accum"]
    tokens_per_step = bs * ga * block
    # Derive a step budget. For the token budget it's exact; for the wall-clock
    # budget we cap by a generous step ceiling and break on time.
    if token_budget is not None:
        max_steps = max(1, token_budget // tokens_per_step)
    else:
        max_steps = 10**9  # effectively unbounded; the timer below stops us
    warmup = max(1, int(0.05 * max_steps)) if token_budget else max(1, knobs.get("warmup_iters", 50))
    decay = max_steps if token_budget else 10**9
    base_lr = knobs["lr"]

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    model.train()
    t_start = time.time()
    last_train_loss = float("nan")
    deadline = (t_start + minute_budget * 60.0) if minute_budget is not None else None
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() >= deadline:
            break
        lr = cosine_lr(step, warmup, decay, base_lr, knobs["min_lr"])
        mult = lr / base_lr if base_lr else 1.0
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult
            opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(ga):
            x, y = train_ds.get_batch(bs, generator=train_gen)
            with ctx:
                _, loss = model(x, y)
                loss = loss / ga
            scaler.scale(loss).backward()
            loss_sum += loss.detach().float().item()
        if knobs.get("grad_clip"):
            for opt in opts:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), knobs["grad_clip"])
        for opt in opts:
            scaler.step(opt)
        scaler.update()
        last_train_loss = loss_sum
        step += 1
        # Early abort on a diverged run — don't waste the budget on NaNs.
        if not math.isfinite(last_train_loss):
            break

    wall_s = time.time() - t_start
    tokens = step * tokens_per_step
    tok_per_s = tokens / wall_s if wall_s > 0 else 0.0
    vram_mb = (torch.cuda.max_memory_allocated() / 2**20) if is_cuda else 0.0

    # Final eval (fresh seeded generator → comparable across candidates).
    knob_eval = {"batch_size": bs, "eval_iters": eval_iters}
    ev = evaluate(model, {"train": train_ds, "val": val_ds}, knob_eval, ctx,
                  eval_seed=seed + 12345)
    val_loss, train_loss = ev["val"], ev["train"]
    val_bpb = val_loss / (LN2 * max(bytes_per_token, 1e-9))
    gen_gap = val_loss - train_loss

    ok, reason = gate(val_loss, last_train_loss, gen_gap, knobs)
    return Result(
        val_bpb=round(val_bpb, 5), val_loss=round(val_loss, 5),
        train_loss=round(train_loss, 5), gen_gap=round(gen_gap, 5),
        tok_per_s=round(tok_per_s, 1), vram_mb=round(vram_mb, 1),
        params_m=round(params_m, 3), tokens=tokens, wall_s=round(wall_s, 1),
        ok=ok, reason=reason,
    )


def gate(val_loss: float, last_train_loss: float, gen_gap: float, knobs: dict) -> tuple[bool, str]:
    """Correctness/quality gates — a candidate the agent must not be able to fake.

    * **finite**: training loss and val loss must be real numbers (no NaN/Inf).
    * **descended**: the model must have actually learned (val below the
      ~log(vocab) random-init ceiling), so a no-op or broken patch can't pass.
    * **generalization**: the train↔val gap must stay under a cap, so a
      candidate can't win val_bpb by quietly overfitting (nanogpt-edu's signature
      `tiny` vs `tiny_clean` lesson, turned into a gate against the agent).
    """
    if not (math.isfinite(val_loss) and math.isfinite(last_train_loss)):
        return False, "non-finite loss"
    # Random-init CE ceiling ~ ln(vocab); require a clear margin below it.
    ceiling = math.log(max(knobs["vocab_size"], 2)) if "vocab_size" in knobs else 6.0
    if val_loss >= ceiling - 0.10:
        return False, f"did not descend (val {val_loss:.3f} ≥ ~ln(vocab) {ceiling:.3f})"
    gap_cap = float(knobs.get("max_gen_gap", 1.5))
    if gen_gap > gap_cap:
        return False, f"overfit (train↔val gap {gen_gap:.3f} > {gap_cap})"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description="GPU-only research harness (do not edit).")
    ap.add_argument("--candidate", default=str(Path(__file__).with_name("candidate.py")))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--tokens", type=int, default=2_000_000,
                    help="token budget (reproducible default; ~mins on a 5060 Ti)")
    ap.add_argument("--minutes", type=float, default=None,
                    help="wall-clock budget instead of tokens (autoresearch-style)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-iters", type=int, default=100)
    ap.add_argument("--bytes-per-token", type=float, default=1.0,
                    help="for val_bpb: chars≈1 byte; set per your tokenizer")
    ap.add_argument("--json-out", default=None, help="write the Result as JSON here")
    args = ap.parse_args()

    device = require_cuda()
    token_budget = None if args.minutes is not None else args.tokens
    res = run_candidate(
        args.candidate, data_dir=args.data_dir, token_budget=token_budget,
        minute_budget=args.minutes, seed=args.seed, eval_iters=args.eval_iters,
        bytes_per_token=args.bytes_per_token, device=device,
    )
    budget = f"{args.minutes} min" if args.minutes is not None else f"{args.tokens:,} tok"
    print(f"[harness] device={device}  budget={budget}")
    print(f"[harness] params={res.params_m:.2f}M  tokens={res.tokens:,}  wall={res.wall_s:.1f}s  "
          f"tok/s={res.tok_per_s/1e3:.1f}k  vram={res.vram_mb:.0f}MiB")
    # Grep-friendly metric lines (autokernel-style) so an agent can parse stdout.
    print(f"val_bpb:    {res.val_bpb:.5f}")
    print(f"val_loss:   {res.val_loss:.5f}")
    print(f"gen_gap:    {res.gen_gap:.5f}")
    print(f"ok:         {res.ok}")
    print(f"reason:     {res.reason}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res.as_row(), indent=2))
        print(f"[harness] wrote {args.json_out}")


if __name__ == "__main__":
    main()
