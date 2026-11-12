"""Fixed research harness — candidate-search agents must NOT edit this file.

Maintainer contract revision 2: exact positive whole-step token budgets, validated
normalization/time inputs, and explicit search-validation versus final-holdout
roles. The CLI remains GPU-only; tiny direct CPU calls are regression tests, not
quality or performance evidence. Candidate code is trusted, not sandboxed.

Equal token exposure controls one variable, not hardware numerics, FLOPs, wall
cost, or adaptive-selection bias. ``val_bpb`` estimates bits per UTF-8 byte from
sampled CE and the selected split's bytes/token factor. BPE needs measured corpus
normalization (or a documented explicit factor); one byte per token is not a
valid general default. Gates are diagnostics, not proof against overfitting.
"""
from __future__ import annotations

import argparse
import hashlib
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

from data import ShardDataset, character_bytes_per_token, load_manifest, load_meta  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from train import cosine_lr, evaluate, make_autocast  # noqa: E402

LN2 = math.log(2.0)
HARNESS_CONTRACT_VERSION = 2
DEFAULT_TOKEN_BUDGET = 2_097_152  # 128 baseline steps of 64 * 1 * 256 tokens


@dataclass
class Result:
    """One experiment's measured outcome (legacy positional fields unchanged)."""
    val_bpb: float          # estimated validation bits/byte (lower better)
    val_loss: float         # raw CE in nats
    train_loss: float       # final train CE (for the generalization gate)
    gen_gap: float          # val_loss minus train_loss
    tok_per_s: float        # training throughput
    vram_mb: float          # peak allocated MiB during training (0 off-GPU)
    params_m: float         # total params, millions
    tokens: int            # tokens actually trained on
    wall_s: float          # training wall-clock, excluding setup/final evaluation
    ok: bool               # passed all gates
    reason: str            # gate verdict / failure cause
    verifier_cost_s: float | None = None  # final evaluation plus gate time
    evaluation_role: str = "search_validation"
    evaluation_split: str = "val"
    bytes_per_token: float | None = None
    byte_normalization: str = "explicit"
    data_manifest_sha256: str | None = None
    harness_contract_version: int = HARNESS_CONTRACT_VERSION

    def as_row(self) -> dict:
        return asdict(self)


def require_cuda() -> str:
    """GPU-only CLI. Fail loudly with an actionable message off-GPU."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "research harness is GPU-only (a CPU loop is too slow to be useful).\n"
            "  - need a CUDA device; `nvidia-smi` should work.\n"
            "  - for CPU/MPS experimentation use train.py directly (configs/*.py)."
        )
    return "cuda"


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(name: str, value: float) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def validate_budget(knobs: dict, *, token_budget: int | None,
                    minute_budget: float | None, eval_iters: int,
                    bytes_per_token: float | None) -> tuple[int, int]:
    """Return (tokens_per_step, max_steps), rejecting rounding or overshoot."""
    if (token_budget is None) == (minute_budget is None):
        raise ValueError("exactly one of token_budget or minute_budget is required")
    for key in ("batch_size", "grad_accum", "block_size"):
        _positive_int(key, knobs[key])
    _positive_int("eval_iters", eval_iters)
    if bytes_per_token is not None:
        _positive_finite("bytes_per_token", bytes_per_token)
    tokens_per_step = knobs["batch_size"] * knobs["grad_accum"] * knobs["block_size"]
    if token_budget is not None:
        _positive_int("token_budget", token_budget)
        if token_budget < tokens_per_step or token_budget % tokens_per_step:
            raise ValueError(
                f"token_budget must be at least one full step and divisible by "
                f"batch_size * grad_accum * block_size = {tokens_per_step}; got {token_budget}"
            )
        return tokens_per_step, token_budget // tokens_per_step
    _positive_finite("minute_budget", minute_budget)
    _positive_finite("minute_budget in seconds", minute_budget * 60.0)
    _positive_int("warmup_iters", knobs.get("warmup_iters", 50))
    return tokens_per_step, 10**9  # timer ends the wall-clock mode between steps


def resolve_byte_normalization(meta: dict, split: str, supplied: float | None, *,
                               data_dir: str | None = None) -> tuple[float, str]:
    """Resolve measured UTF-8 bytes per stored token, never a guessed BPE ratio.

    Prepared web shards include a zero-byte EOT token per document. A legacy
    character shard is scanned in bounded chunks using its actual UTF-8 map.
    Sampled CE times this corpus ratio estimates BPB, not exact byte likelihood.
    An explicit factor on unmeasured legacy data is a recorded caller assertion.
    """
    if supplied is not None:
        _positive_finite("bytes_per_token", supplied)
    measured = meta.get("bytes_per_token", {}).get(split)
    normalization = ("prepared_split_utf8_bytes_per_stored_token" if "data_manifest_schema" in meta
                     else "legacy_metadata_utf8_bytes_per_stored_token")
    if data_dir is not None and meta.get("itos"):
        actual = character_bytes_per_token(data_dir, split, meta)
        if measured is not None:
            _positive_finite(f"{split} bytes_per_token", measured)
            if not math.isclose(actual, measured, rel_tol=1e-12):
                raise ValueError("character byte measurement disagrees with metadata")
        measured = actual
        normalization = "measured_character_split_utf8_bytes_per_stored_token"
    if measured is not None:
        _positive_finite(f"{split} bytes_per_token", measured)
        if supplied is not None and not math.isclose(supplied, measured, rel_tol=1e-12):
            raise ValueError("explicit bytes_per_token disagrees with the prepared split")
        return float(measured), normalization
    if supplied is not None:
        return float(supplied), "explicit_caller_factor"
    chars = meta.get("itos", {})
    chars = chars.values() if isinstance(chars, dict) else chars
    if chars and all(isinstance(char, str) and len(char.encode("utf-8")) == 1 for char in chars):
        return 1.0, "single_byte_character_vocabulary"
    raise ValueError("bytes_per_token is unknown; prepare a split manifest or supply a measured --bytes-per-token")


def _load_candidate(path: str):
    """Import the editable, trusted candidate fresh so the loop picks up edits."""
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
                  bytes_per_token: float | None, device: str,
                  final_holdout: bool = False) -> Result:
    """Train with exact whole steps, then evaluate the explicitly selected split.

    Invalid budgets fail before data/model/optimizer construction. A divergent
    run may abort early and reports actual work, never the requested budget.
    Search reads only train/val. Final holdout retrains a frozen candidate and
    reads train/holdout; its result must never feed the keep/revert loop.
    """
    cand = _load_candidate(candidate_path)
    knobs = dict(cand.KNOBS)
    tokens_per_step, max_steps = validate_budget(
        knobs, token_budget=token_budget, minute_budget=minute_budget,
        eval_iters=eval_iters, bytes_per_token=bytes_per_token,
    )
    if type(final_holdout) is not bool:
        raise ValueError("final_holdout must be a boolean")
    gap_cap = knobs.get("max_gen_gap", 1.5)
    if isinstance(gap_cap, bool) or not isinstance(gap_cap, (int, float)) or not math.isfinite(gap_cap) or gap_cap < 0:
        raise ValueError("max_gen_gap must be finite and nonnegative")
    split = "holdout" if final_holdout else "val"
    meta = load_meta(data_dir)
    manifest = load_manifest(data_dir, splits=("train", split), meta=meta)
    if final_holdout and manifest is None:
        raise ValueError("final_holdout requires a prepared document-split manifest; legacy val is not a sealed test")
    bytes_per_token, normalization = resolve_byte_normalization(
        meta, split, bytes_per_token, data_dir=data_dir,
    )
    manifest_path = Path(data_dir) / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest is not None else None
    # Validation precedes model/optimizer/RNG construction; search never opens
    # holdout tokens or its document ledger while checking these identities.
    torch.manual_seed(seed)
    train_gen = torch.Generator().manual_seed(seed + 1)
    patch_fn = getattr(cand, "patch_model", None)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[knobs.get("dtype", "bfloat16")]
    ctx = make_autocast(device, dtype)
    is_cuda = device.startswith("cuda")
    vocab_size = meta["vocab_size"]
    knobs["vocab_size"] = vocab_size
    block = knobs["block_size"]
    train_ds = ShardDataset(data_dir, "train", block, device)
    eval_ds = ShardDataset(data_dir, split, block, device)

    model, _ = build_model(knobs, vocab_size, device, patch_fn=patch_fn)
    params_m = model.num_params(non_embedding=False) / 1e6
    opts = build_optimizers(model, knobs, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and is_cuda))
    bs, ga = knobs["batch_size"], knobs["grad_accum"]
    warmup = max(1, int(0.05 * max_steps)) if token_budget is not None else knobs.get("warmup_iters", 50)
    decay = max_steps
    base_lr = knobs["lr"]

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    t_start = time.perf_counter()
    last_train_loss = float("nan")
    deadline = (t_start + minute_budget * 60.0) if minute_budget is not None else None
    step = 0
    while step < max_steps:
        if deadline is not None and time.perf_counter() >= deadline:
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
        if not math.isfinite(last_train_loss):
            break

    if is_cuda:
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_start
    tokens = step * tokens_per_step
    tok_per_s = tokens / wall_s if wall_s > 0 else 0.0
    vram_mb = (torch.cuda.max_memory_allocated() / 2**20) if is_cuda else 0.0

    eval_start = time.perf_counter()
    knob_eval = {"batch_size": bs, "eval_iters": eval_iters}
    ev = evaluate(model, {"train": train_ds, "val": eval_ds}, knob_eval, ctx,
                  eval_seed=seed + 12345)
    val_loss, train_loss = ev["val"], ev["train"]
    val_bpb = val_loss / (LN2 * bytes_per_token)
    gen_gap = val_loss - train_loss
    ok, reason = gate(val_loss, last_train_loss, gen_gap, knobs)
    if is_cuda:
        torch.cuda.synchronize()
    verifier_cost_s = time.perf_counter() - eval_start
    return Result(
        val_bpb=round(val_bpb, 5), val_loss=round(val_loss, 5),
        train_loss=round(train_loss, 5), gen_gap=round(gen_gap, 5),
        tok_per_s=round(tok_per_s, 1), vram_mb=round(vram_mb, 1),
        params_m=round(params_m, 3), tokens=tokens, wall_s=round(wall_s, 6),
        ok=ok, reason=reason, verifier_cost_s=round(verifier_cost_s, 6),
        evaluation_role="final_holdout" if final_holdout else "search_validation",
        evaluation_split=split, bytes_per_token=bytes_per_token,
        byte_normalization=normalization, data_manifest_sha256=manifest_hash,
    )


def gate(val_loss: float, last_train_loss: float, gen_gap: float, knobs: dict) -> tuple[bool, str]:
    """Finite/descent/gap diagnostics, not independent generalization evidence."""
    if not all(math.isfinite(value) for value in (val_loss, last_train_loss, gen_gap)):
        return False, "non-finite loss or generalization gap"
    ceiling = math.log(max(knobs["vocab_size"], 2)) if "vocab_size" in knobs else 6.0
    if val_loss >= ceiling - 0.10:
        return False, f"did not descend (val {val_loss:.3f} ≥ ~ln(vocab) {ceiling:.3f})"
    gap_cap = float(knobs.get("max_gen_gap", 1.5))
    if not math.isfinite(gap_cap) or gap_cap < 0:
        return False, "invalid generalization gap cap"
    if gen_gap > gap_cap:
        return False, f"overfit (train↔val gap {gen_gap:.3f} > {gap_cap})"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description="GPU-only research harness (candidate agents must not edit).")
    ap.add_argument("--candidate", default=str(Path(__file__).with_name("candidate.py")))
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    budgets = ap.add_mutually_exclusive_group()
    budgets.add_argument("--tokens", type=int, default=DEFAULT_TOKEN_BUDGET,
                         help="exact positive whole-step token budget (default: 2097152)")
    budgets.add_argument("--minutes", type=float, default=None,
                         help="positive finite wall-clock budget instead of tokens")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-iters", type=int, default=100)
    ap.add_argument("--bytes-per-token", type=float, default=None,
                    help="measured factor for legacy shards; otherwise use prepared split metadata")
    ap.add_argument("--final-holdout", action="store_true",
                    help="evaluate holdout.bin after freezing selection; never feed this result to loop.py")
    ap.add_argument("--json-out", default=None, help="write the Result as JSON here")
    args = ap.parse_args()

    device = require_cuda()
    res = run_candidate(
        args.candidate, data_dir=args.data_dir,
        token_budget=None if args.minutes is not None else args.tokens,
        minute_budget=args.minutes, seed=args.seed, eval_iters=args.eval_iters,
        bytes_per_token=args.bytes_per_token, device=device, final_holdout=args.final_holdout,
    )
    budget = f"{args.minutes} min" if args.minutes is not None else f"{args.tokens:,} tok"
    print(f"[harness] contract={HARNESS_CONTRACT_VERSION}  device={device}  budget={budget}  role={res.evaluation_role}")
    print(f"[harness] params={res.params_m:.2f}M  tokens={res.tokens:,}  wall={res.wall_s:.1f}s  "
          f"tok/s={res.tok_per_s/1e3:.1f}k  vram={res.vram_mb:.0f}MiB")
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
