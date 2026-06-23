"""The ONE file the research agent edits.

`harness.py` trains this candidate under a fixed budget and measures `val_bpb`
(+ throughput, VRAM, params). The keep/revert loop (`loop.py`) keeps the edit
iff `val_bpb` improves AND the gates pass, else reverts it with git.

Two levers, in increasing power:

1. **KNOBS** — a flat config dict. Flip documented techniques, resize the model,
   retune the optimizer. This is the safe, high-signal search space.
2. **patch_model(model)** — an OPTIONAL architecture patch applied right after
   construction. Return the (possibly new) model, or mutate in place and return
   None. This is the "everything is fair game" lever — but a broken patch is a
   clean crash the loop reverts, so experiment freely.

The seeded knob menu below mirrors nanogpt-edu's documented harvest (Muon,
QK-norm, zero-init, MTP, FlexAttention) so the agent starts from real SOTA
levers rather than a blind search. Change one thing at a time and keep a short
`DESCRIPTION` of what you tried — it lands in the ledger and the chart.
"""
from __future__ import annotations

# One-line summary of THIS experiment (shows up in ledger.tsv + progress.png).
DESCRIPTION = "baseline: 6L d256, AdamW, dropout 0.1"

KNOBS = {
    # --- model size (DEPTH is the primary complexity knob; rest scale with it) ---
    "n_layer": 6,
    "n_head": 8,
    "n_kv_head": 8,            # set < n_head for GQA (cheaper KV)
    "d_model": 256,
    "d_ffn": 768,
    "block_size": 256,
    "dropout": 0.1,            # regularization — nanogpt-edu's overfit antidote
    "rope_base": 10000.0,

    # --- SOTA technique knobs (the seeded search space; all default-off here) ---
    # "qk_norm": True,        # per-head RMSNorm on Q,K — stabilises, allows higher LR
    # "zero_init_proj": True, # zero-init residual-write mats — stable high-LR warmup
    # "tie_embeddings": False,# untie lm_head — helps once you have the tokens
    # "mtp_tokens": 2,        # Multi-Token Prediction aux heads — denser gradient
    # "attention_backend": "flex",  # FlexAttention (long-ctx/mask experiments)

    # --- optimizer ---
    "optimizer": "adamw",     # "adamw" | "muon"
    "lr": 1.0e-3,
    "min_lr": 1.0e-4,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    # "muon_lr": 0.02,        # used when optimizer == "muon"
    # "muon_momentum": 0.95,

    # --- batch / precision ---
    "batch_size": 64,
    "grad_accum": 1,
    "dtype": "bfloat16",      # "bfloat16" | "float16" | "float32"
    "grad_clip": 1.0,
    "warmup_iters": 50,       # used only under the wall-clock budget

    # --- gate tuning (don't loosen to cheat the overfit guard) ---
    "max_gen_gap": 1.5,       # max train↔val CE gap before a run is rejected
}


def patch_model(model):
    """Optional architecture patch — return the model (or None to keep in place).

    Default is a no-op. Examples the agent might try (one at a time):

        # untie embeddings without re-init:
        # import torch.nn as nn
        # model.lm_head = nn.Linear(model.cfg.d_model, model.cfg.vocab_size, bias=False)
        # nn.init.normal_(model.lm_head.weight, std=0.02)

        # scale the residual stream at init, add a custom hook, etc.

    Keep patches small and reviewable; the loop reverts anything that crashes
    or fails to improve val_bpb.
    """
    return None
