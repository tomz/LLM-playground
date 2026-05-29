"""tiny_mtp: the `tiny_clean` architecture + Multi-Token Prediction.

Adds auxiliary heads that predict tokens n+2 (and n+3) alongside the standard
n+1 head (DeepSeek-V3 style, simplified to plain linear heads). The extra
gradient signal improves sample efficiency in early training — exactly the
regime our tiny runs live in. Train-only: sample.py / generate() use the main
head only, so there is zero inference cost.

Diffs vs tiny_clean.py:
  * mtp_tokens 0 -> 2     (predict n+2 and n+3)
  * mtp_weight 0.3        (DeepSeek-V3's λ)
  * out_dir -> out/tiny_mtp

Pair with the Muon speedrun knobs by copying those fields from tiny_muon.py if
you want the full stack; kept on the AdamW baseline here so the MTP effect is
isolated for a clean A/B against tiny_clean.py.
"""
config = dict(
    out_dir='out/tiny_mtp',
    data_dir='data',
    # model
    n_layer=6, n_head=6, n_kv_head=6, d_model=384, d_ffn=1024,
    block_size=256, vocab_size=None,
    dropout=0.1, rope_base=10000.0,
    # multi-token prediction
    mtp_tokens=2, mtp_weight=0.3,
    # optim
    batch_size=64, grad_accum=1,
    lr=6e-4, min_lr=6e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=100, lr_decay_iters=1500, max_iters=1500,
    grad_clip=1.0,
    # runtime
    eval_interval=50, eval_iters=100, log_interval=10, ckpt_interval=500,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
