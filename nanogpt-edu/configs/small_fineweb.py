"""small_fineweb: the `small` (~25M) architecture trained on FineWeb-Edu BPE
tokens instead of 1 MB of char-level Shakespeare.

This is the config that makes the README's "bigger model is worse on val"
inversion *disappear*: once tokens scale with params, the larger model wins.
Pairs the modern speedrun knobs (Muon + QK-norm + zero-init + untied head)
with real data.

Prereq:
    python prepare_fineweb.py --tokens 100_000_000 --out-dir data_fineweb

Notes:
  * vocab_size is filled in by train.py from meta.pkl (GPT-2 BPE = 50257).
  * dropout 0.0 is fine here — with 100M tokens the model can't memorize the
    set, so the regularizer that `tiny_clean` needed isn't required.
  * max_iters/lr_decay are set for a multi-hour run; trim for a quick look.
"""
config = dict(
    out_dir='out/small_fineweb',
    data_dir='data_fineweb',
    n_layer=8, n_head=8, n_kv_head=8, d_model=512, d_ffn=1408,
    block_size=512, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    # speedrun knobs
    qk_norm=True, zero_init_proj=True, tie_embeddings=False,
    optimizer='muon', muon_lr=0.02, muon_momentum=0.95,
    batch_size=32, grad_accum=2,
    lr=5e-4, min_lr=5e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=200, lr_decay_iters=20000, max_iters=20000,
    grad_clip=1.0,
    eval_interval=500, eval_iters=100, log_interval=20, ckpt_interval=2000,
    device='auto', dtype='bfloat16', compile=True, seed=1337,
)
