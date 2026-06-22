"""tiny_add: a small char-level model for the multi-digit ADDITION task.

Trains on data_add/ (see prepare_addition.py) — a *verifiable* task used to
measure DeepConf's accuracy lift. Small block_size (the longest line
"99+99=198\\n" is ~10 chars) and a compact model that nonetheless learns 2-digit
addition well enough that confidence correlates with correctness.

    python prepare_addition.py --digits 2 --train 20000 --val 2000
    python train.py --config configs/tiny_add.py
    python tools/bench_deepconf.py --ckpt out/tiny_add/ckpt_best.pt --task add
"""
config = dict(
    out_dir='out/tiny_add',
    data_dir='data_add',
    # model — small but enough for 2-digit addition
    n_layer=4, n_head=4, n_kv_head=4, d_model=192, d_ffn=512,
    block_size=48, vocab_size=None,   # filled in by train.py from meta
    dropout=0.0, rope_base=10000.0,
    qk_norm=True,                      # stabilises the small-model training
    # optim
    batch_size=128, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=100, lr_decay_iters=3000, max_iters=3000,
    grad_clip=1.0,
    # runtime
    eval_interval=250, eval_iters=50, log_interval=50, ckpt_interval=1000,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
