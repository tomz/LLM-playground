config = dict(
    out_dir='out/tiny',
    data_dir='data',
    # model
    n_layer=6, n_head=6, n_kv_head=6, d_model=384, d_ffn=1024,
    block_size=256, vocab_size=None,  # filled in by train.py from meta
    dropout=0.0, rope_base=10000.0,
    # optim
    batch_size=64, grad_accum=1,
    lr=6e-4, min_lr=6e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=100, lr_decay_iters=5000, max_iters=5000,
    grad_clip=1.0,
    # runtime
    eval_interval=250, eval_iters=50, log_interval=10, ckpt_interval=1000,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
