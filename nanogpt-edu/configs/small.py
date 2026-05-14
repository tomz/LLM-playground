config = dict(
    out_dir='out/small',
    data_dir='data',
    n_layer=8, n_head=8, n_kv_head=8, d_model=512, d_ffn=1408,
    block_size=512, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=32, grad_accum=2,
    lr=5e-4, min_lr=5e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=200, lr_decay_iters=15000, max_iters=15000,
    grad_clip=1.0,
    eval_interval=500, eval_iters=100, log_interval=20, ckpt_interval=2000,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
