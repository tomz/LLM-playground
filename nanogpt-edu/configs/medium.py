config = dict(
    out_dir='out/medium',
    data_dir='data',
    n_layer=12, n_head=12, n_kv_head=12, d_model=768, d_ffn=2048,
    block_size=1024, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=12, grad_accum=4,
    lr=3e-4, min_lr=3e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=500, lr_decay_iters=50000, max_iters=50000,
    grad_clip=1.0,
    eval_interval=1000, eval_iters=200, log_interval=20, ckpt_interval=5000,
    device='auto', dtype='bfloat16', compile=True, seed=1337,
)
