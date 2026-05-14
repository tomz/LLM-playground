config = dict(
    out_dir='out/smoke',
    data_dir='data',
    n_layer=4, n_head=4, n_kv_head=4, d_model=128, d_ffn=384,
    block_size=128, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    batch_size=32, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=20, lr_decay_iters=300, max_iters=300,
    grad_clip=1.0,
    eval_interval=100, eval_iters=20, log_interval=25, ckpt_interval=200,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
