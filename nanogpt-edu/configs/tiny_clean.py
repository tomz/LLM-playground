"""tiny_clean: same architecture as tiny.py but with dropout and early stopping
to produce a textbook U-shaped val-loss curve (descent → minimum → mild ascent)
instead of the overfit-into-the-floor trajectory of tiny.py.

Diffs vs tiny.py:
  * dropout 0.0 -> 0.1
  * max_iters 5000 -> 1500
  * lr_decay_iters 5000 -> 1500 (cosine bottoms out with the run)
  * eval_interval 250 -> 50    (denser val curve around the U-bottom)
  * eval_iters 50 -> 100       (smoother val estimates)
  * out_dir out/tiny -> out/tiny_clean
"""
config = dict(
    out_dir='out/tiny_clean',
    data_dir='data',
    # model
    n_layer=6, n_head=6, n_kv_head=6, d_model=384, d_ffn=1024,
    block_size=256, vocab_size=None,
    dropout=0.1, rope_base=10000.0,
    # optim
    batch_size=64, grad_accum=1,
    lr=6e-4, min_lr=6e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=100, lr_decay_iters=1500, max_iters=1500,
    grad_clip=1.0,
    # runtime
    eval_interval=50, eval_iters=100, log_interval=10, ckpt_interval=500,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
