"""tiny_add3: char-level model for 3-DIGIT addition — the harder DeepConf substrate.

3-digit operands span a 1M-pair space; with 40k train examples the model sees
~4% and *must generalize*, so when it errs it errs with genuine uncertainty
(diverse, low-confidence wrong answers) — exactly the regime where DeepConf's
confidence filtering beats a plain majority vote (2-digit addition is too easy:
errors there are confident and systematic, so filtering can't help).

    python prepare_addition.py --digits 3 --train 40000 --val 2000
    python train.py --config configs/tiny_add3.py
    python tools/bench_deepconf.py --ckpt out/tiny_add3/ckpt_best.pt --task add
"""
config = dict(
    out_dir='out/tiny_add3',
    data_dir='data_add',
    # model — a bit larger than tiny_add for the harder task
    n_layer=6, n_head=8, n_kv_head=8, d_model=256, d_ffn=768,
    block_size=48, vocab_size=None,
    dropout=0.0, rope_base=10000.0,
    qk_norm=True,
    # optim
    batch_size=128, grad_accum=1,
    lr=1e-3, min_lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=150, lr_decay_iters=6000, max_iters=6000,
    grad_clip=1.0,
    # runtime
    eval_interval=250, eval_iters=50, log_interval=50, ckpt_interval=1000,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
