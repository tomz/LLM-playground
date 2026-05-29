"""tiny_muon: the modded-nanogpt speedrun recipe at the `tiny` scale.

Same 10.65M architecture as tiny_clean.py, but flips on the speedrun knobs:
  * optimizer = "muon"   — Newton-Schulz-orthogonalized updates on 2D hidden
    weights, AdamW on embeddings/lm_head/norms. ~1.35x sample-efficiency on
    FineWeb in Keller Jordan's records.
  * qk_norm = True        — RMSNorm on Q/K; stabilises higher LR.
  * zero_init_proj = True — residual-write matrices start at zero (identity
    blocks); supports the aggressive warmup.
  * tie_embeddings = False— untie embed/head (helps loss once tokens allow).

Muon's effective LR lives on a different scale than AdamW's, hence muon_lr is
set independently (0.02 is a sane single-GPU starting point; the 8xH100 records
go higher). `lr` here is the AdamW LR for the non-Muon params.

This is meant as a head-to-head against tiny_clean.py: same data, same iters,
same wall-budget — read the val curve to see Muon's sample-efficiency win.
On TinyShakespeare (1 MB) the dataset is the bottleneck, not the optimizer, so
the gap is modest; for the real demo pair this with the FineWeb-Edu data
(see prepare_fineweb.py) and the `small`/`medium` scale.
"""
config = dict(
    out_dir='out/tiny_muon',
    data_dir='data',
    # model
    n_layer=6, n_head=6, n_kv_head=6, d_model=384, d_ffn=1024,
    block_size=256, vocab_size=None,
    dropout=0.1, rope_base=10000.0,
    # speedrun architecture knobs
    qk_norm=True, zero_init_proj=True, tie_embeddings=False,
    # optim
    optimizer='muon', muon_lr=0.02, muon_momentum=0.95,
    batch_size=64, grad_accum=1,
    lr=6e-4, min_lr=6e-5, weight_decay=0.1, betas=(0.9, 0.95),
    warmup_iters=100, lr_decay_iters=1500, max_iters=1500,
    grad_clip=1.0,
    # runtime
    eval_interval=50, eval_iters=100, log_interval=10, ckpt_interval=500,
    device='auto', dtype='bfloat16', compile=False, seed=1337,
)
