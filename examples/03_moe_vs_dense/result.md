# 03 — MoE vs dense at matched active params: result

Both models trained on TinyShakespeare shards from example 01,
identical optimizer (AdamW, lr 3e-4 cosine, warmup 50, 1000 steps),
micro-batch 8 × seq 256.

## Headline

| model | params | tokens/sec (last 500) | first-2 mean loss | last-2 mean loss |
|---|--:|--:|--:|--:|
| dense | 4.01 M | 127,183 | 7.651 | 4.441 |
| moe | 4.89 M | 39,190 | 7.867 | 4.631 |

## Loss curves (step, loss)

### dense

```
    0   8.354
   50   6.949
  100   5.954
  150   5.736
  200   5.559
  250   5.102
  300   5.181
  350   5.106
  400   4.787
  450   4.919
  500   4.870
  550   4.592
  600   4.724
  650   4.708
  700   4.451
  750   4.598
  800   4.597
  850   4.358
  900   4.517
  950   4.537
  999   4.345
```
### moe

```
    0   8.355
   50   7.379
  100   6.267
  150   5.938
  200   5.772
  250   5.298
  300   5.328
  350   5.270
  400   4.944
  450   5.064
  500   5.031
  550   4.757
  600   4.901
  650   4.892
  700   4.632
  750   4.789
  800   4.801
  850   4.556
  900   4.715
  950   4.748
  999   4.515
```

## MoE diagnostics

### Aux-loss trajectory (z-loss + load-balance) — layer 0

```
    0  0.0019
   50  0.0012
  100  0.0003
  150  0.0005
  200  0.0007
  250  0.0007
  300  0.0007
  350  0.0008
  400  0.0008
  450  0.0007
  500  0.0009
  550  0.0008
  600  0.0009
  650  0.0010
  700  0.0010
  750  0.0010
  800  0.0011
  850  0.0011
  900  0.0010
  950  0.0011
  999  0.0011
```

### Final per-expert token counts (layer 0, one batch)

| expert | count | utilisation |
|---|--:|--:|
| 0 | 1049 | 25.6% |
| 1 | 993 | 24.2% |
| 2 | 985 | 24.0% |
| 3 | 1069 | 26.1% |

_Total wall time: 68.7s. Peak GPU memory: 0.43 GiB._
