# 03 — MoE vs dense at matched active params: result

Both models trained on TinyShakespeare shards from example 01,
identical optimizer (AdamW, lr 3e-4 cosine, warmup 50, 1000 steps),
micro-batch 8 × seq 256.

## Headline

| model | params | tokens/sec (last 500) | first-2 mean loss | last-2 mean loss |
|---|--:|--:|--:|--:|
| dense | 4.01 M | 41,132 | 7.648 | 4.425 |
| moe | 4.89 M | 32,355 | 7.925 | 4.686 |

## Loss curves (step, loss)

### dense

```
    0   8.376
   50   6.921
  100   5.911
  150   5.712
  200   5.555
  250   5.100
  300   5.178
  350   5.096
  400   4.775
  450   4.897
  500   4.841
  550   4.577
  600   4.712
  650   4.681
  700   4.456
  750   4.589
  800   4.574
  850   4.367
  900   4.512
  950   4.518
  999   4.332
```
### moe

```
    0   8.428
   50   7.422
  100   6.315
  150   5.990
  200   5.804
  250   5.346
  300   5.379
  350   5.305
  400   4.989
  450   5.109
  500   5.078
  550   4.805
  600   4.942
  650   4.934
  700   4.688
  750   4.838
  800   4.851
  850   4.613
  900   4.778
  950   4.797
  999   4.575
```

## MoE diagnostics

### Aux-loss trajectory (z-loss + load-balance) — layer 0

```
    0  0.0124
   50  0.0111
  100  0.0102
  150  0.0107
  200  0.0104
  250  0.0106
  300  0.0105
  350  0.0106
  400  0.0106
  450  0.0106
  500  0.0106
  550  0.0107
  600  0.0107
  650  0.0107
  700  0.0108
  750  0.0107
  800  0.0108
  850  0.0109
  900  0.0108
  950  0.0109
  999  0.0108
```

### Final per-expert token counts (layer 0, one batch)

| expert | count | utilisation |
|---|--:|--:|
| 0 | 1225 | 29.9% |
| 1 | 859 | 21.0% |
| 2 | 988 | 24.1% |
| 3 | 1024 | 25.0% |

_Total wall time: 162.5s. Peak GPU memory: 0.43 GiB._
