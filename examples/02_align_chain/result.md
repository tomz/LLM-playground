# 02 — SFT → RM → DPO alignment chain: result

Recorded on **NVIDIA GeForce RTX 5060 Ti**, built on the example 01 base checkpoint.

## Summary

| stage | first-10 loss | last-10 loss |
|---|--:|--:|
| SFT (200 steps, lr=1e-4) | 9.418 | 0.001 |
| RM  (200 steps, lr=5e-5) | 0.665 | 0.000 |
| DPO (200 steps, lr=5e-6, β=0.1) | 0.672 | 0.286 |

## Held-out test prompts (10 characters)

| char | gold | base | SFT | DPO |
|---|---|---|---|---|
| Juliet | _Romeo and Juliet_ |   `but hear not am in your then w` | ✓ `Romeo and Juliet as a most` | ✓ `Romeo and Juliet as a most sta` |
| Othello | _Othello_ |   `but did not but that which, bu` | ✓ `Othello of water, on the me a ` | ✓ `Othello of water, oning, is ve` |
| Menenius | _Coriolanus_ |   `may kind, me before I clate.` | ✓ `Coriolanus of your quake a qua` | ✓ `Coriolanus of your qual it sti` |
| Hamlet | _Hamlet_ |   `` | ✓ `Hamlet much` | ✓ `Hamlet much` |
| Romeo | _Romeo and Juliet_ |   `` | ✓ `Romeo and Juliet as live hiss ` | ✓ `Romeo and Juliet as live hiss ` |
| Ophelia | _Hamlet_ |   `or knife rouse, a man, a!` | ✓ `Hamlet to at gentle sir which` | ✓ `Hamlet to at gentle sir which` |
| Mercutio | _Romeo and Juliet_ |   `` | ✓ `Romeo and Juliet as diet of lo` | ✓ `Romeo and Juliet as diet of lo` |
| Portia | _The Merchant of Venice_ |   `` | ✓ `The Merchant of Venice` | ✓ `The Merchant of Venice` |
| Iago | _Othello_ |   `for of foot be out a noble coo` | ✓ `Othello my embrace, ones!` | ✓ `Othello my embrace, the loss,` |
| Brutus | _Julius Caesar_ |   `but that,, of love two speed f` | ✓ `Julius Caesar condemn'd air,, ` | ✓ `Julius Caesar condemn'd air,, ` |

## Aggregate

| model | exact-match acc | mean RM score |
|---|--:|--:|
| base | 0% | +0.099 |
| SFT  | 100%  | +1.146 |
| DPO  | 100%  | +2.041 |

_Total wall time: 38.9s. Peak GPU memory: 1.02 GiB._
