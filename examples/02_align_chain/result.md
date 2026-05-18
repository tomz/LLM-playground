# 02 — SFT → RM → DPO alignment chain: result

Recorded on **RTX 3050**, built on the example 01 base checkpoint.

## Summary

| stage | first-10 loss | last-10 loss |
|---|--:|--:|
| SFT (200 steps, lr=1e-4) | 5.652 | 0.001 |
| RM  (200 steps, lr=5e-5) | 0.570 | 0.000 |
| DPO (200 steps, lr=5e-6, β=0.1) | 0.633 | 0.087 |

## Held-out test prompts (10 characters)

| char | gold | base | SFT | DPO |
|---|---|---|---|---|
| Juliet | _Romeo and Juliet_ |   `too, how he will hence! king a` | ✓ `Romeo and Juliet` | ✓ `Romeo and Juliet justice` |
| Othello | _Othello_ |   `but not but you can give my la` | ✓ `Othello` | ✓ `Othello stand stand` |
| Menenius | _Coriolanus_ |   `but thou law for thee! thou it` | ✓ `Coriolanus` | ✓ `Coriolanus` |
| Hamlet | _Hamlet_ |   `too, that he'll be my oath` | ✓ `Hamlet` | ✓ `Hamlet this this this this` |
| Romeo | _Romeo and Juliet_ |   `too, how is it did fiven` | ✓ `Romeo and Juliet` | ✓ `Romeo and Juliet` |
| Ophelia | _Hamlet_ |   `no thou didst fivet to the wor` | ✓ `Hamlet` | ✓ `Hamlet this say say comes come` |
| Mercutio | _Romeo and Juliet_ |   `too, how is wary too,` | ✓ `Romeo and Juliet` | ✓ `Romeo and Juliet` |
| Portia | _The Merchant of Venice_ |   `what is my head the worth is i` | ✓ `The Merchant of Venice` | ✓ `The Merchant of Venice` |
| Iago | _Othello_ |   `what is your fellow of your? a` | ✓ `Othello` |   `O this this` |
| Brutus | _Julius Caesar_ |   `too, like a truth, full, but a` | ✓ `Julius Caesar this` | ✓ `Julius Caesar;` |

## Aggregate

| model | exact-match acc | mean RM score |
|---|--:|--:|
| base | 0% | -2.462 |
| SFT  | 100%  | +8.758 |
| DPO  | 90%  | +0.121 |

_Total wall time: 42.8s. Peak GPU memory: 0.93 GiB._
