# 01 — TinyShakespeare pretraining: result

Recorded from one real run on a **NVIDIA GeForce RTX 5060 Ti** using
frontier-platform components end-to-end.

**This-month's-harvest features enabled** (config-gated knobs in `platform.*`):
**Muon** optimizer (Newton-Schulz-orthogonalized 2D hidden weights; AdamW for
embeddings/lm_head/MTP heads/norms), **QK-norm** attention stabilizer, and
**Multi-Token Prediction** (2 auxiliary heads, train-only — they account for the
param bump vs the vanilla ~16 M baseline and are discarded at inference).

## Summary

| metric                          | value |
|--------------------------------|------:|
| parameters                     | 19.27 M |
| training steps                 | 3000 |
| tokens seen                    | 12.29 M |
| first-100-step mean loss       | 8.450 |
| last-100-step mean loss        | 1.035 |
| reduction                      | +7.415 |
| training wall time             | 315.5 s |
| total wall time (incl. download/BPE/gen) | 321.3 s |
| peak GPU memory                | 1.72 GiB |

## Generated sample (first 1000 chars)

The sample starts from the prompt `ROMEO:` with `temperature=0.8, top_p=0.9`.

```
ROMEO:
This hand mayil, if thou greet some head to mine there!
Now Tybalt shall infect him. But then, friend, to make
Where what our lords; we mean, not we stand
His glory conjure like.

LADY CAPULET:
Nay, but, thou art none: therefore hang her wild,
But not to speak in a little sing's curse
For some new is parceier than any king.
The skill thou, and Romans: one death,
For then of death: therefore wilt, both fain too
My parliament am light and merited
My parliament be set'd only's death.

PARIS:
Then let us right.
But say not so Richard, if we may be great
The mummaster with death, part not speak no.

DUKE VINCENTIO:
I'll prove more cause.
But that Caius any of the world doth sounds.
So that we this alliance as you we thought,
But only he may butoriolanus knew.

CAPULET:
Pray you, believe he say to shame of you.
Farewell, my lord, I'll keep him I love.

DUKE VINCENTIO:
I am truly Hastings, if I was poet.
Farewell, my lord, let us both post rage out
In every one I part of friends or two
```

Full 500-token sample in `out/sample.txt`.
