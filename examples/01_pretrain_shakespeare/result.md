# 01 — TinyShakespeare pretraining: result

Recorded from one real run on an **RTX 3050 (8 GB, sm_86)** using
frontier-platform components end-to-end.

## Summary

| metric                          | value |
|--------------------------------|------:|
| parameters                     | 16.13 M |
| training steps                 | 3000 |
| tokens seen                    | 12.29 M |
| first-100-step mean loss       | 6.713 |
| last-100-step mean loss        | 0.148 |
| reduction                      | +6.566 |
| training wall time             | 407.3 s |
| total wall time (incl. download/BPE/gen) | 412.3 s |
| peak GPU memory                | 1.44 GiB |

## Generated sample (first 1000 chars)

The sample starts from the prompt `ROMEO:` with `temperature=0.8, top_p=0.9`.

```
ROMEO:
Their wrath, to desire us!

AUFIDIUS:
I'll not to't,
I'll none but like the, our danger.
I can do with thee, who's my heavens,
If thou please thee here, if thou know'st it.

AUFIDIUS:
I can tell thee aughty, if thou darest.

Vouch:

First Keeper:

Alless I will. What, Signior Menenius,
For'tus dispatch:

All whom thou perjury the world,
Which must be so arrived?

MENENIUS:
Leave her squer of her lips,

I cannot holding to, between herane some other.
But now, for thy mind does consider, be overgracious,
And blame you not to, the princesset
To some other promotion; and if thou shalt
I take the public great that of your most fit
Of the key-crowing sisterhood of mine arm,
Wherewith the thing you mountitows and Rome,
I hear him swo that were his hold,
To you, to make her voymery found
To what we do have.

Provost:
You will, for the world sufficulte you--' the time--

MENENIUS:
Nay, give me thy head with a wife:
Leave off topplow, now in the ground as well were to me as

Hortensio, Cl
```

Full 500-token sample in `out/sample.txt`.
