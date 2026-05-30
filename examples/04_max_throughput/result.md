# 04 — Max-throughput benchmark: result

Recorded from one real run on a **NVIDIA GeForce RTX 5060 Ti** (15.5 GiB).
Selective activation checkpointing, batch auto-tuned to fit VRAM.
Compute in **bf16** (autocast); master weights + AdamW state in fp32.

## Configuration

| field | value |
|---|--:|
| parameters | 381.73 M |
| seq_len | 1024 |
| micro_batch (auto) | 40 |
| tokens / step | 40,960 |
| training steps | 1500 |
| optimizer | AdamW, peak_lr=3e-4, warmup=100, cosine |
| precision | bf16 autocast (fp32 master) |
| activation checkpointing | selective (per Block) |

## Throughput

| metric | value |
|---|--:|
| training wall time | 7001.6 s |
| total wall time | 7043.2 s |
| tokens / second | 8,775 |
| achieved TFLOPS (6·N·tps) | 20.10 |
| theoretical peak (fp16) | 28.48 TFLOPS |
| **MFU** | **70.6%** |

_Theoretical peak derived from device: cc12.0 36SM x 128 x 3.09GHz._

## GPU saturation (`nvidia-smi -lms 500`)

| stat | utilization.gpu |
|---|--:|
| mean | **99.7%** |
| P50 | 100.0% |
| P95 | 100.0% |
| peak memory | 12.82 GiB / 15.48 GiB |

## Loss

| window | mean loss |
|---|--:|
| first 50 steps | 6.811 |
| last 50 steps | 0.005 |
| reduction | +6.806 |

## Generated sample (200 tokens from `ROMEO:`)

```
ROMEO:
Is the day so young?

BENVOLIO:
But new struck nine.

ROMEO:
Ay me! sad hours seem long.
Was that my father that went hence so fast?

BENVOLIO:
It was. What sadness lengthens Romeo's hours?

ROMEO:
Not having that, which, having, makes them short.

BENVOLIO:
In love?

ROMEO:
Out--

BENVOLIO:
Of love?

ROMEO:
Out of her favour, where I am in love.

BENVOLIO:
Alas, that love, so gentle in his view,
Should be so tyrannous and rough in proof!

ROMEO:
Alas, that love, whose view is muffled still,
Should, without eyes, see pathways to his will!
Where shall we dine? O me! What fray was here?
Yet tell me not, for I have heard it all.
Here's much
```

Full sample in `out/sample.txt`. Raw nvidia-smi log in `out/nvsmi.log`.
