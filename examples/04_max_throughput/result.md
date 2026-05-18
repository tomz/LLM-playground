# 04 — Max-throughput benchmark: result

Recorded from one real run on an **RTX 3050 (8 GB, sm_86)**.
Selective activation checkpointing, batch auto-tuned to fit VRAM.
Compute in **bf16** (autocast); master weights + AdamW state in fp32.

## Configuration

| field | value |
|---|--:|
| parameters | 381.73 M |
| seq_len | 1024 |
| micro_batch (auto) | 8 |
| tokens / step | 8,192 |
| training steps | 1500 |
| optimizer | AdamW, peak_lr=3e-4, warmup=100, cosine |
| precision | bf16 autocast (fp32 master) |
| activation checkpointing | selective (per Block) |

## Throughput

| metric | value |
|---|--:|
| training wall time | 3309.7 s |
| total wall time | 3362.2 s |
| tokens / second | 3,713 |
| achieved TFLOPS (6·N·tps) | 8.50 |
| theoretical peak (fp16, 3050) | 9.05 TFLOPS |
| **MFU** | **94.0%** |

## GPU saturation (`nvidia-smi -lms 500`)

| stat | utilization.gpu |
|---|--:|
| mean | **99.8%** |
| P50 | 100.0% |
| P95 | 100.0% |
| peak memory | 6.47 GiB / 8.00 GiB |

## Loss

| window | mean loss |
|---|--:|
| first 50 steps | 6.905 |
| last 50 steps | 0.082 |
| reduction | +6.824 |

## Generated sample (200 tokens from `ROMEO:`)

```
ROMEO:
At thy choice, I think there was the duke.

DUKE OF YORK:
You wrong me not, nor no more to thy oath.

DUKE OF YORK:
My lord, I do beseech you for your grace.

DUKE OF YORK:
I will be gone, sir, to do't.

DUKE OF YORK:
What is it, then?

DUKE OF YORK:
No, almost a word.
Gentlemen, we now show it.
You'll tell you what a letter?

DUKE OF AUMERLE:
You shall be fond of the matter you give.

DUKE OF AUMERLE:
Good night, thou hast; thy son is beloved and light.

DUKE OF AUMERLE:
Why, then, what's thy name, I pray,
That I should live to thy father's house,
And presently your subject to thy drift,
If thou wilt, perform this noble duke,
Or as I
```

Full sample in `out/sample.txt`. Raw nvidia-smi log in `out/nvsmi.log`.
