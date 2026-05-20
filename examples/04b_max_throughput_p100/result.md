# 04b — Max-throughput benchmark on Tesla P100: result

Recorded from one real run on a **Tesla P100-PCIE (16 GB, sm_60)**.
Selective activation checkpointing, batch auto-tuned to fit VRAM.
Compute in **fp16** (autocast + GradScaler); master weights + AdamW state in fp32.
Pascal has no bf16 and no tensor cores; throughput comes from packed-fp16 on the FP32 ALUs.

## Configuration

| field | value |
|---|--:|
| parameters | 503.38 M |
| seq_len | 1024 |
| micro_batch (auto) | 32 |
| tokens / step | 32,768 |
| training steps | 1500 |
| optimizer | AdamW, peak_lr=3e-4, warmup=100, cosine |
| precision | fp16 autocast + GradScaler (fp32 master) |
| activation checkpointing | selective (per Block) |

## Throughput

| metric | value |
|---|--:|
| training wall time | 37359.5 s |
| total wall time | 38102.8 s |
| tokens / second | 1,316 |
| achieved TFLOPS (6·N·tps) | 3.97 |
| theoretical peak (fp16, P100) | 18.7 TFLOPS |
| **MFU** | **21.2%** |

## GPU saturation (`nvidia-smi -lms 500`)

| stat | utilization.gpu |
|---|--:|
| mean | **99.9%** |
| P50 | 100.0% |
| P95 | 100.0% |
| peak memory | 13.51 GiB / 16.00 GiB |

## Loss

| window | mean loss |
|---|--:|
| first 50 steps | 6.800 |
| last 50 steps | 0.007 |
| reduction | +6.793 |

## Generated sample (200 tokens from `ROMEO:`)

```
ROMEO:
Come, come, you mock me; this is not the way
To win our daughter.

QUEEN ELIZABETH:
There is no other way
Unless thou couldst put on some other shape,
And not be Richard that hath done all this.

KING RICHARD III:
Say that I did all this for love of her.

QUEEN ELIZABETH:
Nay, then indeed she cannot choose but hate thee,
Having bought love with such a bloody spoil.

KING RICHARD III:
Look, what is done cannot be now amended:
Men shall deal unadvisedly sometimes,
Which after hours give leisure to repent.
If I did take the kingdom from your sons,
To make amends, Ill give it to your daughter.
If I have kill'd the issue of your womb,
To quicken your increase, I will beget
Mine issue of your blood upon your daughter
A grandam's name is little less in love
Than
```

Full sample in `out/sample.txt`. Raw nvidia-smi log in `out/nvsmi.log`.
