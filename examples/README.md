# Examples

End-to-end pipelines that exercise the `frontier-platform` code on a real GPU.
The scripts are device-agnostic — example 04 derives its theoretical fp16 peak
(for the MFU denominator) from the live device at runtime. Examples are
currently run on the **RTX 5060 Ti (16 GB, sm_120)**; example `04b` is a
P100-specific counterpart kept for cross-card comparison.

| # | Example                                                  | What it shows                                   |
|---|----------------------------------------------------------|-------------------------------------------------|
| 1 | [`01_pretrain_shakespeare`](01_pretrain_shakespeare/)    | Real BPE tokenizer + shard pipeline + 3 k-step pretrain of a 12 M-param transformer on TinyShakespeare, then `TorchEngine` generation. |
| 2 | [`02_align_chain`](02_align_chain/)                      | SFT → reward model → DPO on a synthetic Shakespeare-character QA task, built on the example 01 checkpoint. Side-by-side base / SFT / DPO comparison. |
| 3 | [`03_moe_vs_dense`](03_moe_vs_dense/)                    | Dense vs 4-expert top-2 MoE at matched active params, on the example 01 shards. Loss curves, tokens/sec, per-expert utilisation. |
| 4 | [`04_max_throughput`](04_max_throughput/)                | Batch-autotuned ~500 M-param transformer (bf16 autocast + activation ckpt) pushing the GPU to near-saturation, with background `nvidia-smi` sampling and device-derived MFU. |
| 4b | [`04b_max_throughput_p100`](04b_max_throughput_p100/)   | P100-specific counterpart of 04 (sm_60, dedicated `.venv-p100` with torch 2.4.1+cu121). |

## Running

```bash
cd 01_pretrain_shakespeare && bash run.sh    # required first; produces the base checkpoint
cd ../02_align_chain        && bash run.sh
cd ../03_moe_vs_dense       && bash run.sh
cd ../04_max_throughput     && bash run.sh
```

Each `run.sh` pins `CUDA_VISIBLE_DEVICES` to a specific GPU UUID and uses the
`frontier-platform/.venv` interpreter (torch 2.11.0+cu130, which has `sm_120`
kernels for the 5060 Ti). The P100 example uses `.venv-p100` instead.

> **Note:** the `CUDA_VISIBLE_DEVICES` UUIDs in `run.sh` are host-specific.
> Re-verify them against `nvidia-smi -L` after any GPU/driver change.

All generated artefacts go to `<example>/out/` (gitignored). Each example
checks in a `result.md` produced from one real run so reviewers can compare.
