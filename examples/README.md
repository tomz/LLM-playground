# Examples

Three end-to-end pipelines that exercise the `frontier-platform` code on a real GPU
(targeted at an RTX 3050, 8 GB, sm_86).

| # | Example                                                  | What it shows                                   | Wall time (3050) |
|---|----------------------------------------------------------|-------------------------------------------------|------------------|
| 1 | [`01_pretrain_shakespeare`](01_pretrain_shakespeare/)    | Real BPE tokenizer + shard pipeline + 3 k-step pretrain of a 12 M-param transformer on TinyShakespeare, then `TorchEngine` generation. | ~5 min |
| 2 | [`02_align_chain`](02_align_chain/)                      | SFT → reward model → DPO on a synthetic Shakespeare-character QA task, built on the example 01 checkpoint. Side-by-side base / SFT / DPO comparison. | ~3 min |
| 3 | [`03_moe_vs_dense`](03_moe_vs_dense/)                    | Dense vs 4-expert top-2 MoE at matched active params, on the example 01 shards. Loss curves, tokens/sec, per-expert utilisation. | ~5 min |

## Running

```bash
cd 01_pretrain_shakespeare && bash run.sh    # required first; produces the base checkpoint
cd ../02_align_chain        && bash run.sh
cd ../03_moe_vs_dense       && bash run.sh
```

Each `run.sh` pins `CUDA_VISIBLE_DEVICES` to the RTX 3050 (the Pascal P100 in
this box is older than the cu130 torch build supports) and uses the
`frontier-platform/.venv` interpreter.

All generated artefacts go to `<example>/out/` (gitignored). Each example
check-ins a `result.md` produced from one real run so reviewers can compare.
