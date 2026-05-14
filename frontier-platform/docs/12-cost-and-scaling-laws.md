# 12 — Cost & Scaling Laws

## Chinchilla-optimal

For compute-optimal training: tokens ≈ 20 × params. Modern frontier runs are *over-trained* (100–300× params) because inference cost dominates lifetime cost.

Compute (FLOPs) ≈ 6 × N(params) × D(tokens).

## Cost reference (rentable H100 @ ~$2/GPU-hr; assume 50% MFU)

| Model | Params | Tokens | FLOPs    | GPU-hours   | Cost (rent) |
|-------|-------:|-------:|---------:|------------:|------------:|
| 1B    | 1.2e9  | 1e12   | 7.2e21   | ~2,000      | ~$4k        |
| 7B    | 6.7e9  | 2e12   | 8.0e22   | ~22,000     | ~$45k       |
| 70B   | 7.0e10 | 5e12   | 2.1e24   | ~580,000    | ~$1.2M      |
| 400B  | 4.0e11 | 1.5e13 | 3.6e25   | ~10,000,000 | ~$20M       |

Plus: data pipeline (~10% of training cost), eval & RLHF (~15%), failed runs (×1.5–3 multiplier — first runs always fail), salaries, datacenter buildout if owning hardware.

## Real program budget (annual)

- Small lab (1–7B models, OSS release): $5M–$20M
- Mid lab (70B-class): $50M–$200M
- Frontier lab (400B+, multimodal, agents): $500M–$5B

## Sanity gates

Before committing to a full run:
1. Scaling-law sweep: train 5 sizes (50M, 150M, 500M, 1.5B, 5B) on identical data, fit L(N, D), extrapolate.
2. Lit-rate sweep on the smallest 2 sizes.
3. Data ablation: train two 1B models on candidate mixes, compare evals.
4. Stability dry-run: 10% of full token budget at full scale; abort if loss curve diverges from extrapolation.
