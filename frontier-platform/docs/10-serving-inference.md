# 10 — Serving & Inference

## Stack

- **Engine**: vLLM, TensorRT-LLM, or SGLang (we abstract behind `serving.engine.Engine`).
- **Quantization**: FP8 (Hopper+) for weights+KV; INT4 AWQ/GPTQ for cost-sensitive tiers.
- **KV cache**: paged attention, prefix caching across requests, optional offload to CPU/NVMe.
- **Batching**: continuous batching with token-level scheduling; chunked prefill to prevent decode-starvation.
- **Speculative decoding**: draft model (1B) + target model (70B) for ~2× decode speedup on easy tokens.
- **Parallelism**: TP within node, PP across nodes if model > node HBM. Disaggregated prefill/decode for high-traffic deployments.

## Routing tiers

| Tier   | Model | Hardware             | Target latency | Cost/Mtok |
|--------|-------|----------------------|---------------:|----------:|
| nano   | 1B Q4 | 1× L40S              | TTFT 50ms      | ~$0.05    |
| mid    | 7B FP8| 1× H100              | TTFT 100ms     | ~$0.40    |
| pro    | 70B FP8| 8× H100 TP=8        | TTFT 250ms     | ~$3       |
| max    | 400B  | 64× H100 TP=8 PP=8   | TTFT 600ms     | ~$15      |

## SLOs

P50 TTFT, P99 TTFT, inter-token latency, throughput (tok/s/GPU), goodput (tok/s honoring SLO). Page at >2× SLO for 5 min.
