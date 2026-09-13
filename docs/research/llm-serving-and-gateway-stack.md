# The LLM Serving & Gateway Stack — Gateways, Engines & Distributed Serving

**Source:** Web research digest (vendor docs, benchmarks, comparison blogs, project repos)
**Type:** Landscape survey / decision guide
**One-line:** A map of the self-hosted LLM serving stack — **gateways/routers**
(LiteLLM, Bifrost, Portkey…), **inference engines** (vLLM, SGLang, TensorRT-LLM…),
and **distributed orchestration** (llm-d, NVIDIA Dynamo) — with when-to-use-what.

---

## The one-paragraph version

These tools are constantly confused but live in **three distinct layers**. A
**gateway/router** (LiteLLM, Bifrost, Portkey, OpenRouter) decides *which model
to call, what it costs, auth, fallbacks* — it does **not** run weights. An
**inference engine** (vLLM, SGLang, TensorRT-LLM, TGI, LMDeploy) actually loads
the model and generates tokens on the GPU. A **distributed-serving orchestrator**
(llm-d, NVIDIA Dynamo, KServe, Ray Serve) scales an engine across a GPU fleet
with KV-cache-aware routing and prefill/decode disaggregation. The 2025–26 SOTA
self-hosted stack is roughly **gateway → llm-d/Dynamo → vLLM/SGLang → GPUs**.
They compose; they don't compete.

---

## The layer cake

```
   Your app
      │
   ┌──┴───────────────┐
   │  GATEWAY/ROUTER  │  ← LiteLLM, Bifrost, Portkey, OpenRouter, Helicone
   │  picks a model,  │     "Which model? Auth? Cost? Fallback? Caching?"
   │  tracks cost     │     Does NOT run weights.
   └──┬───────────────┘
      │
   ┌──┴───────────────┐
   │ ORCHESTRATION    │  ← llm-d, NVIDIA Dynamo, KServe, Ray Serve, NIM
   │ scale across     │     "Many GPUs/nodes, KV-cache-aware routing,
   │ many GPUs/pods   │      prefill/decode disaggregation"
   └──┬───────────────┘
      │
   ┌──┴───────────────┐
   │ INFERENCE ENGINE │  ← vLLM, SGLang, TensorRT-LLM, TGI, LMDeploy
   │ runs the weights,│     "PagedAttention, batching, the actual
   │ generates tokens │      token generation on the GPU"
   └──────────────────┘
```

The key insight: **"vLLM vs LiteLLM" is a category error.** You run a gateway
(LiteLLM/Bifrost) *in front of* a vLLM deployment. The right comparisons are
*within* a layer.

---

## Layer 1 — Gateways / Routers (LiteLLM and its alternatives)

**LiteLLM** is the popular default: one OpenAI-compatible API to 100+ providers,
SDK + proxy server, cost tracking, fallbacks, budgets. It's genuinely good and
ubiquitous — but it has known pain points that drive teams to alternatives.

### Why people leave LiteLLM
- **Performance under load** — Python/GIL-bound. Independent benchmarks measure
  ~**38% higher p95 latency** vs direct calls; GitHub issues report latency
  spikes, memory growth over time, instability at scale (#12067, #6345, #13546).
  Teams report scaling to ~20 proxy pods to cope.
- **Operational weight** — proxy + Postgres + Redis gets heavy.
- **Not a security product** — auth/audit/zero-trust are bolt-ons.
- *Caveat:* LiteLLM is responding — a new sidecar architecture claims
  sub-millisecond overhead, so some gaps are narrowing.

### Alternatives, grouped by *why* they're better

| Pick | Category | Why it beats LiteLLM | Caveat |
|------|----------|----------------------|--------|
| **Bifrost** (Go, Maxim AI) | Drop-in OSS gateway | Raw throughput — vendor claims **~9–50× faster, ~11 µs overhead @ 5,000 RPS**, 100% success under load | "50×" is vendor-published; independent reruns confirm *meaningfully* faster, distrust the exact multiplier |
| **OpenRouter** | Managed/hosted | **Zero-ops**, one key, instant access to newest models | Pay-per-token markup; not self-hosted |
| **Portkey** | Production gateway | Batteries-included: caching, retries, cross-provider fallbacks, guardrails, prompt mgmt, analytics | Managed tiers for full features |
| **Helicone** | Observability-first | Best-in-class tracing/cost analytics; ~1-line adoption | Lighter on routing/governance |
| **Cloudflare AI Gateway** | Edge | Edge caching, rate-limits, analytics, minimal ops, global low latency | Less control than self-host |
| **Pomerium / Kong** | Security/zero-trust | Identity-aware access, enterprise rate-limiting, audit | Heavier; infra-oriented |
| **new-api / one-api** | Self-host multi-tenant | OSS key mgmt, quotas, billing dashboards | Community-driven |

**Verdict:** No single tool strictly dominates — LiteLLM's **provider breadth +
ubiquity** keep it the default. The most common "direct upgrade" is **Bifrost**
(same idea, higher performance ceiling). Pattern: keep LiteLLM for breadth in
dev/prototyping, graduate to Bifrost/Portkey/Cloudflare for production scale.

---

## Layer 3 — Inference Engines (run the model)

| Engine | Backed by | Superpower | Best when |
|--------|-----------|------------|-----------|
| **vLLM** | OSS (UC Berkeley origin, huge community) | **PagedAttention**, broadest model support, de-facto default | Safe, flexible, well-supported standard |
| **SGLang** | OSS (fast-moving) | **RadixAttention** (KV-cache reuse for shared prefixes); often **leads throughput** | Max throughput, agents/structured gen, prefix reuse |
| **TensorRT-LLM** | NVIDIA | Most out of NVIDIA HW (lowest latency on H100/H200) | NVIDIA-only, latency-critical |
| **TGI** | Hugging Face | Polished, HF-ecosystem integration | Living in the HF ecosystem, simple ops |
| **LMDeploy** | OpenMMLab | Strong quantization + speed | Quantized serving, smaller GPUs |

**2025 trend: the engines are converging.** SGLang leads throughput, TRT-LLM
leads NVIDIA-specific latency, but **vLLM adopted the best ideas** (incl.
disaggregated serving) and stays the **default** on model coverage + community
gravity. For most teams: **start with vLLM**; move to SGLang/TRT-LLM only if
benchmarks on *your* workload justify it.

---

## Layer 2 — Distributed Serving / Orchestration (run at scale)

A single vLLM instance serves one model on one node well — production fleets
need smart routing across many GPUs. This is the new frontier.

### llm-d
- **What:** Kubernetes-native distributed inference stack **built around vLLM**.
  Launched **May 2025 by Red Hat + Google + IBM** — serious industry backing.
- **Key ideas:**
  - **Disaggregated serving** — split **prefill** (prompt processing,
    compute-bound) from **decode** (token generation, memory-bound) into
    separate pods so each scales independently → better GPU utilization.
  - **KV-cache-aware routing** — route to the GPU that already holds the
    relevant cache (via the K8s **Gateway API Inference Extension**), not
    round-robin.
  - GPU-aware scheduling, hybrid-cloud ready.
- **Best for:** teams already on Kubernetes self-hosting OSS models at scale
  cost-effectively. Young (2025), K8s-heavy.

### NVIDIA Dynamo
- NVIDIA's datacenter-scale inference framework (spiritual successor to Triton
  for the LLM era). Also does **disaggregated prefill/decode**, KV-cache-aware
  routing, smart KV cache manager. **Engine-agnostic** (TRT-LLM, vLLM, SGLang).
- **Best for:** large NVIDIA GPU fleets wanting NVIDIA's own orchestration.

### Others
- **KServe** — mature K8s model serving (broader than LLMs).
- **Ray Serve** — Python-native scaling, flexible custom pipelines.
- **NVIDIA NIM** — packaged, supported containers (TRT-LLM/vLLM inside) for
  enterprises wanting a vendor-supported box.

---

## The modern SOTA self-hosted stack

```
LiteLLM / Bifrost   →   llm-d (or Dynamo)   →   vLLM (or SGLang)   →   GPUs
  gateway:                orchestration:           engine:
  routing, cost,          disaggregation,          PagedAttention,
  multi-provider          KV-aware routing         batching
```

The layers **complement** rather than compete: the gateway decides
*what/where/cost*; the engine does *token generation*; llm-d/Dynamo make the
engine scale across a GPU fleet efficiently.

---

## Decision cheat-sheet

| Your situation | Reach for |
|----------------|-----------|
| LiteLLM too slow / GIL bottleneck | **Bifrost** |
| Don't want to run *any* infra | **OpenRouter** |
| Need routing + caching + analytics, batteries-included | **Portkey** |
| Mostly need visibility / cost tracking | **Helicone** |
| Edge caching + rate-limits, minimal ops | **Cloudflare AI Gateway** |
| Security, SSO, zero-trust, audit | **Pomerium / Kong** |
| Self-host multi-tenant w/ quotas & billing | **new-api / one-api** |
| Default inference engine | **vLLM** |
| Maximum throughput / prefix-cache reuse | **SGLang** |
| Lowest latency on NVIDIA HW | **TensorRT-LLM** |
| Distributed self-hosted serving on K8s | **llm-d** (or **Dynamo** on big NVIDIA fleets) |

---

## Relevance to LLM-playground

- The **frontier-platform** project (1B–500B+) is the natural home for a
  vLLM-based serving path; **llm-d**'s prefill/decode disaggregation is the
  "ideal hardware tier" technique to track for multi-GPU/multi-node serving.
- A thin **gateway** (LiteLLM for breadth, or Bifrost if we hit Python overhead)
  in front of a local vLLM endpoint gives a clean OpenAI-compatible surface for
  eval harnesses and the smaller projects (nanogpt-edu, midgpt) to share.
- **Harvest candidates:** (1) minimal vLLM server + gateway for local eval;
  (2) PagedAttention vs RadixAttention writeup; (3) prefill/decode
  disaggregation explainer sized to minimal vs ideal GPU envelopes.

---

## Sources

1. Maxim AI — Bifrost vs LiteLLM benchmarks: https://www.getmaxim.ai/bifrost/resources/benchmarks
2. markaicode — LiteLLM latency benchmark (38% p95 overhead): https://markaicode.com/benchmarks/litellm-latency-benchmark/
3. LiteLLM — sub-millisecond proxy overhead (sidecar): https://docs.litellm.ai/blog/sub-millisecond-proxy-overhead
4. dev.to — When LiteLLM becomes a bottleneck: https://dev.to/therealmrmumba/when-litellm-becomes-a-bottleneck-exploring-gateway-alternatives-3a4h
5. awesome-ai-gateway (50+ gateways compared): https://github.com/cuihuan/awesome-ai-gateway
6. Pomerium — LiteLLM alternatives / best LLM gateways 2025: https://www.pomerium.com/blog/litellm-alternatives
7. Red Hat — llm-d Kubernetes-native distributed inferencing: https://developers.redhat.com/articles/2025/05/20/llm-d-kubernetes-native-distributed-inferencing
8. Google Cloud — Enhancing vLLM for distributed inference with llm-d: https://cloud.google.com/blog/products/ai-machine-learning/enhancing-vllm-for-distributed-inference-with-llm-d
9. llm-d project: https://llm-d.ai/  ·  https://github.com/llm-d/llm-d/
10. JarvisLabs — vLLM vs SGLang vs TensorRT-LLM serving benchmark: https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison
11. Inference Engineering — engine comparison + decision framework: https://inferenceengineering.tech/learn/vllm-vs-sglang-vs-tensorrt-llm/
12. getmaxim — Top 5 LLM gateways for scaling (Bifrost/Helicone/LiteLLM/OpenRouter/TrueFoundry): https://www.getmaxim.ai/articles/top-5-llm-gateways-for-scaling-ai-applications-in-2025/
