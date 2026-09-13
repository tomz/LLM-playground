# Best local open-weight models for this machine

**Researched:** 2026-09-08 · **Machine:** `i714700k` · **Scope:** local inference,
primarily coding/agents, with general assistance, vision, and writing alternatives.

> **Recommendation:** use the already-installed **Qwen3.8-27B Q4_K_M** as the
> first general/coding model to evaluate across both GPUs. Compare it with the
> already-installed **North Mini Code 1.0** and **KAT-Coder-V2.5-Dev** for agentic
> coding. For a single-GPU service, try **Gemma 4 12B** or **gpt-oss-20b**, or use
> the existing **Qwen3.5-9B** without another download. Keep **Artemis/Orion** for
> creative-writing comparisons, not as presumed upgrades in factual or coding
> reliability. There is no need to download another dozen models before testing
> the strong candidates already present.

**Evidence boundary:** hardware, installed model metadata, and package sizes were
inspected directly. Upstream cards, configuration files, and quantized artifact
metadata were retrieved. **No model was loaded, no generation benchmark was run,
no weights were downloaded, and no service/configuration was changed.** “Best”
means a reasoned shortlist for this hardware, not a measured local leaderboard.
Memory-fit recommendations are estimates; runtime, parser, vision, and tool-use
compatibility still need a smoke test.

## 1. Actual hardware and runtime

| Component | Observed on September 8 | Consequence |
|-----------|-------------------------|-------------|
| CPU | Intel Core i7-14700K, 20 cores / 28 threads | Useful for tokenization, tools, and limited offload; not a substitute for full GPU residency on a large dense model |
| GPUs | 2× NVIDIA GeForce RTX 5060 Ti, compute capability 12.0 | Blackwell-aware runtime required; Ollama documents RTX 5060 Ti support |
| VRAM | 16,311 MiB per GPU; 32,622 MiB total = **31.86 GiB** | Separate allocations, not a single 32GB memory pool |
| Free VRAM at inspection | 15,713 / 15,846 MiB, about **30.82 GiB combined** | Leave margin for the desktop, runtime workspaces, KV/state caches, and other processes |
| GPU topology | `PHB`; negotiated link widths x8 for GPU 0 and x4 for GPU 1 | Transfers traverse the host bridge; no NVLink. Single-GPU operation is attractive when a model fits |
| RAM | About 62 GiB OS-visible; 49 GiB available | Some CPU offload is feasible, but disk/model size is not the entire RAM budget |
| Swap | 8 GiB total, 7.3 GiB occupied | Historical swap occupancy alone is not proof of current thrashing; do not count swap as useful inference memory |
| Disk | About 2.8 TiB available | Storage is not the limiting resource |
| Runtime | Ollama **0.33.1** installed and responding; `ollama ps` empty | Start here rather than replacing the serving stack |

Both cards were idling at 2.5 GT/s when inspected. NVIDIA reported maximum link
generations 5/3, whereas the individual PCI devices expose Gen5 capability in
sysfs. This does **not** establish sustained path bandwidth. The x8/x4 widths and
host-bridge topology are observations; a motherboard bottleneck diagnosis or
specific multi-GPU slowdown would require further measurement. No P2P bandwidth
or tensor-parallel speed claim is made. [1, 11]

**Practical envelope:** one 16GB GPU is comfortable for many 8–14B Q4/Q5 models
and selected compact MoEs. Both GPUs can plausibly keep the shortlisted 27–35B
Q4 models resident at modest context. Full BF16 weights for a 27–35B model do not
fit in the combined VRAM; use the quantized artifacts, not an upstream BF16
quickstart copied without modification.

## 2. Shortlist by job

**GiB below means bytes / 2³⁰.** “Installed” sizes are Ollama package totals;
“artifact” sizes describe the specific upstream GGUF and may exclude projectors
or drafters. These are **not measured peak VRAM**. Initial context recommendations
assume one request at a time and modest output length; they are starting points,
not architectural limits or equivalent settings to vendor benchmarks.

| Priority / job | Model and weight choice | Verified size / presence | Recommended placement and initial context | Why / caveat |
|----------------|-------------------------|--------------------------|------------------------------------------|--------------|
| **Default quality candidate** | **Qwen3.8-27B**, installed Q4_K_M + MTP package | **16.52 GiB installed** | Both GPUs; start **16K**, then 32K | Strong current coding/agent/vision candidate; dense computation and MTP overhead need measurement [2] |
| **Coding specialist challenger** | **North Mini Code 1.0**, 30B-total/3B-active, Q4_K_M | **17.32 GiB installed** | Both GPUs; **16K→32K** | Coding-focused MoE; use its actual tool/reasoning format and do not equate local 500K metadata with validated context [3] |
| **Second coding specialist** | **KAT-Coder-V2.5-Dev**, 35B-total/3B-active, Q4_K_M | **19.92 GiB installed** | Both GPUs; **16K→32K** | Post-trained for tool-based coding; text-only, with less memory headroom than North [4] |
| **Single-GPU multimodal candidate** | **Gemma 4 12B IT**, Q4_K_M; Q5_K_M if preferred after testing | **6.63 / 7.84 GiB** text GGUF; ~0.16 GiB F16 projector; not installed | One GPU; **16K→32K** | Generous headroom compared with a 27B model. Official Ollama Q4 package is listed at 7.6 decimal GB [5, 11, 12] |
| **Single-GPU reasoning/tool candidate** | **gpt-oss-20b**, native **MXFP4** | **11.28 GiB** GGUF; not installed | One GPU; **8K→16K**, then test longer | Designed for 16GB-class deployment; native Harmony formatting is essential; text-only [7] |
| **Fast existing fallback** | **Qwen3.5-9B**, Q4_K_M | **6.14 GiB installed** | One GPU; **16K→32K** | Already available with vision/tools/thinking metadata; no new download required [8] |
| **Additional general/vision agent candidate** | **Muse Glimmer 30B**, official KQuant-17GB | **15.61 GiB** main GGUF + **1.30 GiB** vision projector; optional **1.52 GiB** DFlash drafter; not installed | Both GPUs; **16K→32K** | Purpose-built local agent. Vendor RTX 5090 speedups are not predictions for two 5060 Ti cards [6] |
| **Alternate generalist / math comparison** | **Gemma 4 31B IT**, Q4_K_M | **17.07 GiB** text GGUF + ~1.12 GiB F16 projector; not installed | Both GPUs; **16K→32K** | Official baseline is preferable to assuming a creative derivative retains all instruction/coding quality [5] |
| **Narrow single-GPU experiment** | **Gemma 4 26B-A4B IT QAT**, official Q4_0 | **13.45 GiB** text GGUF + optional **1.11 GiB** vision projector; not installed | Text-only **4K→8K** may fit one GPU; both GPUs are safer | QAT makes this interesting, but runtime buffers can exhaust the small remaining margin. The generic Ollama `gemma4:26b` is a different, larger package [5, 12] |
| **Creative writing** | Existing **Artemis 31B v1.1 Q4_K_M** / **Orion 26B-A4B v1 Q5_K_M** | **18.25 / 17.99 GiB installed** | Both GPUs; **16K→32K** | Creator explicitly prioritizes creativity/entertainment; not evidence of superior coding, factuality, or tool safety [9] |

These priorities are **not** a total ordering of intelligence. For example,
Gemma 12B can be the better daily service if single-GPU responsiveness and spare
capacity matter more than the last increment of task success.

## 3. Why these models make the cut

### Qwen3.8-27B: strongest first default, already available

The primary card identifies a native vision-language model with configurable
thinking and a hybrid layout: **64 main layers, 16 full-attention layers**, and
Gated DeltaNet layers in between. It supports 262,144 tokens natively, with a
separate extension claim to one million. Its MTP module is additional to the
main layer stack. [2]

In Qwen's own reporting table, Qwen3.8-27B scores **73.0 vs 63.4** for Qwen3.6-27B
on Terminal Bench 2.1 (Terminus), and **61.7 vs 53.5** on SWE-bench Pro. That is a
useful reason to evaluate it first—not a guarantee that the installed Q4 model,
a smaller context, or a different coding harness reproduces those results.
Neither figure is an independent local measurement. [2]

Your tag is `qwen3.8:27b-mtp-q4_K_M`; metadata records `draft_num_predict=4`.
This establishes a configured MTP setting, **not** that speculation is active,
correctly supported, or faster. Compare MTP on/off only after a working baseline;
accepted draft length and end-to-end throughput matter more than the tag name.

### North Mini Code and KAT: compare executable work, not model labels

**North Mini Code** is Apache-2.0, text-only, 30B total/3B active, and explicitly
trained for agentic coding. Cohere reports three-seed averages and discloses
its SWE-Agent/terminal harnesses, but some competitor scores come from other
reports. Its recommendation to preserve interleaved reasoning makes client
compatibility important: an OpenAI-compatible endpoint alone does not prove that
a coding client passes the expected history. [3]

**KAT-Coder-V2.5-Dev** is Apache-2.0 and based on Qwen3.6-35B-A3B, with only the
language-model weights released. Its in-house table reports **69.40% SWE-bench
Verified** and **45.96% SWE-bench Pro**. The card explicitly says each model was
tested once unless an obvious error required a rerun, and documents significant
harness-dependent deviations. Treat those numbers as a reason to test the model,
not as directly comparable with Qwen's or Cohere's headline scores. [4]

**Hardware implication:** 3B active parameters reduce per-token expert compute,
not the total weight footprint. Both packages exceed one card's capacity. Their
speed advantage over dense Qwen on this PCIe topology is a hypothesis, not an
observed result.

### Single-GPU choices: Gemma 12B, gpt-oss-20b, existing Qwen 9B

**Gemma 4 12B** has ample weight-memory headroom at Q4/Q5. The upstream model
supports image/audio input, but the inspected Ollama listing advertises
**text and image**, not audio. Do not promise end-to-end audio support through
Ollama merely because the underlying model supports it. [5, 11]

**gpt-oss-20b** offers a mature local path, native MXFP4 MoE weights, reasoning
effort control, and documented Ollama support. OpenAI says it can run within
16GB; context, allocator overhead, and concurrency still affect whether it fits
on the currently available card memory. Use Harmony-aware tools, not a generic
chat template. It is a good *deployment-fit* candidate, not a claim that a 2025
model wins all 2026 quality comparisons. [7]

**Qwen3.5-9B** is the sensible no-download fallback. A single GPU can hold the
installed 6.14GiB package with substantially more context/workspace margin than
the larger candidates. It can also leave the second GPU available for another
small model or training experiment, provided memory is explicitly budgeted. [8]

### Muse Glimmer: worth one controlled trial, not an assumed speed breakthrough

Meta releases full weights, two quantized variants, a perception encoder, and a
DFlash drafter under Apache-2.0. Its KQuant-17GB package is intended for 24GB-class
hardware; this machine needs a two-GPU split for the complete package. [6]

Meta reports **74.9 tok/s without speculation and 233.4 tok/s with DFlash on an
RTX 5090**, using greedy decoding and batch size one. **Do not transfer those
numbers to this machine.** GPU bandwidth, runtime kernels, draft acceptance,
multi-GPU communication, context, and sampling all differ. The inspected Ollama
registry provides separate `30b-q4_K_M` and `30b-q4_K_M-dflash` tags; the latter
adds memory and requires an actual runtime test. Avoid downloading an MLX tag
for this Linux/NVIDIA system. [6, 11]

### Creative variants: useful, but a different objective

Artemis and Orion are already present. Their creator's cards explicitly emphasize
creativity, writing, and entertainment rather than making correctness the primary
objective. The installed packages advertise text/tools/thinking but **not
vision**; the base family's capabilities are not necessarily included. [1, 9]

Neither local package includes license text in `/api/show`, and the inspected
upstream cards do not supply a clear license declaration. Gemma 4's base license
is Apache-2.0, but derivative packaging and notices still need verification before
redistribution. Orion's upstream metadata names a 31B base while its local tensor
metadata identifies a 25.2B MoE, another reason not to treat a display name as
complete provenance. No claim of exact lineage verification is made here.

## 4. Fit is weights + state + runtime, not parameter count alone

Budget for:

```text
peak device memory ≈ weights resident on that GPU
                   + KV cache / recurrent state
                   + activations and runtime workspaces
                   + vision encoder/projector
                   + optional speculative drafter
                   + allocator margin and other GPU users
```

A split must fit **each device**, not merely the combined total. GPU 0 and GPU 1
can have different workspace/embedding allocations even with equal VRAM.

### Worked example: Qwen3.8-27B context cost

For one sequence, the **main model's full-attention KV tensors only**, stored in
FP16/BF16:

```text
bytes = 2 (K and V) × 16 full-attention layers × 4 KV heads
      × 256 head dimension × 2 bytes × context tokens
```

| Context tokens | Main full-attention KV only |
|---------------:|----------------------------:|
| 8,192 | 0.5 GiB |
| 16,384 | 1 GiB |
| 32,768 | 2 GiB |
| 65,536 | 4 GiB |
| 131,072 | 8 GiB |
| 262,144 | 16 GiB |

This excludes DeltaNet state, MTP state/cache, image processing, batch buffers,
and runtime overhead. Cache quantization can reduce the KV portion when the
backend supports it; it does not shrink all those other components. The installed
16.52GiB package plus 16GiB of main KV already exceeds combined device memory at
256K, even before the omitted costs. Hence **16K/32K first, 64K after checking
residency**, not “the card says 256K, so set 256K.” [1, 2]

For Gemma, North, and gpt-oss, sliding-window/global patterns and, for Gemma,
shared/unified KV details change the calculation. Do not multiply every layer by
full context or assume the runtime implements every potential saving. Backend
allocation measurements are the final check.

## 5. Local configuration findings worth preserving

1. **North's context discrepancy is upstream too.** The local GGUF and upstream
   `config.json` say 500,000; Cohere's narrative card specifies 256K context and
   64K maximum output. Treat 500K as a configuration field, not verified usable
   context, and start far below either ceiling. [1, 3]
2. **KAT has a non-default local sampler.** Local parameters use temperature 0.2
   and 32,768 context. The upstream thinking example uses temperature 1.0,
   top-p 0.95, and additional controls; its non-thinking example differs again.
   Record/choose the mode explicitly when comparing rather than silently mixing
   presets. No settings were changed. [1, 4]
3. **Artemis/Orion already set 32K context.** An explicit request-level context
   overrides this for an initial smaller-memory trial. [1]
4. **Qwen's MTP and vision labels need smoke tests.** The local metadata exposes
   those capabilities, but no generation, image, or structured-tool test was run.
5. **Runtime variables belong to the server process.** Setting
   `CUDA_VISIBLE_DEVICES` or cache variables on an `ollama run` client does not
   reconfigure an already-running Ollama daemon. No daemon was restarted. [11]
6. **Preserve each model's chat/history contract.** North, Gemma, Qwen, and
   gpt-oss differ in reasoning retention and serialization. Do not globally
   strip all reasoning or concatenate raw special tokens across model families.
   Use the model-aware template/parser and test an entire tool-result round trip.
   [2, 3, 5, 7]

## 6. Runtime recommendation and concrete starting commands

**Use the existing Ollama first.** Its documentation says it places a model on
one GPU when it fits and spreads it when necessary. Recent GPU support includes
compute capability 12.0 / RTX 5060 Ti. This is a supported starting point, not
proof that every new GGUF/MTP architecture path has been exercised locally. [11]

For subsequent controlled tuning, llama.cpp offers explicit layer-split control.
A layer-split configuration is a reasonable first experiment on host-bridge GPUs;
do not assume tensor parallelism will be faster. vLLM/SGLang become more
interesting for batching and specialized kernels, but SM120, quantization,
architecture, and parser support must all match. In particular, **do not run the
upstream North/KAT BF16 multi-GPU examples unchanged on 2×16GB**. [3, 4]

### Try the existing default without changing persistent configuration

The following are **suggested commands, not commands executed for this report**.
They load a model and consume GPU resources; run one trial at a time.

```bash
curl --fail-with-body http://localhost:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8:27b-mtp-q4_K_M",
    "messages": [{"role": "user", "content": "Write a Python function that merges two sorted integer lists, then give edge-case tests."}],
    "stream": false,
    "options": {"num_ctx": 16384, "num_predict": 4096},
    "keep_alive": "5m"
  }'

ollama ps
nvidia-smi
```

Check `ollama ps` for **100% GPU** and the intended context, plus per-GPU memory
with `nvidia-smi`. CPU offload can still generate answers, but it is a different
latency/memory configuration. A loaded model or one correct answer does not yet
validate agentic tool use. If the response exhausts its output budget while
thinking, record it as truncated—not as a complete task result.

For comparable existing-model trials, substitute the exact local model name:

```text
north-mini-code-1.0:q4_K_M
kat-coder-v2.5-dev:q4_K_M
qwen3.5:9b
```

Keep output/context budgets explicit, but use and record appropriate per-model
sampling/reasoning settings rather than assuming identical temperatures represent
equal reasoning effort.

### Optional downloads, in priority order

The listed registry tags were checked. These commands **would download weights**;
none was run. Choose a missing capability rather than pulling all of them.

```bash
# First new download for a compact text/image assistant:
ollama pull gemma4:12b-it-q4_K_M

# Alternative single-GPU text reasoning / tools:
ollama pull gpt-oss:20b

# Additional two-GPU general/vision agent, after testing installed contenders:
ollama pull muse-glimmer:30b-q4_K_M
```

Google's 13.45GiB **Gemma 26B QAT text-only** single-GPU experiment requires the
specific official text GGUF and appropriate import/template setup. Do **not**
substitute the approximately 19 decimal GB `gemma4:26b` default package, nor assume
that a vision-inclusive QAT package leaves the same headroom.

For a first controlled run, keep one large model and one request active. Ollama's
Flash Attention and quantized KV settings are backend-dependent; test `q8_0`
cache quality/support before using more aggressive KV quantization. They are
server-wide settings, so changing them can affect all models. If pinning a
single-GPU server later, use GPU UUIDs and keep it loopback-bound. [11]

## 7. What not to prioritize

- **Another generic Qwen3.6/3.5 27–35B download:** you already have both a newer
  dense Qwen and specialist MoEs. Keep older models as baselines, not an urgent
  replacement path.
- **Qwen2.5-Coder-32B as the default:** the installed 18.49GiB package remains a
  useful older baseline. Its 64-layer conventional GQA cache is comparatively
  expensive: about **8GiB at 32K** in FP16 for one sequence, before runtime costs
  (derived from local metadata: 64 layers, 8 KV heads, head dimension 128). [1]
- **Abliterated/uncensored variants as automatic quality upgrades:** reduced
  refusals do not establish better coding, instruction retention, calibration,
  or tool reliability. Evaluate separately; keep external permission boundaries.
- **70B Q4 or gpt-oss-120b as comfortable all-GPU models:** 70B at even an ideal
  four bits is ~32.6GiB before quantization metadata and caches. The 120B class is
  larger still. CPU/GPU offload may make some such models load in combined RAM
  and VRAM, especially sparse ones, but that is an experimental, potentially
  slower operating mode—not the recommended responsive default. [7]
- **Hundreds-of-billions/trillion-parameter open MoEs:** small active parameter
  counts do not make their full weights fit this machine.
- **BF16 27–35B, or Q8 at the edge of total VRAM:** leave room for context and
  runtime. A higher-quality quant that repeatedly offloads may be a worse daily
  choice than Q4/Q5 with all weights resident.
- **Granite 4.2 8B as an urgent upgrade:** it is a legitimate Apache-2.0,
  reasoning/tool-use candidate for a compact enterprise baseline, but no evidence
  gathered here establishes it as better than your existing Qwen9B for your
  workload. Runtime/artifact support should be checked before adoption. [10]

### If “use” also includes fine-tuning

These recommendations concern **inference**. For QLoRA, begin with a supported
3–9B model and short sequences. A model fitting for inference does not establish
training fit: activations, gradients, optimizer state, and checkpointing change
the budget. Two 16GB GPUs using ordinary data parallelism replicate model state;
they do not automatically train a model requiring 32GB on each replica. Full
27–35B tuning is not a practical default here.

## 8. Evaluation that would settle the ranking

The next useful work is a small, controlled comparison of models already present,
not another catalogue. This is a **proposed evaluation**, not completed evidence.

- Use the same held-out tasks: repository bug fixes with tests, structured tool
  calls with a real result round trip, instruction retention, long-context fact
  retrieval, and screenshot/OCR tasks only for vision-capable packages.
- Freeze runtime version, artifact digest, template/parser, context/output caps,
  tool permissions, task revision, and model-specific sampling/reasoning mode.
- Start at 16K with one request, compare warm runs separately from model loading,
  then try 32K and 64K only where memory and task needs justify it.
- Record task success, malformed/hallucinated tool calls, truncations, prompt
  processing rate, generation rate, time to first token, full task latency,
  peak per-GPU VRAM, CPU offload, and human interventions. Reasoning tokens can
  inflate tok/s without improving time to a useful final answer.
- Repeat multiple seeds; compare cost/time to an accepted result. Test MTP or
  DFlash separately with its extra memory and acceptance statistics recorded.
- Sandbox generated code and require explicit approval for consequential actions.

**Decision rule:** keep Qwen3.8 as default unless North/KAT wins on *your*
accepted coding tasks or latency. Prefer a one-GPU Gemma12/Qwen9/gpt-oss service
when its task quality is sufficient and spare GPU capacity matters more.

## 9. Reproducibility notes and sources

### Selected installed artifact identities

Ollama digests identify local manifests/packages, not proof of upstream training
lineage. Size includes package components and is not peak VRAM.

| Local tag | Package bytes | Manifest digest prefix |
|-----------|--------------:|------------------------|
| `qwen3.8:27b-mtp-q4_K_M` | 17,741,872,154 | `22130167c4c2` |
| `north-mini-code-1.0:q4_K_M` | 18,593,967,008 | `d8b269ad5c7c` |
| `kat-coder-v2.5-dev:q4_K_M` | 21,391,448,864 | `7977282510fa` |
| `artemis-31b-v1.1:q4_K_M` | 19,598,490,321 | `34c1e411961e` |
| `orion-26b-a4b-v1:q5_K_M` | 19,319,199,057 | `99d74973255b` |
| `qwen3.5:9b` | 6,594,474,711 | `6488c96fa5fa` |

### Sources

Numbering is local to this hardware research note, **not** the SOTA Watch source
sequence. All live sources were retrieved **2026-09-08**. Primary model cards are
vendor evidence, not independent reproductions. Discovery searches were used to
find candidates, not to justify anonymous aggregate leaderboard rankings.

1. **Local inspection:** `lscpu`, `free -h`, `df -h`, `nvidia-smi` GPU/memory/PCIe
   queries and `topo -m`, PCI sysfs link attributes, `ollama --version`, `ollama
   list`, `ollama ps`, and the read-only localhost `/api/version`, `/api/tags`,
   `/api/show` metadata endpoints. No generation endpoint was called.
2. **Qwen3.8-27B:** [official card](https://huggingface.co/Qwen/Qwen3.8-27B),
   [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
   Apache-2.0; architecture, benchmarks, thinking, MTP, and vision support.
3. **North Mini Code:** [official card](https://huggingface.co/CohereLabs/North-Mini-Code-1.0),
   [configuration](https://huggingface.co/CohereLabs/North-Mini-Code-1.0/blob/main/config.json).
   Apache-2.0; coding specialization, context discrepancy, evaluation methodology,
   interleaved reasoning, and parser requirements.
4. **KAT-Coder-V2.5-Dev:** [official card](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev).
   Apache-2.0; 35B-A3B, text-only release, benchmarks/methodology, sampling, and
   serving requirements.
5. **Google Gemma 4:** [12B IT](https://huggingface.co/google/gemma-4-12B-it),
   [26B-A4B IT](https://huggingface.co/google/gemma-4-26B-A4B-it),
   [31B IT](https://huggingface.co/google/gemma-4-31B-it).
   Apache-2.0; architecture, modalities, thinking/history contracts. Configurations
   for 12B and 26B were also inspected.
6. **Muse Glimmer:** [Meta's official card](https://huggingface.co/meta-models/Muse-Glimmer-30B).
   Apache-2.0; quantized deployment envelopes, DFlash, benchmarks, and the exact
   hardware/decoding conditions of the reported speed measurements.
7. **gpt-oss-20b:** [official card](https://huggingface.co/openai/gpt-oss-20b),
   [configuration](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json).
   Apache-2.0; MXFP4, 16GB deployment claim, Harmony format, and Ollama commands.
8. **Qwen3.5-9B:** [official card](https://huggingface.co/Qwen/Qwen3.5-9B).
   Apache-2.0; vision, thinking, and runtime guidance.
9. **Creative derivatives:** [Artemis 31B v1.1](https://huggingface.co/TheDrummer/Artemis-31B-v1.1),
   [Orion 26B-A4B v1](https://huggingface.co/TheDrummer/Orion-26B-A4B-v1).
   Creator-stated objectives and usage; incomplete derivative license/provenance
   metadata is explicitly not treated as a verified permissive grant.
10. **Granite 4.2 8B:** [official card](https://huggingface.co/ibm-granite/granite-4.2-8b).
    Apache-2.0; optional compact enterprise/reasoning baseline.
11. **Ollama:** [FAQ](https://docs.ollama.com/faq),
    [GPU support](https://docs.ollama.com/gpu),
    [context](https://docs.ollama.com/context-length),
    [Gemma registry tags](https://ollama.com/library/gemma4/tags),
    [Muse Glimmer registry tags](https://ollama.com/library/muse-glimmer/tags).
    Runtime behavior and advertised packages/capabilities, not local load tests.
12. **Quantized artifact sizes:** Hugging Face model metadata (`?blobs=true`)
    supplied actual file byte sizes. Exact inspected revisions and repositories
    follow. Projectors/drafters are separate files; third-party quants are not
    identical to an official or locally imported package merely because both
    say Q4_K_M.

| Quant repository | Inspected revision |
|------------------|--------------------|
| [Qwen3.8-27B / Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | `4ca720788d1e01f1bff70c033e0d0028fd02e502` |
| [North Mini Code / Unsloth](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF/tree/e306bb4bf0df610f5471d97a01de2b6e0b24d356) | `e306bb4bf0df610f5471d97a01de2b6e0b24d356` |
| [KAT / bartowski](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/tree/d8f684f08d2950ea9d2db6a35ef7dada0707858b) | `d8f684f08d2950ea9d2db6a35ef7dada0707858b` |
| [Gemma 26B QAT / Google](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/tree/d1c082be9cf3c8a514acf63b8761f4b41935842e) | `d1c082be9cf3c8a514acf63b8761f4b41935842e` |
| [Gemma 12B / Unsloth](https://huggingface.co/unsloth/gemma-4-12B-it-GGUF/tree/fc034cfff751157913579611efad8462ac1be606) | `fc034cfff751157913579611efad8462ac1be606` |
| [Gemma 31B / Unsloth](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/c1ac76e99d5513b141e8adde7288b85c3f9c32ec) | `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` |
| [Muse Glimmer / Meta](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/tree/70bf1b61ac09f91b24d39038091b41c582bc5d7a) | `70bf1b61ac09f91b24d39038091b41c582bc5d7a` |
| [gpt-oss-20b / ggml-org](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/tree/ef9b12f2ff56c69cf32153a02784e7a3c88bf524) | `ef9b12f2ff56c69cf32153a02784e7a3c88bf524` |

Search was intermittently unavailable, so this is a researched, hardware-aware
shortlist rather than an exhaustive census. Exact local performance, full-GPU
residency, draft acceleration, multimodal functionality, and tool-call fidelity
remain unmeasured; the report does not fabricate tokens/second or task success.
