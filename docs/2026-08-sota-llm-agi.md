# SOTA Watch — LLM & AGI · 2026-08

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-08-02  ·  **Status:** draft — month to date

> **Living edition as of August 2, 2026.** The first frontier signal of the month
> is unusually consequential: OpenAI reports that an internal successor model,
> Astra, generated ten new results on open problems across mathematics and
> theoretical computer science, after which humans prepared manuscripts with the
> model and the model formalized the arguments in Lean. The claimed results span
> geometry, coding theory, complexity, group theory, operator algebras, quantum
> games, lattice cryptography, and extremal combinatorics. This is not yet a
> complete August survey, and the mathematical community still needs to scrutinize
> the manuscripts and their significance. **The sole source currently included
> is dated August 1, 2026; this edition will grow as the month progresses.**

---

## The frontier this month

- **Astra moved automated research from conjecture search toward a proof
  pipeline.** OpenAI reports ten results on problems with no progress on the main
  result for at least a decade. The workflow separates discovery, exposition,
  and verification: the model generated the arguments; humans prepared the
  manuscripts with model assistance; the model then formalized each argument as
  a Lean certificate. [57]
- **The reported breadth matters as much as the count.** The claims include new
  bounds in sphere packing and coding theory, existence of non-sofic groups, a
  disproof of Connes's rigidity conjecture, arithmetic-circuit lower bounds, a
  quantum parallel-repetition theorem, closest-vector hardness, resolution of
  Ehrhart's volume conjecture, and extremal/Ramsey results. If independently
  validated, this is evidence of cross-domain research capability rather than a
  benchmark-specific math solver. [57]
- **Formal verification became part of the release artifact.** Lean certificates
  do not establish novelty or importance, but they can make logical correctness
  more auditable than prose-only model output. The practical frontier is therefore
  a coupled system: search → proof sketch → human mathematical review → formal
  certificate → community validation. [57]
- **Research cost was reported in API-equivalent terms.** OpenAI estimates that
  the tokens used to find all ten solutions would cost roughly **$2,000 at GPT‑5.6
  Sol API rates**. This excludes model-development, researcher, formalization,
  and verification costs, but it makes marginal search compute concrete enough
  to compare with human and automated-research workflows. [57]
- **Attribution and responsibility became explicit design questions.** OpenAI
  states that human authorship would misrepresent arguments generated entirely
  by the system, while taking responsibility for manuscript preparation and
  formal correctness. The field now needs durable provenance for prompts,
  attempts, model versions, human edits, certificates, and prior-art checks—not
  only a final PDF. [57]

---

## TL;DR — what we harvested

- **Full attempt provenance shipped.** Nanogpt's JSONL sidecar records every
  keep/discard/crash with candidate hash, git revision, seed, budget, tokens,
  wall/verifier time, gate verdict, and human interventions. Frontier's research
  artifact schema adds model revision, parent attempts, human edits, certificate,
  reviewers, and complete generation/verifier/reviewer/preparation cost.
- **Verifier-gated search shipped as a measurable protocol.** Midgpt counts the
  valid-candidate rate, all failed-branch generation tokens, and verifier time;
  frontier release gates reject artifacts missing certificate validity,
  theorem/claim identity, prior-art review, significance review, or independent
  reviewers.
- **Structured proof output shipped without pretending to ship Lean.**
  `coder-finetune/eval/structured_proof.py` validates theorem/certificate/
  citations, blocks wrong-target certificates before checker execution, and
  injects the external checker boundary. A real Lean binary/jail remains planned.
- **The full architecture is documented.** Frontier's
  `docs/15-july-august-2026-harvest.md` pins provenance, held-out review, cost,
  and backend invariants; existing production `NotImplementedError` stubs remain
  untouched.

---

## Hardware envelopes per project

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| **nanogpt-edu** | 10M–100M | laptop CPU / 8 GB GPU | 1× H100 80 GB | fast experiment-search baselines and full attempt ledgers |
| **midgpt** | 124M–1.5B | 1× 16 GB GPU | 8× H100 single node | research-agent ablations with useful models and controlled token budgets |
| **distgpt** | 1B–70B | 1 node × 8× A100 | 8–64 nodes × 8× H100/B200 + fast fabric | distributed candidate generation and verifier workloads |
| **coder-finetune** | 0.5B–7B | 1× 8–16 GB GPU | 1–8× H100 | theorem/proof-domain adaptation and structured-output evaluation |
| **frontier-platform** | 1B–500B+ | design docs and CPU protocol tests | research fleet plus proof-assistant and literature infrastructure | parallel search, formal checking, novelty review, provenance, release governance |

---

## Tier 1 — high-ROI, broadly applicable

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| **Machine-checkable research artifacts** | Couples a claimed proof/result to a formal certificate checked by a small trusted kernel | Makes logical verification reproducible and separates proof validity from model confidence | Formalization can encode the wrong theorem; novelty and significance remain external | CPU proof assistant | [57] | 🟡 schema + checker protocol shipped; Lean adapter planned | coder-finetune; frontier-platform |
| **End-to-end attempt provenance** | Records model/version, prompts, branches, failures, human edits, citations, and certificates | Enables attribution, reproducibility, contamination review, and honest cost accounting | Large logs may expose private reasoning/data; provenance schemas can omit decisive human work | CPU + durable store | [57] | ✅ shipped | nanogpt-edu; frontier-platform |
| **Verifier-gated research search** | Rejects candidates that fail formal or executable checks before expensive review | Converts an open-ended search into a measurable funnel | Optimizes only what the verifier captures; can reward trivial or misformalized claims | CPU verifier + model/API | [57] | ✅ protocol + cost benchmark | midgpt; coder-finetune; frontier-platform |
| **Human/community release gate** | Requires domain review for novelty, context, significance, attribution, and responsible publication | Prevents machine-checked but irrelevant or already-known outputs from being called discoveries | Expert review is scarce and cannot be reduced to one scalar metric | No special hardware | [57] | ✅ protocol; real reviewers external | frontier-platform |

---

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **Parallel theorem/research search** | Explores many lemmas, constructions, counterexamples, and proof plans concurrently | Raises the chance of rare useful discoveries and exposes multiple proof routes | Frontier model/API budget, deduplication, branch scheduling, and verifier throughput | [57] | ideal | frontier-platform |
| **Model-assisted formalization** | Translates informal arguments into Lean and iterates on checker errors | Produces auditable certificates and can expose gaps in prose proofs | Needs domain libraries, proof-assistant expertise, and exact theorem alignment | [57] | 🟡 structured output/checker contract; Lean loop ideal | coder-finetune; frontier-platform |
| **Literature/novelty verification** | Searches prior results and maps a candidate claim to the existing field | Distinguishes a new theorem from rediscovery or a known corollary | Reliable scholarly corpora, citation graphs, expert adjudication, and long context | [57] | ideal | frontier-platform |

---

## Tier 3 — research bets (track, don't build yet)

- **Recursive automated mathematics.** Ten claimed results do not establish an
  open-ended self-improvement loop. Track whether generated results improve the
  next system's search process rather than only adding isolated solutions. [57]
- **AI-generated formal libraries.** Large reusable theorem libraries could
  compound research speed, but poisoned abstractions or mis-stated definitions
  can contaminate many downstream certificates despite each term type-checking.
- **Marginal-token economics as a complete research metric.** The reported
  ~$2,000 is useful but incomplete. Comparisons need model-training amortization,
  failed searches, verifier compute, human review, manuscript preparation, and
  opportunity cost. [57]
- **Autonomous publication.** Formal correctness does not settle novelty,
  significance, exposition, attribution, or social responsibility. Keep humans
  and the relevant scholarly community in the release path. [57]

---

## Roadmap by project

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Add explicit external-verifier timing to a real search run | minimal | Per-attempt identity, budgets, failed branches, wall/verifier fields, and human interventions now persist in JSONL |
| **midgpt** | Drive candidate search with a model and deterministic checker | minimal→ideal | Benchmark and complete cost accounting shipped; current tests use deterministic generators/verifiers |
| **distgpt** | Attach generation GPUs and CPU proof verifiers to the lease queue | ideal | Resumable queue state machine shipped; production backend and worker pools remain |
| **coder-finetune** | Integrate a jailed Lean checker and held-out theorem set | minimal | Structured schema, exact theorem matching, checker result, and novelty-review boundary shipped |
| **frontier-platform** | Add literature backend and real independent reviewer workflow | minimal→ideal | Artifact JSONL and release gate now define claim, attempts, edits, certificate, reviewers, and full cost |

---

## What shipped this month

- **`nanogpt-edu/research/provenance.py`:** schema-versioned full-attempt JSONL,
  candidate hashes, budgets, failed branches, human interventions, total cost.
- **`midgpt/research_search.py`:** deterministic candidate-search benchmark with
  valid rate, all-branch generation tokens, verifier time, and first-valid result.
- **`distgpt/eval/rollout_queue.py`:** resumable generation/verifier work state
  machine with leases, retries, lineage, dead-lettering, and snapshots.
- **`coder-finetune/eval/structured_proof.py`:** structured theorem/certificate/
  citation parsing, exact target matching, injected checker, separate novelty gate.
- **`frontier-platform/infra/research_artifacts.py`:** research artifact JSONL,
  complete cost model, and release gate requiring certificate, claim identity,
  prior-art/significance review, and independent reviewers.
- No actual theorem discovery, Lean installation, or novelty corpus is claimed;
  those remain external adapters and expert workflows.

---

## Sources

Sources continue the cumulative numbering from July (52–56).

57. OpenAI, *Ten advances in mathematics and theoretical computer science*,
    **1 August 2026** — reports ten Astra-generated results on longstanding open
    problems, human/model manuscript preparation, Lean formalization of each
    argument, and roughly $2,000 in Sol-rate token cost for solution search.
    <https://openai.com/index/ten-advances-in-mathematics/>

> **Methodology note.** This is a first-party announcement one day old. The
> existence of Lean certificates strengthens logical auditability but does not by
> itself establish novelty, significance, correct theorem formalization, or
> community acceptance. This edition reports the claims as OpenAI's and will be
> updated as manuscripts, certificates, and independent expert assessments are
> examined. August is still open; absence of other entries means “not yet
> verified as of August 2,” not “nothing else happened.”
