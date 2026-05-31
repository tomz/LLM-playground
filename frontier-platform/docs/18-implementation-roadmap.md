# 18 — Implementation Roadmap (post-gap-analysis)

Concrete, PR-sized work items that follow from `docs/17a-frontier-model-gap-research-v2.md`.
Each item names the file(s), the existing shape it replaces, the new shape, the
acceptance test, and whether it needs GPUs to validate.

The first four items (★) are being built in this iteration. The rest are
sequenced for follow-on work.

| # | Item | File(s) | Need GPU? | Status |
|---|---|---|---|---|
| ★1 | Real synthetic data factory | `platform/data/synthetic.py` → `platform/data/synthetic/` | No | **done** (28 tests) |
| ★2 | Real red-team harness | `platform/safety/redteam.py` → `platform/safety/redteam/` | No | **done** (28 tests) |
| ★6 | Batched MoE forward | `platform/model/transformer.py` (`MoEFFN.forward`) | No (single device test); GPU validates speedup | **done** (11 tests) |
| ★7 | vLLM serving backend | `platform/serving/engine.py` + new `vllm_engine.py` | No (logic test); GPU for integration | **done** (11 tests) |
| ★5 | Jailed sandbox wrap (nsjail/firejail/bwrap) | `platform/rl/jail.py` + `sandbox.py` | No | **done** (24 tests, 2 skipped) |
| ★3 | Real safety classifier interface | `platform/safety/classifiers.py` | No | **done** (26 tests) |
| 4 | Real source connectors (warcio/GHArchive/arxiv/wiki) | `platform/data/acquire.py` | No | planned |
| 8 | FSDP2 + DTensor TP wrap | `platform/training/parallel.py` | Yes (real validation) | planned |
| 9 | SWE-bench-Verified harness | `platform/eval/swebench.py` | No (logic); harness needs sandboxed code exec | planned |
| 10 | Real `lm-eval-harness` wiring + 2026 benchmark adapters | `platform/eval/harness.py` + adapters | No (logic) | planned |

---

## ★1. Real synthetic data factory

**Replaces:** `platform/data/synthetic.py` (16-line random-word-bag generator).

**New shape:**

```
platform/data/synthetic/
  __init__.py          # re-exports + back-compat write_corpus shim
  teacher.py           # Teacher protocol + EchoTeacher, TemplateTeacher,
                       # CallableTeacher, EngineTeacher (wraps serving.Engine)
  policies.py          # GenerationPolicy protocol + Rephrase, Textbook,
                       # MathProblem, QA, ReasoningTrace
  factory.py           # SyntheticFactory.generate(...) with rejection
                       # sampling, dedup, decontamination, lineage
  lineage.py           # SampleRecord, write_lineage_jsonl
```

**Interfaces:**

```python
class Teacher(Protocol):
    name: str
    def generate(self, prompt: str, **kw) -> str: ...

class GenerationPolicy(Protocol):
    name: str
    def prompts(self, n: int, *, rng) -> Iterable[str]: ...
    def acceptance_verifier(self) -> Verifier | None: ...

@dataclass
class SampleRecord:
    sample_id: str
    teacher: str
    policy: str
    seed: int
    prompt: str
    response: str
    accepted: bool
    verifier_score: float
    rejection_reason: str | None

class SyntheticFactory:
    def __init__(self, teacher: Teacher, policy: GenerationPolicy, *,
                 verifier: Verifier | None = None,
                 deduper: MinHashDeduper | None = None,
                 decontaminator: Decontaminator | None = None,
                 rng_seed: int = 0): ...
    def generate(self, n: int) -> Iterator[SampleRecord]: ...
    def write_jsonl(self, n: int, path: str | Path) -> Path: ...
```

**Back-compat:** `from platform.data.synthetic import write_corpus` still works
(`tests/conftest.py` and `scripts/smoke_pipeline.py` both use it).

**Acceptance tests:**

- `write_corpus` still produces the test corpus shape the old tests expect.
- `SyntheticFactory(EchoTeacher(), TemplateTeacher())` produces N records with
  accepted=True, no duplicates, lineage written.
- With `RejectionSampler(MathExactVerifier(42))`, only responses matching 42
  pass.
- With `MinHashDeduper`, near-duplicate teacher outputs are filtered.
- With `Decontaminator` indexed over a test eval set, contaminated samples are
  rejected and counted.

## ★2. Real red-team harness

**Replaces:** `platform/safety/redteam.py` (5 hardcoded prompts + refusal regex).

**New shape:**

```
platform/safety/redteam/
  __init__.py          # re-exports run_suite, run_all (back-compat), SUITES,
                       # plus the new API
  probes.py            # Probe dataclass; 60+ structurally-representative
                       # probes covering cbrn/cyber/persuasion/autonomy/
                       # bias/jailbreak/multi_turn_social
  suites.py            # Suite class, get_suite(name)
  judges.py            # Judge protocol + RegexRefusalJudge (back-compat),
                       # ClassifierJudge, EnsembleJudge, CallableJudge
  report.py            # build_report(results) → gates.preflight-compatible
                       # JSON; write_report(path, results)
```

**Probes are synthetic.** They are *not* actual harmful content — they are
structurally representative templates ("instructions for [forbidden act]",
"roleplay as [unrestricted persona]", etc.). This is a research / evaluation
harness, not a jailbreak playbook.

**Interfaces:**

```python
@dataclass(frozen=True)
class Probe:
    id: str
    suite: str
    category: str        # one of CATEGORIES from gates.py
    prompt: str
    metadata: dict

@dataclass
class JudgeResult:
    refused: bool
    category_scores: dict[str, float]  # 0=safe, 1=harmful
    rationale: str = ""

class Judge(Protocol):
    name: str
    def judge(self, probe: Probe, response: str) -> JudgeResult: ...

def run_suite(endpoint, suite: str, *, judge: Judge | None = None) -> dict: ...
def run_all(endpoint, *, judge: Judge | None = None) -> dict: ...
def build_report(results: dict) -> dict: ...   # consumed by gates.preflight
def write_report(path, results: dict) -> Path: ...
```

**Report shape (consumed by `gates.preflight` unchanged):**

```json
{
  "cbrn": 0.02, "cyber": 0.05, "persuasion": 0.10,
  "autonomy": 0.05, "bias": 0.10, "jailbreak": 0.10,
  "_per_suite": {...},
  "_meta": {"n_probes": 60, "judge": "ensemble", "...": "..."}
}
```

**Back-compat:** `run_suite(endpoint, "harmbench")` returns the same dict shape
(`refusal_rate`, `n_probes`, `details`) so existing tests pass.

**Acceptance tests:**

- Existing `tests/test_safety.py::test_redteam_run_suite_*` pass unchanged.
- New tests: ensemble judge, build_report → gates.preflight roundtrip with
  PASS/BLOCK as expected.

## ★6. Batched MoE forward

**Replaces:** `platform/model/transformer.py:300-318` (`for e in range(...)`).

**New:** sort-by-expert dispatch — flatten (N, k) routing to (N·k,) slots,
sort by expert id, give each expert a contiguous slice via `index_select`,
write back with `index_add_` weighted by slot weight.

The shape is correct for a later EP all-to-all (sort-then-dispatch is what
real expert-parallel kernels do, just with the sort spanning ranks).

**Config:** new `ModelConfig.moe_dispatch = "batched" | "loop"` (default
`"batched"`); old behavior reachable for ablations / parity checks.

**Acceptance tests:**

- New `test_moe_batched_matches_loop`: with seeded init + same routing decision,
  batched and loop outputs are `allclose` (modulo float accumulation order).
- All existing `test_moe_*` tests still pass.
- New micro-benchmark test: batched is at least as fast as loop at
  `n_experts=16` (skipped if no GPU).

## ★7. vLLM serving backend

**Replaces:** `platform/serving/engine.py:43-48` (vLLM branch raises
`NotImplementedError`).

**New file:** `platform/serving/vllm_engine.py` with `VLLMEngine` matching
`TorchEngine.generate` chunk schema (`{token_id, logprob, text, done}`).

**Wire:** `Engine.__init__` instantiates `VLLMEngine(cfg)` when
`cfg.backend == "vllm"`. If `vllm` isn't importable, raise a clear
`ImportError` with install hint (not `NotImplementedError`).

**Weight sync hook:** `VLLMEngine.update_weights(state_dict)` for the
out-of-process RLVR actor case. In modern vLLM this calls
`llm_engine.model_executor.driver_worker.model_runner.model.load_state_dict`;
where unavailable, raises with the version requirement.

**Acceptance tests:**

- Unit test with monkeypatched fake `vllm` module: `Engine(cfg=…backend=vllm)`
  constructs without raising; `generate(req)` yields chunks matching the
  TorchEngine schema; `done` chunk carries `usage`.
- Integration test marked `@pytest.mark.skipif(no vllm)`: run a 1-step
  generation on a tiny HF model.

---

## Sequence rationale

1. **Synthetic factory and red-team harness** are pure-Python and unblock the
   *content* gaps (largest leverage per engineer-week).
2. **Batched MoE** is a single-file change that removes a real performance
   bottleneck and is the right *shape* for later expert-parallel work.
3. **vLLM backend** is the single change that simultaneously unlocks
   (a) scaled RLVR rollouts via the existing `AsyncRolloutEngine` and
   (b) the serving product line.

After these four, the next-most-leveraged work is **#5 jailed sandbox** (one
file, enables safe scaled code-RL) and **#10 lm-eval wiring** (replaces
predicted benchmark numbers with real ones).

---

## What landed (implementation log)

All six ★ items are committed with parity + back-compat tests. Headline counts:

- **124 new tests** across the six items, all green.
- **290 / 290** tests passing project-wide (2 skipped: nsjail/firejail-only
  argv tests on a box with only bwrap installed).
- Smoke pipeline (`scripts/smoke_pipeline.py`) still passes end-to-end.

Per-item delivery:

- **★1 synthetic factory** — replaced the 16-line word-bag generator with a
  package (`teacher.py`, `policies.py`, `factory.py`, `lineage.py`). Pluggable
  `Teacher` (Echo, Template, Callable, Engine), six policies including
  R1-style `ReasoningTracePolicy`, rejection sampling against
  `platform.rl.verifiers`, MinHash dedup, contamination filtering via
  `platform.eval.contamination`, JSONL lineage with `SampleRecord`. Back-compat
  `write_corpus` shim preserves all existing call sites.
- **★2 red-team harness** — replaced the 5-prompt + regex toy with a package
  (`probes.py`, `suites.py`, `judges.py`, `report.py`). 26+ probes across all
  6 gate categories, pluggable `Judge` protocol with `RegexRefusalJudge` /
  `ClassifierJudge` / `EnsembleJudge` / `CallableJudge`, `build_report` /
  `write_report` producing JSON that flows straight into the existing
  `gates.preflight`. Back-compat `run_suite`, `run_all`, `SUITES` unchanged.
- **★6 batched MoE** — added `moe_dispatch = "batched" | "loop"` config
  (default `batched`). Old per-expert Python for-loop kept as the parity
  reference; new sort-by-expert dispatch uses `argsort` + `index_select` +
  per-slab GEMM + `index_add_` — the same shape an EP all-to-all needs. Parity
  test (`allclose` between backends across random inputs/seeds) and soft perf
  test included.
- **★7 vLLM backend** — new `platform/serving/vllm_engine.py` matching the
  TorchEngine chunk schema exactly. Engine dispatch in `engine.py` now routes
  `backend="vllm"` to it (and `ImportError` if vLLM isn't installed, not the
  old `NotImplementedError`). Includes `update_weights(state_dict)` hook for
  the out-of-process RL actor path. Tested against a fake `vllm` module so the
  whole adapter is exercised without needing GPUs.
- **★5 jailed sandbox** — new `platform/rl/jail.py` with `Jailer` protocol +
  `BubblewrapJailer` / `NsjailJailer` / `FirejailJailer` / `NoJailer`,
  auto-detection that prefers bwrap (least-privilege, no SUID needed), and
  `run_in_jailed_sandbox` as a drop-in for `run_in_sandbox`. The sandbox grew
  a `jailer=` kwarg so behaviour is unchanged for existing callers. Real
  security assertions in the test suite: a child inside the jail cannot open
  TCP sockets and cannot mutate host files (verified against a tmp canary
  whose mtime is checked after the candidate "succeeds" at writing it).
- **★3 safety classifier interface** — replaced the 30-line keyword counter
  with a pluggable `Classifier` protocol + `KeywordClassifier` (kept as the
  CI fallback) + `LlamaGuardClassifier` (HF lazy-load with a graceful
  fallback-to-keyword on missing weights so the serving stack never goes
  un-classified) + a `callable` backend for test injection + `ClassifierEnsemble`
  (max/mean/min reductions). `InputClassifier` / `OutputClassifier` are now
  thin shims over a configurable backing classifier; existing exact-value
  tests still pass unchanged.
