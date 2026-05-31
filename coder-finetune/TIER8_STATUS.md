# coder-finetune Tier 8 — status & next steps

_Generated: end of the Tier 8.4 session. Tier 8 complete._

## Where we are

**Goal:** bring `coder-finetune` up to the same "real bugs caught + opt-in
frontier toolbox + pedagogical pins + structured docs" bar that
`nanogpt-edu` / `midgpt` / `distgpt` reached in earlier sessions.

**Plan:** four hermetic commits (Tier 8.1 → 8.4). **All four done.**

| Tier | Commit | Δ tests | Subject |
|------|---------|--------:|---------|
| 8.1 | `14400df` | +13 | Bug fixes |
| 8.2 | `766e3a1` | +13 | Subprocess sandbox hardening |
| 8.3 | `ab03c83` | +16 | Frontier opt-ins + pedagogical pins |
| 8.4 | `be2adb5` | +0  | README + docs refresh |

Total: **24 → 66 tests** (+42). Full suite green across repeated runs,
zero flakes. Working tree clean.

---

## Tier 8.1 — bug fixes (`14400df`)

Six real bugs, each caught by a fresh test:

1. **`extract_code` prose contamination** (`eval/run_humaneval.py`). Base
   models often emit prose before the code (`"Sure, here you go:\ndef ..."`).
   The old `extract_code` returned the whole string when no fence was
   present, then `exec()` died with `SyntaxError` and the verifier scored
   the answer 0 — counting a correct answer as wrong. New version
   recovers the code suffix from the first `def`/`async def`/`class`/
   `import`/`from` line.

2. **`async def` invisible to format reward** (`cf_rl/reward.py`).
   `_DEF_RE = r"^\s*def\s+\w+\s*\("` didn't match `async def`, so async
   helpers (HTTP clients, fetchers, asyncio examples) lost the format
   bonus even with a clean fenced block — biasing GRPO against valid
   async outputs. Now matches both.

3. **`eval/run_humaneval.py --n-samples` was a no-op.** Argparse
   accepted it; `quick_eval` never used it. Pass@k silently ran as
   pass@1. Now uses `num_return_sequences` in one `.generate()` call and
   short-circuits on the first pass (a problem counts as solved if any of
   the k samples passes). Also forces `temperature > 0` when
   `n_samples > 1` (otherwise samples are identical greedy decode).

4. **Eval missing `eos_token_id`.** Generation ran to `max_new_tokens`
   past `<|im_end|>` — wasted wall-clock and polluted completions with
   whatever the model rambled next. `infer/generate.py` had this right
   via `_eos_ids(tok)`; copied the helper into the eval CLI too.

5. **`infer/generate.py` used `dtype=` instead of `torch_dtype=`.**
   `requirements.txt` pins `transformers>=4.45,<5.0`. The `dtype` kwarg
   only became valid in 4.46+; on a 4.45 install this is a `TypeError`.
   Switched to `torch_dtype=` for full pin-range compatibility (matches
   `merge_lora.py` + `train.py`).

6. **GRPO divisibility footgun in shipped config.**
   `configs/grpo_3050.yaml` had `num_generations=6` against an effective
   batch of 8 (`batch_size=2 × grad_accum=4`). TRL ≥0.13 requires the
   effective batch be divisible by `num_generations`, so the user would
   hit an opaque tensor-shape error several minutes into a real run.
   Added a `build_grpo_trainer` validator that fails fast with an
   actionable message naming the offending fields. Fixed the shipped
   config to `G=8`.

13 new tests in `tests/test_bugfixes.py`.

**Also seen but not separated into its own tier:** the same change-set
also brought a stray nanogpt-edu Tier 7.1 series along, which I split
out into commit `ae37ca5` ("nanogpt-edu Tier 7.1: bug fixes ..." — 15 →
30 tests). That was from earlier-session context that got conflated
into the first Tier 8.1 commit; I soft-reset and split it for clean
provenance.

---

## Tier 8.2 — subprocess sandbox hardening (`766e3a1`)

The eval and RL verifier `exec()` model-generated code locally. The
README always warned "not a security boundary, run untrusted models in
Docker" — but the floor was thin: bare `fork()` + `exec()` with no
resource limits, child `print()` flooded trainer logs, `mp.Queue` race
could return wrong-or-no result.

**Added in `eval/run_humaneval.py`:**

- **`resource.setrlimit` floor** (POSIX, best-effort):
  - `RLIMIT_FSIZE = 0` — can't write to disk
  - `RLIMIT_NOFILE = 64` — can't open many fds
  - `RLIMIT_CPU = 2 × timeout` — CPU-budget backstop under wall-clock
  - `RLIMIT_AS = 1 GiB` — **spawn-mode only** (see gotcha below)
  - Per-call overridable via `run_one(limits={...})`; opt out entirely
    with `limits=False`.

- **stdout/stderr silencing at two layers:**
  - fd-level: `os.dup2(/dev/null, 1)` and `2`
  - Python-level: `sys.stdout = open(os.devnull, 'w')` (and stderr)
  - Both are needed because pytest's `capfd` replaces `sys.stdout` with
    a capture-file-backed object — the child inherits it under `fork`,
    so a `print()` bypasses fd 1 entirely.
  - Order matters: **silence first, rlimits second**. Otherwise the
    inherited capture-file stdout under `RLIMIT_FSIZE=0` trips `EFBIG`.

- **`mp_mode='spawn'` opt-in** for untrusted models. Fresh interpreter,
  no inherited heap / fds / imported modules. Default stays `fork`
  (~10 ms vs ~100 ms) since the rlimit floor applies in both modes and
  flipping the default would 10× the existing test suite runtime.

- **Replaced `q.empty()` + `q.get()` with `q.get(timeout=5.0)`.**
  `mp.Queue.empty()` is documented as unreliable; it gave us a ~5%
  flake rate where a child's pickled result hadn't drained to the
  parent's pipe yet at the moment we checked. Now the parent blocks up
  to 5 s for the queue to deliver (child already exited per `p.join`).

- **Kill-signal names in error messages** — `"killed-SIGKILL"`,
  `"killed-SIGXFSZ"` etc. so a debugging user can tell rlimit-kill
  from wall-clock timeout from a normal failure.

13 new tests in `tests/test_sandbox.py`.

### Failed experiment, documented honestly

I tried `run_many` as a `ThreadPoolExecutor(max_workers=N)` over
per-program `run_one` calls. Measured a real **~4× speedup** on a
batch of 8 sleeps. But under Python 3.14:

- Concurrent `mp.Queue()` creation from threads races. Pickled
  results were delivered to the wrong consumer's queue.
- Measured **22% flake rate** in the wild (8 reference solutions
  occasionally scored 0 instead of 1).
- Tried widening a `threading.Lock` around the entire
  `Queue() + Process() + start()` sequence — flake rate dropped to
  ~12%, didn't vanish.
- Tried `mp_mode='spawn'` (no fork-from-threads) — still ~45% flake.
  The race is in `mp.Queue` setup itself, not the fork.

**Decision:** reverted `run_many` to sequential. Kept the API shape
(one batch call from `code_unit_test_reward`, not N per-row calls) so a
future `ProcessPoolExecutor`-with-persistent-workers implementation is
a drop-in replacement with no caller churn. Added an explicit
**order-preservation pin** so whatever the future parallel version
looks like, it has to keep results in input order. Postmortem in the
commit message.

### Gotchas worth remembering

- **`RLIMIT_AS=1 GiB` under `fork()` is a `SIGKILL` trap.** A
  pytest+HF parent has VSZ ~5 GiB; the fork child inherits the
  address-space mapping (COW), so a `setrlimit(AS, 1G)` SIGKILLs the
  child before `exec()` even runs. Only apply `RLIMIT_AS` under
  `spawn` (fresh interpreter).
- **`RLIMIT_FSIZE=0` doesn't block `open('w')`** — open just truncates
  to zero bytes which is `<= 0` so doesn't trip. The `write()` call on
  a `TextIOWrapper` *buffers*, so the `EFBIG` only fires at
  `flush()`/`close()`, often during gc *after* `exec()` returns. Tests
  that pin file-write blocking must call `flush()` explicitly.

---

## Tier 8.3 — frontier opt-ins + pedagogical pins (`ab03c83`)

Mirrors the structure of nanogpt-edu Tier 7.3. Three categories, +16 tests
in `tests/test_pedagogy.py`.

### A. Opt-in speed knob (config-gated, default off)

| Knob | Where | Effect | Cost |
|------|-------|--------|------|
| **vLLM rollouts for GRPO** | `grpo.use_vllm: true` | TRL's `GRPOConfig.use_vllm` delegates generation to vLLM — ~3–8× faster rollouts on the same GPU. Huge for the gen-heavy GRPO step. | extra vLLM dependency; warm-up cost |

Implemented as a **pure `grpo_extra_kwargs(cfg)` helper** in
`cf_rl/grpo_train.py` so it's unit-testable without importing TRL or
loading a model. Only threaded into `GRPOConfig` when truthy (old TRL
builds aren't handed an unknown kwarg); optional
`vllm_gpu_memory_utilization` passes through only when explicitly set.
Documented + shipped `use_vllm: false` in `configs/grpo_3050.yaml`.

### B. Eval CLI surfacing (`eval/run_humaneval.py`)

- `--save-completions <path>`: per-problem prompt + every sample +
  extracted code + pass/fail as JSONL (greppable, streams on long runs).
- `--json-out <path>`: machine-readable `{pass@k, n, passes, ...}` summary;
  metric key named by `n_samples` so a diff tool tells pass@1 from pass@5.
- Refactored into pure `build_eval_summary` / `write_json_summary` /
  `write_completions_jsonl` helpers (no model load) so they're testable.

### C. Pedagogical pins

1. **GRPO group standardization** — new reference impl
   `group_standardize_advantages` in `cf_rl/reward.py`. Pins binary rewards
   `[1,0,0,1] → [+1,-1,-1,+1]`, baseline-invariance (uniform offset can't
   bias the gradient), per-group independence, zero-signal constant groups
   (no NaN), ragged-batch rejection, and the no-std (Dr. GRPO) variant.
2. **DPO margin monotonicity** — one SGD step on a clean
   `(prompt, chosen, rejected)` triple must raise the implicit-reward margin
   `(logπ_c − logπref_c) − (logπ_r − logπref_r)`. **Resolved the
   model-loading risk** flagged in the original plan by hand-building a
   2-layer `LlamaForCausalLM` from scratch (~50k params, no HF download) —
   the suite stays hermetic and CI-runnable without a token. Plus a
   loss-shape companion pin.
3. **`extract_code` round-trip** — every builtin (chosen/rejected) body
   survives a fence-wrap → `extract_code` cycle, pinning the verifier's
   code recovery against future fence-format drift.

---

## Tier 8.4 — docs refresh (`be2adb5`)

README brought up to date with the Tier 8 work (docs only, no new tests):

1. New **"Sandbox & safety"** section between the training-tracks ladder
   and the speed/quality knobs — rlimit floor table (what's bounded, what
   isn't), fork-vs-spawn tradeoff + why `RLIMIT_AS` is spawn-only,
   kill-cause message tags, and the "Docker is the real boundary" caveat.
2. `grpo.use_vllm` added to the Speed & quality knobs table.
3. New eval CLI flags documented in the quickstart with a reproducible
   pass@5 example.
4. Layout test count updated (24 → 66).
5. "Recent changes (Tier 8)" summary near the bottom, matching the pattern
   in the nanogpt-edu / midgpt READMEs.

---

## Files touched (Tier 8 total)

```
coder-finetune/
├── cf_rl/
│   ├── grpo_train.py        # 8.1: divisibility validator
│   │                        # 8.3: grpo_extra_kwargs (vLLM opt-in)
│   └── reward.py            # 8.1: async-def regex
│                            # 8.2: route through run_many batch
│                            # 8.3: group_standardize_advantages
├── configs/
│   └── grpo_3050.yaml       # 8.1: num_generations 6 → 8
│                            # 8.3: use_vllm knob (default off)
├── eval/
│   └── run_humaneval.py     # 8.1: extract_code + pass@k + eos + --seed
│                            # 8.2: rlimits, silencing, spawn opt-in, q.get fix
│                            # 8.3: --save-completions, --json-out + helpers
├── infer/
│   └── generate.py          # 8.1: dtype → torch_dtype
├── README.md                # 8.4: sandbox section, vLLM knob, CLI, Tier 8 summary
└── tests/
    ├── test_bugfixes.py     # 8.1: +13 tests
    ├── test_sandbox.py      # 8.2: +13 tests
    └── test_pedagogy.py     # 8.3: +16 tests
```

---

## How to verify

```bash
cd coder-finetune   # from the LLM-playground repo root
.venv/bin/python -m pytest -q              # 66 passed
git log --oneline -4                       # 8.1 → 8.4 commits
```

Tier 8 is complete. The post-training ladder (SFT → DPO → GRPO) is now
bug-fixed, sandbox-hardened, opt-in-equipped, pinned, and documented to
the same bar as the sibling projects.

## Open questions / decisions deferred (future Tier 9)

1. **Parallel reward executor.** The `ProcessPoolExecutor` approach (one
   fork per worker up-front, persistent queue) avoids the fork-from-threads
   race that killed the threaded attempt. `run_many` keeps the batch API
   shape so it'd be a drop-in. Flag for Tier 9 if GRPO step time becomes a
   bottleneck on a real run.
2. **`max_workers` kwarg in `run_many`.** Currently silently dropped
   (`del max_workers`). Keep as a forward-compat hint or remove — left for
   the future parallel reimplementation.
3. **Spawn-mode as default for safety.** ~10× slower per call than fork;
   would need a warm worker-pool to amortize. Same family as #1.
4. **Real vLLM rollout run.** The `use_vllm` wiring is unit-pinned but not
   yet exercised end-to-end on a GPU (needs the `vllm` dep installed). A
   future session could ship a measured before/after rollout-throughput
   number, like the nanogpt-edu A/B charts.
