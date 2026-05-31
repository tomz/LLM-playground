# coder-finetune Tier 8 — status & next steps

_Generated: end of the Tier 8.2 session. Hand-off doc for resuming work._

## Where we are

**Goal:** bring `coder-finetune` up to the same "real bugs caught + opt-in
frontier toolbox + pedagogical pins + structured docs" bar that
`nanogpt-edu` / `midgpt` / `distgpt` reached in earlier sessions.

**Plan:** four hermetic commits (Tier 8.1 → 8.4). Two of four done.

| Tier | Commit | Δ tests | Subject |
|------|---------|--------:|---------|
| 8.1 | `14400df` | +13 | Bug fixes |
| 8.2 | `766e3a1` | +13 | Subprocess sandbox hardening |
| 8.3 | — | — | Frontier opt-ins + pedagogical pins |
| 8.4 | — | — | README + docs refresh |

Total so far: **24 → 50 tests** (+26). 14 consecutive full-suite runs
green, zero flakes. Working tree clean. Branch `main` ahead of
`origin/main` by 64 commits.

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

## Tier 8.3 — frontier opt-ins + pedagogical pins (planned)

Mirrors the structure of nanogpt-edu Tier 7.3. Two categories:

### A. Opt-in speed/quality knobs (config-gated, default off)

| Knob | Where | Effect | Cost |
|------|-------|--------|------|
| **vLLM rollouts for GRPO** | `grpo.use_vllm: true` | TRL's `GRPOConfig.use_vllm` delegates generation to vLLM — ~3–8× faster rollouts on the same GPU. Huge for the gen-heavy GRPO step. | extra vLLM dependency; warm-up cost |

**Status:** verified during Tier 8.2 recon that TRL 1.3.0 (the version
in this env) has `GRPOConfig(use_vllm=False)` as a real parameter, so
the wiring is just "thread `cfg['grpo']['use_vllm']` into `GRPOConfig`
+ doc the dep".

### B. Eval CLI upgrades

- `--save-completions <path>`: dump each problem's prompt + completion
  + pass/fail + extracted code as JSONL. Lets the user actually see
  what the model wrote.
- `--json-out <path>`: machine-readable summary (`{"pass@k": ..., "n":
  ...}`) so the eval is comparable between runs.
- Eval is already deterministic given `--seed` (Tier 8.1); this just
  surfaces the results properly.

### C. Pedagogical pins (the "this is what DPO/GRPO actually do" tests)

Mirroring nanogpt-edu 7.3's RoPE/causal-mask/SwiGLU pins:

1. **DPO margin monotonicity**: after one optimizer step on a clean
   `(prompt, chosen, rejected)` pair, the policy's
   `logprob(chosen) - logprob(rejected)` margin must *increase*. Pins
   the entire DPO machinery in ~30 lines (loads a tiny model, builds a
   DPOTrainer, takes one step, compares margins).
2. **GRPO group-standardization invariant**: rewards `[1, 0, 0, 1]`
   standardized within a group of 4 must give advantages
   `[+1, -1, -1, +1]` to within float tolerance, regardless of the
   global mean. (Test the standardization, not the trainer — easier
   and faster.)
3. **`code_unit_test_reward` determinism across calls**: same
   completion + same test → identical reward across N calls. Already
   pinned end-to-end in Tier 8.2; could add a finer-grained per-row
   version.
4. **`extract_code` round-trip**: for every (chosen, rejected) pair in
   the builtin set, `extract_code(fenced(code)) == code` modulo
   whitespace. Pins the extractor against future fence-format drift.

**Risk:** the DPO margin test requires loading a tiny model. The repo
already pins `Qwen/Qwen2.5-Coder-0.5B` in `configs/dpo_3050.yaml`;
that's ~1 GB download and several seconds of forward time per test.
Tier 7.3 in nanogpt-edu got away with a hand-built 32-d model. For
DPO/GRPO we may need to either (a) cache a smaller tokenizer + a
2-layer 64-d transformers `LlamaForCausalLM` instance built from
scratch (no download), or (b) mark the DPO/GRPO ledger tests as
`@pytest.mark.slow` and skip by default. Recommend (a) — keeps the
suite hermetic, downloadable in CI without a HuggingFace token.

**Expected:** ~10 new tests, 50 → ~60.

---

## Tier 8.4 — docs refresh (planned)

Update the README to reflect the Tier 8 work, matching the structure
the other projects use:

1. New "Sandbox & safety" section between "Three training tracks" and
   "Speed & quality knobs":
   - Document the rlimit floor (what's bounded, what isn't)
   - Document the `mp_mode='spawn'` opt-in for untrusted models
   - Re-state the "use Docker for real untrusted models" caveat
2. Add `grpo.use_vllm` to the "Speed & quality knobs" table.
3. Update the test count in the layout section (24 → 60 or whatever
   8.3 settles on).
4. Add a "Recent changes" / "Tier 8" summary section near the bottom,
   matching the pattern in nanogpt-edu / midgpt READMEs.

**Expected:** no new tests, just docs.

---

## Files touched so far

```
coder-finetune/
├── cf_rl/
│   ├── grpo_train.py        # 8.1: divisibility validator
│   └── reward.py            # 8.1: async-def regex
│                            # 8.2: route through run_many batch
├── configs/
│   └── grpo_3050.yaml       # 8.1: num_generations 6 → 8
├── eval/
│   └── run_humaneval.py     # 8.1: extract_code + pass@k + eos + --seed
│                            # 8.2: rlimits, silencing, spawn opt-in, q.get fix
├── infer/
│   └── generate.py          # 8.1: dtype → torch_dtype
└── tests/
    ├── test_bugfixes.py     # 8.1: +13 tests
    └── test_sandbox.py      # 8.2: +13 tests
```

Untouched and still relevant: `cf_data/__init__.py`, `cf_pref/*`,
`cf_rl/prompts.py`, `train.py`, `infer/merge_lora.py`, all configs
except `grpo_3050.yaml`, all worked examples.

---

## How to resume

```bash
cd /home/support/dev-macrohard/LLM-playground/coder-finetune
.venv/bin/python -m pytest -q              # should be 50 passed
git log --oneline -5                       # confirm 766e3a1 is at HEAD
```

Then either:

- `do 8.3` — pedagogical pins + vLLM. Estimated ~10 new tests, one
  commit. Watch out for the DPO-margin-test model-loading issue
  flagged above (recommend a hand-built tiny Llama, not a download).
- `do 8.4` — README only, skip 8.3.
- Pause / pivot — the 8.1 + 8.2 work stands on its own; both are
  honest, bisectable, and the test bar is up from 24 → 50.

## Open questions / decisions deferred

1. **Parallel reward executor revisit.** The `ProcessPoolExecutor`
   approach (one fork per worker up-front, persistent queue) avoids
   the fork-from-threads race that killed the threaded attempt. Would
   need ~30 minutes to wire and verify. Not on the current plan; flag
   for a future Tier 9 if GRPO step time becomes a bottleneck on
   someone's real run.
2. **`max_workers` kwarg in `run_many`.** Currently silently dropped
   (`del max_workers`). Could either keep it as a forward-compat
   hint or remove it entirely. Left for the future parallel
   reimplementation to decide.
3. **Spawn-mode performance for trusted runs.** Spawn is ~10× slower
   per call than fork. If we ever want it as default for safety, we'd
   need a worker-pool that pays the spawn cost once. Same as #1
   really — future Tier.
