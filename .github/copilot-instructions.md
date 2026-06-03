# LLM-playground — Copilot instructions

A monorepo of **five self-contained PyTorch projects** forming a deliberate
complexity ladder for building, training, fine-tuning, and serving GPT-class
models. Each subproject is independent: its own `README.md`, deps, and tests.

| Project | Scale | Focus |
|---|---|---|
| `nanogpt-edu/` | 10M–100M | Single-file educational GPT; char-level TinyShakespeare. |
| `midgpt/` | 124M–1.5B | GPT-2 scale; tiktoken BPE, AMP, grad-ckpt/accum, single-node DDP, resume. |
| `distgpt/` | 1B–70B | Multi-node FSDP2 + Tensor Parallel + Pipeline Parallel on a 3D mesh, DCP checkpoints. |
| `coder-finetune/` | 0.5B–7B | SFT / LoRA / QLoRA + GRPO/RLVR on HF `transformers`+`peft`+`trl`; HumanEval+ in Docker. |
| `frontier-platform/` | 1B–500B+ | Architecture-only blueprint; typed skeletons raise `NotImplementedError`. |

## Build / test / lint

There is **no top-level build** — each subproject installs and runs from its
own directory.

```bash
# Install (per subproject)
cd <subproject>
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # nanogpt-edu, midgpt, coder-finetune
pip install -e .                                                    # distgpt, frontier-platform (have pyproject.toml)
```

Tests run **without installing** the package (a `sys.path.insert(0, parents[1])`
shim makes this work):

```bash
cd <subproject> && pytest          # full suite for one project
cd midgpt && pytest tests/test_model.py                 # single file
cd midgpt && pytest tests/test_model.py::test_cosine_lr # single test
cd midgpt && pytest -k muon                              # by keyword
```

Run the whole repo (pytest in each subproject + root ruff) via the orchestrator:

```bash
python3 tools/orchestrate.py            # tests + lint
python3 tools/orchestrate.py --tests    # tests only
python3 tools/orchestrate.py --lint     # lint only (ruff at root)
python3 tools/orchestrate.py -p midgpt  # one project (repeatable)
```

Lint is repo-wide, configured in the **root `pyproject.toml`** only:

```bash
ruff check .
```

CI (`.github/workflows/tests.yml`) mirrors this: a per-project pytest matrix
(CPU-only torch, `--timeout=180`) plus one root `ruff check .` job.

## Architecture — the complexity ladder

The five projects are meant to be read in order; each reuses the previous
project's vocabulary and adds **one** production concern:

```
nanogpt-edu → midgpt → distgpt → coder-finetune → frontier-platform
  minimal     real      3D          post-training     the whole system
  transformer tokenizer parallelism (LoRA/QLoRA/GRPO) around training
```

- `coder-finetune` is the **orthogonal track**: it starts from pretrained
  weights and aligns them for code, rather than pretraining from scratch.
- `frontier-platform` zooms out to the dozen production systems *around* the
  training loop (data → filter → dedup → tokenizer → pretrain → SFT → RLHF/DPO
  → eval → red-team → serving → observability). Its `docs/` (14 design docs)
  are the source of truth — **read them before touching `platform/`**.

Shared *conceptual* abstractions recur across (but are not imported between)
projects: `ModelConfig.param_count()`, `cosine_with_warmup` LR schedule,
`SpikeMonitor`/`RewindController` for loss-spike detection, sharded streaming
resumable dataloaders, exact/near-duplicate text dedup.

### Per-project layout conventions
- Smaller projects (`nanogpt-edu`, `midgpt`) are **one-file-per-concern**:
  `model.py`, `data.py`, `train.py`, `sample.py` (+ `prepare*.py`, `eval.py`).
- Larger projects are **package-per-concern** (e.g.
  `distgpt/distgpt/{model,parallel,data,training}/`).
- YAML configs live in `configs/`; checkpoints and artifacts go in `out/`.
  (`nanogpt-edu` uses Python config files instead of YAML.)
- Multi-GPU runs in `midgpt`/`distgpt` are launched via `torchrun`.

## Conventions

- Target **Python ≥ 3.10**. Every module starts with
  `from __future__ import annotations`; use PEP-604 unions (`str | None`) and
  built-in generics (`dict`, `list`, `tuple[str, float]`).
- Use `@dataclass` for configs and small value types (`ModelConfig`,
  `OptimConfig`, `FilterVerdict`, …).
- **Skeleton modules** in `frontier-platform/` are docstring + signature +
  `raise NotImplementedError`; tests deliberately skip these and only exercise
  the pure-Python helpers that *are* implemented.
- Module-level docstrings cite the production reference they reimplement (e.g.
  "Heuristic quality filter from Rae et al. 2021 (Gopher paper)").
- Compact code style is intentional and reflected in the ruff ignore list
  (`E501`, `E701/2`, `E731`, `E741`, `E401`, `E402`) — don't reformat to
  "fix" these. `E402` is ignored because many files insert `sys.path` before
  importing.

## Do not

- **No cross-subproject imports** — each project is deliberately standalone.
- **No top-level build/test runner** beyond `tools/orchestrate.py`; tests are
  run from each subproject directory.
- **Don't implement `NotImplementedError` stubs** in `frontier-platform/`
  without a corresponding design-doc update in its `docs/`.

> See `JAAICODE.md` for the original AI-assistant guidance this file builds on,
> and each subproject's `README.md` for project-specific recipes and results.
