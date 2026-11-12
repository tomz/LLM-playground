# November repair profile: explicit objectives and safe evaluation

Use [the independent test/install profile](../../docs/testing-and-reproducibility.md).
The trainer validates the exact tested HF versions; do not install arbitrary
latest libraries or silently filter unsupported fields. QLoRA's bitsandbytes
backend is optional in `requirements-qlora.txt`; its hardware behavior is not
validated by CPU tests.

## Inputs and training

Active YAML recipes now pin model/tokenizer commits and any configured Magicoder
source. [references.json](../configs/references.json) records the metadata evidence.
Only metadata was queried during repair, not weights or dataset rows. Provision
approved snapshots explicitly; recipes default to `model.local_files_only: true`.
Revision identity does not grant a license or establish model/backend quality.

SFT recipes declare `train.loss_scope: all_tokens` to preserve the old objective
explicitly. `assistant_only` is supported only with a template containing working
`{% generation %}` markers and nonempty assistant targets after truncation. It
was tested with an in-memory local tokenizer/model. Record actual tokenizer and
template hashes. Do not merely flip this flag on an incompatible remote template.

Validation is source/group-aware, with exact-prompt connectivity preventing
split leakage even across different declared groups. Metadata, split membership
and hashes are retained. Empty/invalid validation is an error. This is not a
near-duplicate decontamination system or a sealed coding benchmark.

The repaired SDPA baseline disables packing and optional Liger kernels. Packing
requires a validated boundary-aware flash-attention path. Historical timing,
VRAM and quality tables refer to their original configuration/environment;
they are not predictions for these migrated recipes.

The tested preference trainer is **sigmoid DPO**. Full fine-tuning uses a frozen
copy of the supplied model, avoiding TRL's implicit reference reload. PEFT uses
its adapter-disabled base. ORPO and SimPO are not trainer integrations in this
profile; their standalone mathematical helpers remain educational references.
Unknown objective options and obsolete prompt-truncation fields fail explicitly.
Optional remote preference/MBPP data requires an immutable revision. MBPP rows
without meaningful assertion statements are rejected, not turned into `pass`.

GRPO's production entrypoint requires `execution.backend: docker` and an explicitly
reviewed local immutable image. The shipped recipe leaves `execution.image` unset
on purpose. Availability is checked before dataset/model loading, and unassessed
verdicts abort rather than becoming fabricated zero rewards. Reward calls retain
all completions and verdicts. Deterministic rewards can still be gamed; isolation
and reward integrity must both be tested.

## Local evaluator fixture

Provision a reviewed local image containing the configured Python interpreter;
no image is pulled, and there is no host-execution fallback. Docker access and
live containment were unavailable during this repair.

```bash
: "${PYTHON_IMAGE:?Set a trusted local immutable sha256 image ID}"
.venv/bin/python eval/run_humaneval.py --model out/lora/final \
  --python-image "$PYTHON_IMAGE" --problems examples/python_eval_smoke.json \
  --benchmark local-python --prompt-style raw --n-samples 1
```

The two supplied tasks are host-authored plumbing fixtures, **not HumanEval,
HumanEval+, or a capability benchmark**. Canonical base HumanEval requires a
pinned dataset revision; subsets remain explicitly labeled subsets. The runner
records task IDs, all samples/verdicts, generation settings, model/tokenizer
identities, and evaluator identity. Live execution tests require the explicit
local-image environment variable documented by the tests. Mocked Docker commands
are contract evidence only; container isolation and grader tampering need real
backend tests before use with untrusted model outputs.

## Merge and configuration export

Use a new output path and locally provisioned base/adapter/tokenizer artifacts:

```bash
.venv/bin/python infer/merge_lora.py --adapter out/lora/final \
  --out out/merged --dtype float32
.venv/bin/python infer/export_vllm.py --model out/merged \
  --out out/vllm-config --dtype float32
```

Merge validates stored or explicitly pinned base identity, tokenizer/template
fingerprints and source manifests, loads safetensors offline, and publishes
without clobbering existing paths. Legacy remote adapters lacking a revision
require an explicit pinned revision; they never silently merge with `main`.
An explicit base override is recorded as a different operation, not an identical
reproduction. Output receipts hash the actual payloads.

`export_vllm.py` is a **configuration emitter**, not weight quantization or a live
vLLM compatibility test. The test suite proves a nonzero tiny CPU LoRA delta
survives save/merge/reload within numerical tolerances and that tokenizer and
manifest identity survives. No pretrained Qwen model was fine-tuned or benchmarked.

## Real offline integration tests

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q --timeout=180 \
  tests/test_real_sft_repair.py tests/test_recipe_contracts.py \
  tests/test_export_provenance.py tests/test_remote_data_contracts.py
```

These construct random GPT-2-size *fixtures* and tokenizers locally. SFT exercises
both loss scopes and held-out evaluation; DPO exercises a frozen non-aliased
reference and a real update; GRPO exercises actual generation/optimization with
a non-executing reward double. That GRPO test is not a Python-execution or reward
quality result. Full model, QLoRA, vLLM, and live isolation evidence remain separate.
