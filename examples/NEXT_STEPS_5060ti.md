# Next steps — after the P100 → RTX 5060 Ti switch

Status as of the reboot window. The example pipelines have been made
**device-agnostic** and prep is committed. The only things that *must* happen
on the new card are: re-pin the GPU UUID, then re-run the examples and commit
the regenerated reports.

> **The reboot itself is a human action.** This doc assumes you've already
> rebooted and the RTX 5060 Ti is now enumerated.

---

## 0. Confirm the card is present

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
```

Expect a line like `GPU 0: NVIDIA GeForce RTX 5060 Ti ... (UUID: GPU-xxxx)`.

- **Before reboot the only visible GPU was the Tesla P100**
  (`GPU-<p100-uuid>`). The 5060 Ti was NOT enumerated.
- The torch build is ready: `frontier-platform/.venv` is **torch 2.11.0+cu130**
  with `sm_120` in its arch list (Blackwell-capable). No reinstall needed.
- If `nvidia-smi` works but torch can't see the card:
  `frontier-platform/.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"`

---

## 1. Re-pin the GPU UUID in the example scripts

Examples 01–04 hardcode `CUDA_VISIBLE_DEVICES=<uuid>` in their `run.sh`.
The pre-reboot value was `GPU-<old-uuid>`, which **must be re-verified** — a
GPU/driver change can issue a new UUID.

```bash
# Auto-match by name (default pattern: "5060"):
tools/set_example_gpu.sh

# ...or match a more specific name / set an explicit UUID:
tools/set_example_gpu.sh "5060 Ti"
tools/set_example_gpu.sh --uuid GPU-xxxxxxxx-....

# Verify:
grep -n CUDA_VISIBLE_DEVICES examples/0[1-4]*/run.sh
```

This patches examples 01–04. **Do NOT touch `04b_max_throughput_p100`** — it is
the P100-specific run and stays pinned to the P100 UUID + `.venv-p100`.

---

## 2. Re-run all examples (ordered)

02 and 03 depend on 01's checkpoint/shards, so order matters. Use the runner:

```bash
examples/run_all.sh            # runs 01, 02, 03, 04 in order, tee's logs
# or a subset:
examples/run_all.sh 01 04
```

The runner refuses to start if the pinned UUID isn't among the visible GPUs
(bypass with `SKIP_GPU_CHECK=1` if you know better). Each example writes
`examples/<n>/run_<timestamp>.log` and regenerates its `result.md`.

Manual fallback (same effect):
```bash
cd examples/01_pretrain_shakespeare && bash run.sh
cd ../02_align_chain                && bash run.sh
cd ../03_moe_vs_dense               && bash run.sh
cd ../04_max_throughput             && bash run.sh
```

### What's now device-agnostic (no 5060 Ti numbers were hardcoded)
- **Example 04 MFU**: the theoretical fp16 peak is computed at runtime from the
  live device — `theoretical_fp16_tflops()` = SMs × cores/SM × 2 × max-SM-clock
  (×2 for Pascal packed-fp16). Validated on the P100: computes **19.0 TFLOPS vs
  the 18.7 spec (~1.8% off)**. On the 5060 Ti it reads the real Blackwell clock,
  SM count, and sm_120 → 128 cores/SM automatically.
- **Sanity check after the run**: the `[peak] theoretical fp16 = X TFLOPS` line
  in 04's output. This is the **CUDA-core** fp16 rate, NOT the tensor-core
  marketing number — so MFU% will look reasonable (tens of %), not tiny. If it
  reads `~approx` in the note, nvidia-smi's `clocks.max.sm` query failed and it
  fell back to 1.5 GHz — investigate before trusting MFU.
- **Examples 01/02** print the live `get_device_name()` into their `result.md`
  instead of "RTX 3050".
- **Example 04 autotune** now probes up to micro_batch=48 so the 16 GB card
  saturates (was capped at 16 for the 8 GB 3050). Expect a larger chosen batch.
- **04 nvidia-smi sampling** now targets the pinned UUID (was a hardcoded
  physical index `-i 1`, which would sample the wrong card post-switch).

---

## 3. Review and commit the regenerated reports

```bash
git status
git diff -- examples/*/result.md examples/README.md

# Spot-check 04's headline:
grep -E "device:|theoretical peak|MFU|peak memory" examples/04_max_throughput/result.md
```

Sanity expectations on the 5060 Ti vs the old 3050 baseline:
- tokens/sec **up** (more SMs, higher clock, 16 GB lets batch grow)
- peak memory may climb toward ~13–15 GiB at the larger autotuned batch
- mean GPU util should stay **>85%** (04 prints a `[warn]` if not)

Then commit:
```bash
git add examples/*/result.md examples/*/run_*.log examples/README.md
git commit -m "examples: regenerate results on RTX 5060 Ti (sm_120)"
```

(The `run_*.log` files are optional to commit — `out/` is gitignored but the
logs live next to run.sh. Drop them from the add if you'd rather not track.)

---

## 4. (Optional) Full CI pass now that the GPU is back

`tools/orchestrate.py` runs pytest + ruff across the five **core subprojects**
(not the examples). I deliberately did NOT run `--tests` during the prep window
because some subproject tests may touch CUDA and other workloads were live.

```bash
frontier-platform/.venv/bin/python tools/orchestrate.py            # tests + lint
frontier-platform/.venv/bin/python tools/orchestrate.py --lint     # quick, no GPU
```

Lint-only was dry-checked during prep and is **clean**. Note: `examples/**` is
excluded from root ruff by design, so example-script style isn't gated by CI.

---

## Quick reference — what changed during prep (committed `75a4621` + this batch)

| File | Change |
|---|---|
| `examples/04_max_throughput/run.py` | runtime device-derived fp16 peak; autotune→48; nvidia-smi by UUID; F821 cleanup |
| `examples/01/run.py`, `02/run.py` | emit live device name in result.md |
| `examples/*/README.md`, `examples/README.md` | drop hardcoded 3050; document device-agnostic MFU + UUID caveat |
| `tools/set_example_gpu.sh` | **new** — re-pin example GPU UUIDs by name or `--uuid` |
| `examples/run_all.sh` | **new** — ordered one-command examples runner with pin guard |

## Gotchas
- **Don't run `04b`** as part of the 5060 Ti batch — it's the P100 counterpart.
- If `set_example_gpu.sh` matches the wrong card (e.g. multiple NVIDIA GPUs),
  pass `--uuid` explicitly.
- If torch sees the card but kernels fail with "no kernel image", double-check
  you're using `frontier-platform/.venv` (cu130), not `.venv-p100` (cu121).
