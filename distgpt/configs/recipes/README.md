# Recipe configs

These are pre-baked YAML configs for specific phases / regimes of training.
Use them with `python -m distgpt.cli train --config configs/recipes/<name>.yaml`.

| Recipe | Phase | What it does |
|---|---|---|
| `cooldown.yaml` | Post-pretrain anneal | Short LR-decay phase (~5–10% of base steps) on a high-quality data mix. Loads weights only via `load_ckpt:`. Picks up 1–2 pp on benchmarks. |
| `longctx_finetune.yaml` | Context extension | Extend a 4K base model to 32K via rope_base scaling (10000 → 500000) and SP=on. Fresh optim, fine-tune LR (5e-5). |
| `muon_speedrun_1b.yaml` | Base pretrain | 1B run with Muon + QK-norm + zero-init-proj all enabled — the highest-sample-efficiency configuration we've validated. |

## The `load_ckpt:` knob

Used by the cooldown and long-context recipes. When set, the trainer loads
ONLY the model weights from the given step directory, then starts at step
0 with a fresh optimizer, fresh data loader, and a fresh LR schedule. This
is what you want for any "start from a pretrained base, train a phase on
top" workflow.

Do NOT set `load_ckpt:` for ordinary resume — the trainer auto-resumes
from the latest checkpoint in `out_dir/run_id/ckpts/` when one exists. The
resume path also restores optim state and the loader cursor, which is
what you want for resuming an interrupted run, but NOT what you want for
starting a cooldown / fine-tune.

## Editing notes

* The `load_ckpt:` paths in the example recipes (`out/1b/gpt_1b/ckpts/step_000060000`)
  are placeholders — edit them to point at the actual checkpoint of your base run.
* `compile: true` + `sequence_parallel: true` + `activation_ckpt: full` do
  not currently compose cleanly under torch 2.5/2.6; the longctx recipe
  intentionally leaves `compile: false`.
* All recipes set `fp8: off`. Flip to `hybrid` if (a) you have
  transformer-engine installed and (b) you've swapped the model linears
  for `te.Linear` — see `distgpt/training/precision.py` for the contract.
