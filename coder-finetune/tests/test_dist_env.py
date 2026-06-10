"""Tier 9 regression tests: the read-only distributed-env layer (``cf_dist.py``)
and its one cross-module contract — the GRPO divisibility validator must scale
the effective batch by ``WORLD_SIZE`` (accelerate's ``num_processes``), never by
``torch.cuda.device_count()``.

The bugs / contracts these pin:
  * ``dist_env()`` must fall back to the single-process identity
    (rank=0, local_rank=0, world_size=1) when the launcher published nothing,
    so a plain ``python train.py`` behaves byte-for-byte as it did before this
    module existed.
  * ``_env_int`` must tolerate an empty/garbage env value (some launchers export
    ``RANK=`` rather than unsetting it) instead of crashing the entry point.
  * ``placement_device_map`` must return ``None`` on a single process (let
    Trainer/accelerate place the model) and ``{"": local_rank}`` under DDP (each
    rank pins its own quantized replica — bitsandbytes can't be relocated after
    load).
  * **The headline cross-module contract:** ``build_grpo_trainer``'s divisibility
    check reads ``dist_env().world_size``. A config whose *single-process*
    effective batch is NOT divisible by ``num_generations`` becomes valid at
    ``WORLD_SIZE=2`` (world doubles the batch) — proving the validator consults
    the real process count, not ``device_count()``. This is the bug that made
    the old check both miss real mismatches and reject valid DDP configs.
"""
from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cf_dist import DistEnv, dist_env, placement_device_map, rank0_print


# ---------------------------------------------------------------------------
# dist_env(): single-process fallback + env parsing
# ---------------------------------------------------------------------------


def _clear_dist_env(monkeypatch):
    """Remove every distributed env var so we observe the bare fallback."""
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(k, raising=False)


def test_dist_env_single_process_identity(monkeypatch):
    """With nothing published, the snapshot is the single-process identity and
    ``is_distributed`` is False — the documented plain-``python`` path."""
    _clear_dist_env(monkeypatch)
    env = dist_env()
    assert (env.rank, env.local_rank, env.world_size) == (0, 0, 1)
    assert env.is_main is True
    assert env.is_distributed is False


def test_dist_env_reads_published_topology(monkeypatch):
    """Under a 2-process launch, rank 1 reads its real coordinates from the env
    and is no longer ``is_main``."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    env = dist_env()
    assert (env.rank, env.local_rank, env.world_size) == (1, 1, 2)
    assert env.is_main is False
    assert env.is_distributed is True


def test_dist_env_tolerates_empty_env_values(monkeypatch):
    """Some launchers export an empty ``RANK=`` instead of unsetting it; a blank
    or non-integer value must degrade to the default, not raise ValueError and
    take down the entry point before training even starts."""
    monkeypatch.setenv("RANK", "")          # blank
    monkeypatch.setenv("LOCAL_RANK", "  ")  # whitespace
    monkeypatch.setenv("WORLD_SIZE", "garbage")
    env = dist_env()
    assert (env.rank, env.local_rank, env.world_size) == (0, 0, 1)


def test_dist_env_is_frozen():
    """The snapshot is a *view* of launcher state — nothing in this codebase may
    mutate the topology (accelerate owns that), so the dataclass is frozen."""
    env = DistEnv(rank=0, local_rank=0, world_size=1)
    with pytest.raises(Exception):
        env.world_size = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# placement_device_map(): QLoRA-under-DDP device pinning
# ---------------------------------------------------------------------------


def test_placement_device_map_none_on_single_process(monkeypatch):
    """Single process → return None so Trainer/accelerate owns placement (the
    current, working single-GPU path). A ``device_map`` here would wrongly
    pre-empt accelerate's ``.to(device)`` for plain LoRA / full FT."""
    _clear_dist_env(monkeypatch)
    assert placement_device_map() is None


def test_placement_device_map_pins_local_rank_under_ddp(monkeypatch):
    """Under DDP each rank must pin its quantized replica to its *own* GPU —
    bitsandbytes places 4-bit weights at load time and accelerate can't move
    them afterwards. Rank 1 must map to cuda:1, not cuda:0 (the OOM/duplicate
    failure mode)."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert placement_device_map() == {"": 1}


def test_placement_device_map_accepts_explicit_env():
    """The helper takes an explicit ``DistEnv`` (testability / call-site that
    already snapshotted) and honors its ``local_rank``."""
    assert placement_device_map(DistEnv(rank=3, local_rank=3, world_size=4)) == {"": 3}
    assert placement_device_map(DistEnv(rank=0, local_rank=0, world_size=1)) is None


# ---------------------------------------------------------------------------
# rank0_print(): de-duplication gate
# ---------------------------------------------------------------------------


def test_rank0_print_fires_only_on_main(monkeypatch, capsys):
    """On rank 0 it prints; on a non-zero rank it is silent — so a DDP run emits
    each status line once, not ``WORLD_SIZE`` times interleaved."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    rank0_print("hello-from-main")
    assert "hello-from-main" in capsys.readouterr().out

    monkeypatch.setenv("RANK", "1")
    rank0_print("should-be-silent")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Cross-module: the GRPO validator scales the batch by WORLD_SIZE
# ---------------------------------------------------------------------------


def _minimal_grpo_cfg(batch_size, grad_accum, num_generations):
    """Smallest cfg dict that reaches build_grpo_trainer's divisibility check
    without a real model load (mirrors test_bugfixes._minimal_grpo_cfg)."""
    return {
        "out_dir": "/tmp/x",
        "seed": 0,
        "method": "lora",
        "model": {"dtype": "bfloat16"},
        "train": {
            "batch_size": batch_size, "grad_accum": grad_accum, "epochs": 1,
            "lr": 1e-5, "warmup_ratio": 0.0, "weight_decay": 0.0,
            "grad_clip": 1.0, "log_every": 1, "save_every": 1,
            "max_seq_len": 64, "gradient_checkpointing": True,
        },
        "grpo": {"num_generations": num_generations},
    }


def test_grpo_validator_uses_world_size_not_device_count(monkeypatch):
    """The crux of the cf_dist fix, pinned end-to-end.

    Config: bs=2, accum=2, G=8.
      * single process: effective = 2*2*1 = 4, NOT divisible by 8 → the
        validator must SystemExit naming ``num_generations``.
      * WORLD_SIZE=2:  effective = 2*2*2 = 8, divisible by 8 → the validator
        must NOT fire (the call fails *later* on the None model instead).

    If the validator used ``torch.cuda.device_count()`` this test's verdict
    would depend on the host's GPU count instead of the launched world size —
    exactly the bug. Driving it purely through ``WORLD_SIZE`` proves the batch
    is scaled by the process count.
    """
    from cf_rl.grpo_train import build_grpo_trainer

    cfg = _minimal_grpo_cfg(batch_size=2, grad_accum=2, num_generations=8)

    # --- single process: 4 not divisible by 8 → divisibility SystemExit ---
    _clear_dist_env(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        build_grpo_trainer(model=None, tok=None, train_ds=None,
                           reward_funcs=[], cfg=cfg)
    msg = str(exc.value)
    assert "num_generations" in msg
    assert "world=1" in msg  # the message reports the world size it used

    # --- WORLD_SIZE=2: 8 divisible by 8 → must pass divisibility, fail later ---
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(Exception) as exc2:
        build_grpo_trainer(model=None, tok=None, train_ds=None,
                           reward_funcs=[], cfg=cfg)
    # Whatever it failed on, it must NOT be the divisibility SystemExit.
    if isinstance(exc2.value, SystemExit):
        assert "num_generations" not in str(exc2.value), \
            "validator wrongly fired at WORLD_SIZE=2 (effective batch 8 % G=8 == 0)"


def test_grpo_validator_message_names_world_size(monkeypatch):
    """The actionable error must surface the world size it computed so a user
    debugging a DDP launch can see whether WORLD_SIZE reached the validator."""
    from cf_rl.grpo_train import build_grpo_trainer

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    # bs=3, accum=1, world=2 → effective 6, G=8 → 6 % 8 != 0 → raises, and the
    # message must mention both num_generations and the WORLD_SIZE hint.
    cfg = _minimal_grpo_cfg(batch_size=3, grad_accum=1, num_generations=8)
    with pytest.raises(SystemExit) as exc:
        build_grpo_trainer(model=None, tok=None, train_ds=None,
                           reward_funcs=[], cfg=cfg)
    msg = str(exc.value)
    assert "num_generations" in msg
    assert "world=2" in msg
    assert "WORLD_SIZE" in msg  # points the user at the launch knob
