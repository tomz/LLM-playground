"""Pinned TRL DPO preference training (sigmoid objective).

Run: python -m cf_pref.dpo_train --config configs/dpo_3050.yaml
SimPO/KTO educational losses remain in cf_pref.objectives. They are not trainer
integrations. TRL 1.3 does not implement SimPO or ship ORPOTrainer; fail early.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_dist import dist_env, rank0_print
from cf_pref import pairs as pref_data
from cf_provenance import check_tested_stack, digest, strict_config, trainer_evidence, write_json
from train import build_model_and_tokenizer, model_references, training_kwargs


def _make_config(config_cls, kwargs: dict):
    """Never drop requested semantics to accommodate a different TRL version."""
    if "loss_type" in kwargs and kwargs["loss_type"] not in ("sigmoid", ["sigmoid"]):
        raise ValueError("this tested DPO adapter supports only sigmoid; SimPO is not implemented by TRL 1.3")
    return strict_config(config_cls, kwargs)


def build_pref_config(cfg: dict):
    """Validate supported objectives and every preference option before loading."""
    from trl import DPOConfig
    p = cfg.get("pref", {})
    objective = p.get("objective", "dpo")
    if objective != "dpo":
        raise ValueError(f"pref.objective={objective!r} is unsupported in the tested TRL profile; use dpo")
    unknown = set(p) - {"objective", "loss_type", "beta"}
    if unknown:
        raise ValueError(f"unsupported pref settings: {sorted(unknown)}; no settings are silently dropped")
    kwargs = training_kwargs(cfg)
    kwargs.update(max_length=cfg["train"].get("max_seq_len", 1024),
                  beta=p.get("beta", 0.1), loss_type=p.get("loss_type", "sigmoid"))
    return _make_config(DPOConfig, kwargs)


def build_pref_trainer(model, tok, train_ds, cfg: dict):
    """Build DPO without an implicit, potentially unpinned reference reload.

    Full fine-tuning gets a frozen copy of the supplied weights. PEFT uses its
    adapter-disabled base as the reference and avoids an unnecessary full copy.
    """
    from copy import deepcopy
    from trl import DPOTrainer
    args = build_pref_config(cfg)
    reference = None
    if cfg["method"] == "full":
        reference = deepcopy(model).eval().requires_grad_(False)
    return DPOTrainer(model=model, ref_model=reference, args=args,
                      train_dataset=train_ds, processing_class=tok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    check_tested_stack()
    build_pref_config(cfg)
    model_references(cfg)
    torch.manual_seed(cfg["seed"])
    train_ds = pref_data.load(cfg["dataset"])
    if cfg["dataset"].get("max_examples"):
        train_ds = train_ds.select(range(min(cfg["dataset"]["max_examples"], len(train_ds))))
    model, tok = build_model_and_tokenizer(cfg)
    trainer = build_pref_trainer(model, tok, train_ds, cfg)
    evidence = trainer_evidence(trainer, cfg, objective="dpo", data={
        "n": len(train_ds), "rows_sha256": digest(list(train_ds))})
    if dist_env().is_main:
        write_json(Path(cfg["out_dir"]) / "run_manifest.json", evidence)
    trainer.train()
    final = Path(cfg["out_dir"]) / "final"
    trainer.save_model(str(final))
    if dist_env().is_main:
        tok.save_pretrained(final)
        write_json(final / "run_manifest.json", evidence)
    rank0_print(f"[pref] saved -> {final}; evaluation requires --python-image")


if __name__ == "__main__":
    main()
