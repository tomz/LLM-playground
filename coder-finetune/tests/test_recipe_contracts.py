"""Every shipped recipe must construct under the real pinned trainer APIs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from cf_pref.dpo_train import build_pref_config
from cf_provenance import check_tested_stack
from cf_rl.grpo_train import build_grpo_config
from eval.python_jail import executor_from_config
from train import build_sft_config, model_references

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
REFERENCES = json.loads((CONFIGS / "references.json").read_text())


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda path: path.name)
def test_shipped_recipe_has_pinned_identities_and_supported_config(path):
    check_tested_stack()
    cfg = yaml.safe_load(path.read_text())
    model, tokenizer = model_references(cfg)
    assert model["revision"] == REFERENCES["models"][cfg["model"]["name"]]
    assert tokenizer["revision"] == model["revision"]
    assert cfg["model"]["local_files_only"] is True
    if cfg["dataset"]["source"] == "hf":
        assert cfg["dataset"]["revision"] == REFERENCES["datasets"][cfg["dataset"]["hf_name"]]
    # Construct only configs: no weights/datasets, GPU allocation or training.
    cpu = deepcopy(cfg)
    cpu["model"]["dtype"] = "float32"
    cpu["train"]["use_cpu"] = True
    if "pref" in cfg:
        build_pref_config(cpu)
    elif "grpo" in cfg:
        build_grpo_config(cpu)
        assert cfg["execution"]["backend"] == "docker"
        assert cfg["execution"]["image"] is None
        with pytest.raises(ValueError, match="image"):
            executor_from_config(cfg["execution"])
    else:
        assert cfg["train"]["loss_scope"] in {"all_tokens", "assistant_only"}
        assert cfg["train"]["packing"] is False
        assert not cfg["train"].get("use_liger_kernel", False)
        assert 0 < cfg["dataset"]["validation_fraction"] < 1
        assert not cfg.get("eval", {}).get("enable", False)
        build_sft_config(cpu)


def test_legacy_in_process_code_eval_flag_is_not_silently_ignored():
    cfg = yaml.safe_load((CONFIGS / "tiny.yaml").read_text())
    cfg["eval"]["enable"] = True
    with pytest.raises(ValueError, match="isolated"):
        build_sft_config(cfg)
