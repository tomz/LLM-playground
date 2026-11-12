"""Optional remote data paths must be pinned and cannot invent valid rewards."""
from __future__ import annotations

from unittest.mock import Mock

from datasets import Dataset
import datasets
import pytest

from cf_pref import pairs
from cf_rl import prompts

REVISION = "a" * 40  # mocked loader fixture, not a claimed public snapshot


@pytest.mark.parametrize("load", [lambda: pairs.load_hf("fixture/pairs"), lambda: prompts.load_mbpp()])
def test_unpinned_remote_data_fails_before_loading(monkeypatch, load):
    remote = Mock(side_effect=AssertionError("must not access remote data"))
    monkeypatch.setattr(datasets, "load_dataset", remote)
    with pytest.raises(ValueError, match="immutable"):
        load()
    remote.assert_not_called()


def test_preference_revision_reaches_actual_loader(monkeypatch):
    remote = Mock(return_value=Dataset.from_dict({"prompt": ["q"], "chosen": ["a"], "rejected": ["b"]}))
    monkeypatch.setattr(datasets, "load_dataset", remote)
    result = pairs.load({"source": "hf", "hf_name": "fixture/pairs", "revision": REVISION})
    remote.assert_called_once_with("fixture/pairs", split="train", revision=REVISION)
    assert len(result) == 1


def mbpp_row(tests):
    return {"text": "increase by one", "code": "def increase(x):\n    return x + 1",
            "test_list": tests}


def test_mbpp_revision_reaches_actual_loader(monkeypatch):
    remote = Mock(return_value=Dataset.from_list([mbpp_row(["assert increase(1) == 2"])]))
    monkeypatch.setattr(datasets, "load_dataset", remote)
    result = prompts.load({"source": "mbpp", "revision": REVISION})
    remote.assert_called_once_with("mbpp", "sanitized", split="train", revision=REVISION)
    assert result[0]["entry_point"] == "increase"
    assert "assert increase(1) == 2" in result[0]["test"]


@pytest.mark.parametrize("tests", [[], ["pass"], [""]])
def test_mbpp_cannot_create_a_vacuous_reward(monkeypatch, tests):
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: Dataset.from_list([mbpp_row(tests)]))
    with pytest.raises(ValueError, match="assertion"):
        prompts.load_mbpp(revision=REVISION)
