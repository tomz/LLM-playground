"""Regression tests for PyTorch behaviour we'd otherwise re-trip on.

Each test pins a specific gotcha discovered during distgpt development.
When PyTorch fixes one of these in a future version, the test fails,
we can clean up the workaround and delete the pinned test.

Don't add general tests here — these are negative tests of upstream
behaviour, not of our own code.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def test_checkpoint_exception_is_not_an_exception_subclass():
    """torch.distributed.checkpoint.api.CheckpointException inherits from
    BaseException, NOT Exception. This means `try/except Exception` and
    `pytest.raises(Exception)` won't catch it.

    Discovered while wiring `load_ckpt:` validation:
    `pytest.raises(Exception)` failed to match a real CheckpointException
    raised by `dcp.load` when the path didn't exist. We work around by
    raising FileNotFoundError ourselves in
    `CheckpointManager.load_weights_only` BEFORE calling `dcp.load`. If
    this test starts failing, DCP made `CheckpointException` a regular
    Exception and the up-front validation can be relaxed.
    """
    from torch.distributed.checkpoint.api import CheckpointException
    assert issubclass(CheckpointException, BaseException)
    assert not issubclass(CheckpointException, Exception), (
        "torch.distributed.checkpoint.CheckpointException is now an "
        "Exception subclass — the up-front path-validation in "
        "CheckpointManager.load_weights_only can be relaxed and the "
        "test in test_recipes.py can use `pytest.raises(Exception)`."
    )


def test_safetensors_refuses_shared_storage_for_tied_embeddings():
    """safetensors.torch.save_file raises RuntimeError when two keys
    point at the same underlying storage. This bit us during HF export
    because tied embeddings means `tok_emb.weight is lm_head.weight`.

    Workaround: drop the duplicate before save and rely on HF's
    `tie_word_embeddings=True` config flag to recreate the tie on load
    (see distgpt/eval/export_hf.py).

    If this test starts failing, safetensors learned to handle shared
    storage (probably by writing one tensor + a metadata pointer for the
    other key) and the de-dup workaround can be removed.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        import pytest
        pytest.skip("safetensors not installed")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tf:
        path = tf.name

    shared = torch.zeros(4, 4)
    try:
        try:
            save_file({"a": shared, "b": shared}, path)
        except RuntimeError as e:
            assert "share memory" in str(e) or "shared" in str(e).lower(), (
                f"unexpected error from safetensors: {e}"
            )
            return
        # Made it here? safetensors no longer rejects.
        raise AssertionError(
            "safetensors now accepts shared-storage tensors — the lm_head "
            "drop in distgpt/eval/export_hf.py is no longer required."
        )
    finally:
        import os
        os.unlink(path)


def test_nn_module_getattr_still_raises_for_unset_underscore_attrs():
    """Sanity check: nn.Module.__getattr__ raises AttributeError for any
    unset attribute, regardless of leading-underscore patterns.

    This is the EXPECTED, current behaviour as of torch 2.11+. An earlier
    pre-2.11 anecdote claimed `torch.distributed.checkpoint` monkey-patched
    __getattr__ to silently return False for `_dist*` names; this test
    verifies that monkey-patch is not present so we know our attribute
    names are safe. If this test starts failing, the monkey-patch came
    back and we'll want to audit all `_*` attribute names on nn.Module
    subclasses across the codebase.
    """
    import torch.distributed.checkpoint  # noqa: F401  ensure the patch (if any) installs
    m = nn.Module()
    for name in ["_dist_anything", "_anything_else", "_dgpt_sp_enabled"]:
        try:
            _ = getattr(m, name)
        except AttributeError:
            continue
        raise AssertionError(
            f"nn.Module silently returned a value for unset attribute {name!r}; "
            "a monkey-patch is now intercepting __getattr__. Audit "
            "distgpt/parallel/tensor.py (SP marker) and any other `_*` "
            "attributes we set on nn.Module subclasses."
        )

