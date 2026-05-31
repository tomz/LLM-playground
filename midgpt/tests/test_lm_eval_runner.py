"""lm_eval_runner: ImportError surfaces correctly when lm-eval is missing;
the CLI parses arguments.

Light-touch test — the real value is the round-trip you get from
``test_export_hf.py`` (which proves the HF model produced by export_to_hf is
loadable and produces correct logits). The lm-eval-harness integration itself
is too heavy to install in CI, so we only check the structural plumbing here.
"""
from __future__ import annotations
import sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_lm_eval_runner_importable():
    """The module itself must import even when lm-eval isn't installed (the
    heavy import is deferred to ``run_lm_eval()``). This catches the bug
    class where a top-level ``from lm_eval import ...`` would silently break
    ``pip install midgpt && python lm_eval_runner.py --help`` on a clean venv."""
    import importlib
    mod = importlib.import_module("lm_eval_runner")
    assert hasattr(mod, "run_lm_eval")
    assert hasattr(mod, "main")


def test_lm_eval_runner_raises_with_install_hint(tmp_path):
    """When lm-eval isn't installed, ``run_lm_eval`` must raise ImportError
    whose message mentions the pip install command — UX promise so users
    don't have to grep the source for the right package name."""
    try:
        import lm_eval  # noqa: F401
        pytest.skip("lm-eval is installed; can't test the missing-dep path")
    except ImportError:
        pass
    from lm_eval_runner import run_lm_eval
    # Pass a non-existent ckpt — we never get that far because the lm-eval
    # import fails first. The error message must guide the user to fix it.
    with pytest.raises(ImportError, match="pip install lm-eval"):
        run_lm_eval(ckpt_path=str(tmp_path / "no.pt"), tasks=["x"], device="cpu")


def test_lm_eval_runner_cli_help(capsys):
    """`python lm_eval_runner.py --help` must exit cleanly (returns 0); a
    typo in the argparse setup would crash here before lm-eval is even
    looked for."""
    import subprocess, sys as _sys
    import importlib
    mod = importlib.import_module("lm_eval_runner")
    p = subprocess.run(
        [_sys.executable, mod.__file__, "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert p.returncode == 0, p.stderr
    assert "--ckpt" in p.stdout
    assert "--tasks" in p.stdout
