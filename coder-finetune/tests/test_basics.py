import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import data
from eval.run_humaneval import extract_code, build_program


def test_builtin_loader_returns_messages():
    ds = data.load_builtin(repeat=2)
    assert len(ds) > 0
    ex = ds[0]
    assert "messages" in ex
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_extract_code_handles_fenced_and_raw():
    fenced = "sure!\n```python\ndef f(): return 1\n```\nthat's it"
    assert "def f(): return 1" in extract_code(fenced)
    raw = "def g(): return 2"
    assert extract_code(raw) == raw


def test_build_program_includes_prompt_when_completion_has_no_def():
    prompt = "def add(a, b):\n    \"\"\"add a and b\"\"\"\n"
    completion = "```python\n    return a + b\n```"
    test = "def check(fn):\n    assert fn(1, 2) == 3"
    prog = build_program(prompt, completion, test, "add")
    assert "def add" in prog
    assert "check(add)" in prog


def test_build_program_uses_completion_when_self_contained():
    prompt = "def add(a, b):\n"
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    test = "def check(fn):\n    assert fn(2, 3) == 5"
    prog = build_program(prompt, completion, test, "add")
    assert prog.count("def add") == 1   # only the completion's version
