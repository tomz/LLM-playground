"""Offline search acceptance tests: safe verification, budgets, real CPU inference."""
from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from research_model import ByteTokenizer, GPT2Tokenizer, Generation, LocalModelAdapter
from research_search import NumericTask, SearchBudget, benchmark_search, main, run_search


@pytest.fixture(autouse=True)
def offline_single_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def no_network(*args, **kwargs):
        raise AssertionError("network is forbidden in search tests")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    yield
    torch.set_num_threads(previous)


def generated(text: str) -> Generation:
    return Generation(text, tuple(text.encode()))


@pytest.mark.parametrize("text", [
    "", "42", "[]", "null", '{"sum": "42"}', '{"sum": 42.0}',
    '{"sum": true}', '{"sum": 41}', '{"sum": 42, "extra": 0}',
    '{"sum": 42, "sum": 42}', '{"sum": NaN}', '{"sum": Infinity}',
    '{"sum": 42} trailing', '```json\n{"sum":42}\n```',
    '__import__("os").system("echo unsafe")', "[" * 2000,
    " " * 16385 + '{"sum":42}',
])
def test_checker_rejects_malformed_or_wrong_outputs(text):
    assert NumericTask().verify(text) == (False, 0.0)


@pytest.mark.parametrize("left,right", [(19, 23), (-19, 23), (0, 0), (-1_000_000, 1_000_000)])
def test_checker_uses_task_operands(left, right):
    task = NumericTask(left, right)
    assert task.verify(json.dumps({"sum": left + right}))[0]
    assert not task.verify(json.dumps({"sum": left + right + 1}))[0]


def test_all_branches_count_even_after_success():
    answers = iter(["bad", '{"sum":42}', '{"sum":42}'])
    report = run_search(NumericTask(), lambda *_: generated(next(answers)),
                        SearchBudget(3, 32, 96), model_info={"kind": "test-fixture"}, setup_s=0.1)
    summary = report.to_dict()["summary"]
    assert [row.status for row in report.attempts] == ["invalid", "accepted", "accepted"]
    assert report.valid_rate == 2 / 3
    assert report.first_valid == report.attempts[1]
    assert summary["accepted_tasks"] == 1  # not 2 valid candidates
    assert summary["tokens_per_accepted_task"] == 23
    assert summary["seconds_per_accepted_task"] >= 0.1
    assert summary["retries"] == 2
    assert summary["token_accounting_complete"]


def test_total_budget_shrinks_final_call_and_prevents_extra_attempt():
    allowances = []

    def generate(prompt, allowance, seed):
        allowances.append((allowance, seed))
        return Generation("x" * allowance, (120,) * allowance)

    report = run_search(NumericTask(), generate, SearchBudget(5, 4, 10),
                        model_info={}, seed=7)
    assert allowances == [(4, 7), (4, 8), (2, 9)]
    assert report.generation_tokens == 10
    assert report.stop_reason == "token_budget"
    summary = report.to_dict()["summary"]
    assert summary["accepted_tasks"] == 0
    assert summary["tokens_per_accepted_task"] is None
    assert summary["seconds_per_accepted_task"] is None
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("candidates", 0), ("candidates", 129), ("candidates", True),
    ("max_new_tokens", -1), ("max_new_tokens", 513),
    ("total_tokens", 0), ("total_tokens", 8193), ("total_tokens", 1.5),
])
def test_invalid_budgets_rejected(field, value):
    with pytest.raises(ValueError):
        SearchBudget(**{field: value})


def test_generator_exception_reserves_unknown_work_and_retries_within_budget():
    calls = []

    def fail(prompt, allowance, seed):
        calls.append(allowance)
        raise RuntimeError("failed generation")

    report = run_search(NumericTask(), fail, SearchBudget(9, 4, 6), model_info={})
    assert calls == [4, 2]
    assert all(row.status == "generation_error" for row in report.attempts)
    assert all("RuntimeError" in row.error for row in report.attempts)
    assert all(row.generation_s >= 0 for row in report.attempts)
    assert report.generation_tokens == 0  # unknown is not fabricated as emitted tokens
    assert report.to_dict()["summary"]["budget_tokens"] == 6
    assert not report.to_dict()["summary"]["token_accounting_complete"]


def test_unknown_attempt_prevents_measured_token_cost_after_later_success():
    calls = []

    def generate(prompt, allowance, seed):
        calls.append(seed)
        if len(calls) == 1:
            raise RuntimeError("unknown emitted tokens")
        return Generation('{"sum":42}', (1, 2, 3))

    report = run_search(NumericTask(), generate, SearchBudget(2, 4, 8), model_info={})
    summary = report.to_dict()["summary"]
    assert summary["accepted_tasks"] == 1
    assert summary["generation_tokens"] == 3
    assert not summary["token_accounting_complete"]
    assert summary["tokens_per_accepted_task"] is None


def test_partial_generation_error_is_retained_and_not_verified():
    def must_not_check(candidate):
        pytest.fail("partial/error output must not be accepted")

    report = run_search(NumericTask(), lambda *_: Generation("{", (123,), 5, 11, "failure"),
                        SearchBudget(1, 4, 4), model_info={}, verifier=must_not_check,
                        checker_version="test-never-v1")
    row = report.attempts[0]
    assert (row.candidate, row.generation_tokens, row.budget_tokens) == ("{", 1, 4)
    assert row.forward_input_tokens == 11
    assert row.verifier_s == 0
    assert not row.valid


@pytest.mark.parametrize("answer", [None, Generation("x", (1, 2, 3)), Generation("x", (-1,))])
def test_bad_adapter_contract_is_charged_and_recorded(answer):
    report = run_search(NumericTask(), lambda *_: answer, SearchBudget(1, 2, 2), model_info={})
    assert report.attempts[0].status == "generation_error"
    assert report.attempts[0].budget_tokens == 2


@pytest.mark.parametrize("mode", ["raises", "nan", "negative", "not-bool"])
def test_checker_failures_are_counted_without_acceptance(mode):
    def check(candidate):
        if mode == "raises":
            raise RuntimeError("checker broke")
        return {"nan": (True, float("nan")), "negative": (True, -1),
                "not-bool": ("yes", 0)}[mode]

    report = run_search(NumericTask(), lambda *_: generated('{"sum":42}'),
                        SearchBudget(2, 32, 64), model_info={},
                        verifier=check, checker_version="test-checker-v1")
    assert len(report.attempts) == 2
    assert all(row.status == "checker_error" and not row.valid for row in report.attempts)
    assert report.generation_tokens == 20
    assert report.verifier_s > 0
    assert report.accepted_tasks == 0


def test_custom_checker_requires_honest_version():
    with pytest.raises(ValueError, match="checker_version"):
        run_search(NumericTask(), lambda *_: generated("x"), SearchBudget(), model_info={},
                   verifier=lambda _: (True, 0))


def test_legacy_api_does_not_pull_an_extra_candidate():
    def stream(prompt, attempts):
        yield "ok"
        raise AssertionError("must not advance beyond budget")

    report = benchmark_search("x", stream, lambda _: (True, 0.2), 1)
    assert report.first_valid.candidate == "ok"
    assert report.verifier_s >= 0.2
    assert report.attempts[0].token_count_kind == "character-estimate"


def test_legacy_records_initial_and_iteration_errors():
    def initial(prompt, attempts):
        raise RuntimeError("initial")

    def stream(prompt, attempts):
        yield "bad"
        raise RuntimeError("iteration")

    for generate, count in [(initial, 1), (stream, 2)]:
        report = benchmark_search("x", generate, lambda _: (False, 0), 3)
        assert len(report.attempts) == count
        assert report.attempts[-1].status == "generation_error"
        assert report.stop_reason == "generation_error"
    empty = benchmark_search("x", lambda *_: [], lambda _: (True, 0), 1)
    assert empty.valid_rate == 0
    assert empty.to_dict()["summary"]["tokens_per_accepted_task"] is None


def test_tiny_uses_real_gpt_generation_and_preserves_rng(monkeypatch):
    adapter = LocalModelAdapter.tiny(seed=3)
    calls = []
    actual = adapter.model.generate

    def counted(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(adapter.model, "generate", counted)
    state = torch.get_rng_state().clone()
    first = adapter.generate(NumericTask().prompt, 4, 10)
    second = adapter.generate(NumericTask().prompt, 4, 99)
    assert torch.equal(state, torch.get_rng_state())
    assert first == second  # greedy regardless of sampling seed
    assert first.error is None
    assert len(calls) == 2 * len(first.token_ids)
    assert 1 <= len(first.token_ids) <= 4
    assert first.prompt_tokens == len(NumericTask().prompt.encode())
    assert first.forward_input_tokens >= first.prompt_tokens
    assert adapter.provenance["trained"] is False
    assert len(adapter.provenance["weights_sha256"]) == 64
    changed = LocalModelAdapter.tiny(seed=4)
    assert changed.provenance["weights_sha256"] != adapter.provenance["weights_sha256"]


def test_actual_model_failure_keeps_partial_ids_and_attempted_work(monkeypatch):
    adapter = LocalModelAdapter.tiny()
    actual = adapter.model.generate
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected forward failure")
        return actual(*args, **kwargs)

    monkeypatch.setattr(adapter.model, "generate", fail_second)
    result = adapter.generate(NumericTask().prompt, 4, 0)
    assert len(result.token_ids) == 1
    assert "injected forward failure" in result.error
    assert result.forward_input_tokens == 2 * result.prompt_tokens + 1


def test_prompt_limit_fails_before_forward(monkeypatch):
    adapter = LocalModelAdapter.tiny()
    monkeypatch.setattr(adapter.model, "generate", lambda *_: pytest.fail("must not run model"))
    result = adapter.generate("x" * 257, 4, 0)
    assert "context" in result.error
    assert result.token_ids == ()
    assert result.forward_input_tokens == 0


def test_checkpoint_roundtrip_with_compiled_prefix_and_cli(tmp_path):
    adapter = LocalModelAdapter.tiny(11)
    path = tmp_path / "local.pt"
    torch.save({"cfg": {"model": asdict(adapter.model.cfg), "tokenizer": "byte-v1"},
                "model": {"_orig_mod." + k: v for k, v in adapter.model.state_dict().items()}}, path)
    loaded = LocalModelAdapter.from_checkpoint(path)
    assert loaded.generate(NumericTask().prompt, 3, 0) == adapter.generate(NumericTask().prompt, 3, 0)
    assert len(loaded.provenance["checkpoint_sha256"]) == 64
    output = tmp_path / "report.json"
    assert main(["--checkpoint", str(path), "--output", str(output),
                 "--candidates", "1", "--max-new-tokens", "3", "--total-tokens", "3"]) == 0
    report = json.loads(output.read_text())
    assert report["metadata"]["model"]["kind"] == "local-checkpoint"
    assert report["summary"]["generation_tokens"] > 0
    assert report["attempts"][0]["status"] != "generation_error"


def test_checkpoint_never_falls_back_to_unsafe_pickle(tmp_path, monkeypatch):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a checkpoint")
    calls = []

    def restricted_load(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("unsupported pickle")

    monkeypatch.setattr(torch, "load", restricted_load)
    with pytest.raises(ValueError, match="unsupported pickle"):
        LocalModelAdapter.from_checkpoint(path)
    assert len(calls) == 1 and calls[0]["weights_only"] is True


def test_gpt2_loader_uses_local_vocab_without_registry_download(tmp_path, monkeypatch):
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda *_: pytest.fail("remote-capable registry forbidden"))
    # Synthetic GPT-2-shaped vocabulary: tests the byte/rank conversion offline,
    # not equivalence to pretrained GPT-2's actual vocabulary or model quality.
    visible = [b for b in range(256) if chr(b).isprintable() and chr(b) != " "]
    chars = [chr(b) for b in visible] + [chr(256 + i) for i in range(256 - len(visible))]
    vocab = {c: i for i, c in enumerate(chars)}
    for i in range(50000):
        vocab[chars[i // 256] + chars[i % 256]] = i + 256
    vocab["<|endoftext|>"] = 50256
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab))
    tokenizer = GPT2Tokenizer(path)
    text = 'Add 19 and 23.\n{"sum":42}'
    assert tokenizer.decode(tokenizer.encode_ordinary(text)) == text
    assert len(tokenizer.provenance["vocab_sha256"]) == 64
    with pytest.raises(ValueError, match="local file"):
        GPT2Tokenizer(pathlib.Path("https://example.invalid/vocab.json"))


def test_experiment_identity_changes_with_task_and_budget():
    def experiment(task, budget):
        return run_search(task, lambda *_: generated("x"), budget, model_info={"id": "fixture"})

    first = experiment(NumericTask(), SearchBudget(1, 2, 2))
    assert first.metadata["experiment_id"] == experiment(NumericTask(), SearchBudget(1, 2, 2)).metadata["experiment_id"]
    assert first.metadata["experiment_id"] != experiment(NumericTask(1, 2), SearchBudget(1, 2, 2)).metadata["experiment_id"]
    assert first.metadata["experiment_id"] != experiment(NumericTask(), SearchBudget(2, 2, 2)).metadata["experiment_id"]


def test_deterministic_offline_cli_smoke(tmp_path):
    script = pathlib.Path(__file__).resolve().parents[1] / "research_search.py"
    reports = []
    for index in range(2):
        output = tmp_path / f"smoke-{index}.json"
        result = subprocess.run(
            [sys.executable, str(script), "--tiny-smoke", "--seed", "0", "--candidates", "2",
             "--max-new-tokens", "4", "--total-tokens", "6", "--output", str(output)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        reports.append(json.loads(output.read_text()))
    first, second = reports
    assert first["metadata"] == second["metadata"]
    assert [row["token_ids"] for row in first["attempts"]] == [row["token_ids"] for row in second["attempts"]]
    assert first["summary"]["generation_tokens"] == 6
    assert first["summary"]["accepted_tasks"] == 0
    assert first["summary"]["seconds_per_accepted_task"] is None
    assert first["stop_reason"] == "token_budget"
    assert all(row["status"] == "invalid" for row in first["attempts"])


@pytest.mark.parametrize("args", [
    ["--candidates", "0"], ["--temperature", "nan"], ["--temperature", "-1"],
    ["--left", "1000001"], ["--seed", "-1"],
])
def test_cli_rejects_invalid_settings(args, tmp_path):
    output = tmp_path / "must-not-exist.json"
    assert main(["--tiny-smoke", "--output", str(output), *args]) == 2
    assert not output.exists()
