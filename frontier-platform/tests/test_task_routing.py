"""CPU conformance tests for cost accounting and deployment-aware routing."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json

import pytest

from frontier_platform.serving.task_cost import Tariff, TaskLedger, Usage
from frontier_platform.serving.task_routing import (
    BackendFailure, ModelReply, Route, TaskRequest, run_task,
)


def tariff(model="small", **overrides):
    fields = dict(tariff_id=f"{model}-v1", model=model, source="synthetic-test",
                  valid_from="2026-11-01", valid_through="2026-11-30",
                  input_usd="1", output_usd="5", cache_read_usd="0.1", cache_write_usd="2")
    fields.update(overrides)
    return Tariff(**fields)


def route(model="small"):
    return Route(model, "rev1", "low", "public", tariff(model), Usage(20, 10, 5, 20))


def reply(r, text="42", **overrides):
    fields = dict(text=text, model=r.model, revision=r.revision,
                  safeguard_mode=r.safeguard_mode, usage=Usage(10, 2, 1, 5, 3))
    fields.update(overrides)
    return ModelReply(**fields)


def run(backend=None, checker=None, **overrides):
    ledger = overrides.pop("ledger", TaskLedger())
    request = overrides.pop("request", TaskRequest("task", "math", "6 * 7", "2026-11-11", "exact-v1"))
    kwargs = dict(authorize=lambda q, r: True, max_usd="1", max_tokens=1000, ledger=ledger)
    kwargs.update(overrides)
    routes = kwargs.pop("routes", [route(), route("large")])
    result = run_task(request, routes, backend or (lambda q, r: reply(r)),
                      checker or (lambda q, a: a == "42"), **kwargs)
    return result, ledger


def test_disjoint_tokens_and_exact_tariff():
    use = Usage(10, 20, 30, 40, 35, Decimal("0.01"))
    assert use.tokens == 100
    assert tariff().cost(use) == Decimal("0.010272")


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_usage_rejects_bad_counts(value):
    with pytest.raises(ValueError):
        Usage(output_tokens=value)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1"])
def test_tariff_rejects_bad_amounts(value):
    with pytest.raises(ValueError):
        tariff(input_usd=value)


def test_reasoning_is_subset_not_extra_billing():
    with pytest.raises(ValueError):
        Usage(output_tokens=2, reasoning_tokens=3)
    assert tariff().cost(Usage(output_tokens=4, reasoning_tokens=3)) == Decimal("0.000020")


def test_escalation_accounts_rejected_and_accepted_attempts(tmp_path):
    result, ledger = run(backend=lambda q, r: reply(r, "wrong" if r.model == "small" else "42"))
    assert result.accepted and result.answer == "42"
    assert [row.outcome for row in ledger.attempts] == ["rejected", "accepted"]
    assert ledger.attempts[1].parent_attempt_id == ledger.attempts[0].attempt_id
    summary = ledger.summary()
    assert summary["accepted_tasks"] == 1
    assert summary["accounted_tokens"] == 36
    assert summary["known_cost_usd"] == "0.0000744"
    assert summary["cost_per_accepted_task_usd"] == summary["known_cost_usd"]
    path = ledger.write_jsonl(tmp_path / "attempts.jsonl")
    restored = TaskLedger.read_jsonl(path)
    assert restored.attempts == ledger.attempts
    assert restored.summary() == summary
    assert ledger.summary("absent")["cost_per_accepted_task_usd"] is None


def test_budget_prevents_dispatch_not_posthoc_discard():
    calls = []
    result, ledger = run(backend=lambda q, r: calls.append(r), max_usd="0.000001")
    assert result.stopped_reason == "budget"
    assert not calls and ledger.accounted_usd == 0


def test_token_budget_shared_across_tasks():
    _, ledger = run(routes=[route()], max_tokens=60)
    result, ledger = run(ledger=ledger, max_tokens=60,
                         request=TaskRequest("second", "math", "question", "2026-11-11", "v1"))
    assert result.stopped_reason == "budget"  # 18 used + 55 reserved exceeds 60.


def test_attempt_budget_is_shared_across_tasks():
    _, ledger = run(routes=[route()], max_attempts=1)
    calls = []
    result, ledger = run(ledger=ledger, backend=lambda q, r: calls.append(r),
                         max_attempts=1,
                         request=TaskRequest("second", "math", "question", "2026-11-11", "v1"))
    assert result.stopped_reason == "attempt_budget"
    assert not calls and len(ledger.attempts) == 1


def test_expired_tariff_rejected_before_backend():
    with pytest.raises(ValueError, match="not valid"):
        run(request=TaskRequest("task", "math", "x", "2026-12-01", "v1"))


def test_denied_authorization_does_not_dispatch():
    calls = []
    result, ledger = run(backend=lambda q, r: calls.append(r), authorize=lambda q, r: False)
    assert result.stopped_reason == "denied" and not calls
    assert ledger.accounted_tokens == 0


def test_non_boolean_authorization_is_denied():
    result, _ = run(authorize=lambda q, r: "approved")
    assert result.stopped_reason == "denied"


def test_authorization_error_is_denied():
    def fail(q, r):
        raise RuntimeError("policy unavailable")
    result, ledger = run(authorize=fail)
    assert result.stopped_reason == "denied"
    assert ledger.attempts[0].message == "authorization error: RuntimeError"


@pytest.mark.parametrize("changes, reason", [
    ({"refused": True}, "refused"),
    ({"complete": False}, "incomplete"),
    ({"model": "substitute"}, "contract_error"),
    ({"revision": "rev2"}, "contract_error"),
    ({"safeguard_mode": "privileged"}, "contract_error"),
    ({"usage": Usage(output_tokens=21)}, "contract_error"),
    ({"complete": "true"}, "contract_error"),
])
def test_no_escalation_after_safety_or_protocol_failure(changes, reason):
    calls = []
    def backend(q, r):
        calls.append(r)
        return reply(r, **changes)
    result, ledger = run(backend=backend)
    assert result.stopped_reason == reason and len(calls) == 1
    assert ledger.summary()["accepted_tasks"] == 0
    assert ledger.summary()["cost_per_accepted_task_usd"] is None


def test_substitute_has_unknown_price_and_retains_identity():
    _, ledger = run(backend=lambda q, r: reply(r, model="fallback"))
    assert ledger.attempts[0].actual_model == "fallback"
    assert ledger.summary()["unknown_cost_attempts"] == 1
    assert ledger.accounted_usd == tariff().cost(route().ceiling)


def test_unknown_failure_is_not_free():
    def backend(q, r):
        raise RuntimeError("transport failure")
    result, ledger = run(backend=backend)
    assert result.stopped_reason == "backend_error"
    assert ledger.summary()["unknown_cost_attempts"] == 1
    assert ledger.accounted_usd == tariff().cost(route().ceiling)
    assert ledger.summary()["cost_per_accepted_task_usd"] is None


def test_unresolved_usage_blocks_other_tasks():
    def backend(q, r):
        raise RuntimeError("unknown bill")
    _, ledger = run(backend=backend)
    with pytest.raises(ValueError, match="unresolved"):
        run(ledger=ledger,
            request=TaskRequest("second", "math", "question", "2026-11-11", "v1"))


def test_known_failure_keeps_usage():
    def backend(q, r):
        raise BackendFailure("timeout after generation", Usage(output_tokens=10))
    result, ledger = run(backend=backend)
    assert result.stopped_reason == "backend_error"
    assert ledger.accounted_usd == Decimal("0.00005")
    assert ledger.summary()["unknown_cost_attempts"] == 0


def test_checker_exception_keeps_generation_cost():
    def check(q, text):
        raise RuntimeError("verifier crashed")
    result, ledger = run(checker=check)
    assert result.stopped_reason == "checker_error"
    assert ledger.accounted_tokens == 18
    assert ledger.attempts[0].verifier_s >= 0


def test_checker_cannot_return_truthy_object():
    result, _ = run(checker=lambda q, text: "pass")
    assert result.stopped_reason == "checker_error"


def test_cancel_after_dispatch_accounts_usage_and_prevents_acceptance():
    stop = [False]
    def backend(q, r):
        stop[0] = True
        return reply(r)
    result, ledger = run(backend=backend, cancelled=lambda: stop[0])
    assert result.stopped_reason == "cancelled" and ledger.accounted_tokens == 18


def test_cancel_before_dispatch_costs_nothing():
    result, ledger = run(cancelled=lambda: True)
    assert result.stopped_reason == "cancelled" and ledger.accounted_tokens == 0


def test_attempt_limit_and_duplicate_task_guard():
    _, ledger = run(checker=lambda q, text: False, max_attempts=1)
    assert len(ledger.attempts) == 1
    with pytest.raises(ValueError, match="task already present"):
        run(ledger=ledger)
    with pytest.raises(ValueError, match="duplicate attempt"):
        ledger.append(ledger.attempts[0])


def test_ledger_rejects_future_schema(tmp_path):
    _, ledger = run()
    p = ledger.write_jsonl(tmp_path / "attempts.jsonl")
    row = json.loads(p.read_text())
    row["schema_version"] = 2
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="unsupported"):
        TaskLedger.read_jsonl(p)
