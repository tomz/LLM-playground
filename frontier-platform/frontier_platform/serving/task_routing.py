"""Fail-closed, cost-reserved routing reference with injectable local adapters.

The adapter must enforce usage ceilings and its own timeouts. This synchronous
control-plane example cannot interrupt a hung call or enforce provider billing.
Refusals, substitutions, partial streams, and unknown-cost failures stop routing;
only a completed, in-contract answer rejected by the checker may escalate.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter
from typing import Callable

from .task_cost import Tariff, TaskAttempt, TaskLedger, Usage, amount, count


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    domain: str
    prompt: str
    observed_on: str
    grader_revision: str

    def __post_init__(self) -> None:
        from datetime import date
        if not all((self.task_id, self.domain, self.grader_revision)):
            raise ValueError("request identities are required")
        date.fromisoformat(self.observed_on)


@dataclass(frozen=True)
class Route:
    model: str
    revision: str
    effort: str
    safeguard_mode: str
    tariff: Tariff
    ceiling: Usage

    def __post_init__(self) -> None:
        if not all((self.model, self.revision, self.effort, self.safeguard_mode)):
            raise ValueError("route identities are required")
        if self.model != self.tariff.model:
            raise ValueError("route and tariff models differ")


@dataclass(frozen=True)
class ModelReply:
    text: str
    model: str
    revision: str
    safeguard_mode: str
    usage: Usage
    complete: bool = True
    refused: bool = False


class BackendFailure(RuntimeError):
    """Adapter failure with optional measured usage from the requested model."""

    def __init__(self, message: str, usage: Usage | None = None):
        super().__init__(message)
        if usage is not None and not isinstance(usage, Usage):
            raise TypeError("failure usage must be Usage or None")
        self.usage = usage


@dataclass(frozen=True)
class RoutingResult:
    accepted: bool
    answer: str | None
    stopped_reason: str
    attempt_ids: tuple[str, ...]


Backend = Callable[[TaskRequest, Route], ModelReply]
Checker = Callable[[TaskRequest, str], bool]
Authorizer = Callable[[TaskRequest, Route], bool]


def run_task(
    request: TaskRequest,
    routes: list[Route],
    backend: Backend,
    checker: Checker,
    *,
    authorize: Authorizer,
    ledger: TaskLedger,
    max_usd: Decimal | str,
    max_tokens: int,
    max_attempts: int = 2,
    cancelled: Callable[[], bool] = lambda: False,
) -> RoutingResult:
    """Try caller-ordered model/effort routes within a shared ledger budget.

    Limits are cumulative across this ledger, including other tasks and branches.
    Caller ordering is the routing policy; this reference does not learn a router.
    Authorization is checked separately before every dispatch. Adapters get the
    per-bucket ceiling and must report complete usage, including reasoning.
    """
    budget = amount(max_usd)
    count(max_tokens)
    count(max_attempts)
    if max_tokens == 0 or max_attempts == 0:
        raise ValueError("positive token and attempt budgets are required")
    if any(row.task_id == request.task_id for row in ledger.attempts):
        raise ValueError("task already present; use a unique id for each trial")
    if any(row.known_cost_usd is None for row in ledger.attempts):
        raise ValueError("ledger has unresolved usage or tariffs; reconcile before dispatch")
    # Validate all prices before any side effects; never silently use an expired tariff.
    for route in routes:
        route.tariff.check_date(request.observed_on)
    ids: list[str] = []
    reason = "no_routes"
    remaining_attempts = max_attempts - len(ledger.attempts)
    if remaining_attempts <= 0:
        return RoutingResult(False, None, "attempt_budget", ())
    for route in routes[:remaining_attempts]:
        started = perf_counter()
        usage: Usage | None = Usage()
        actual_model = actual_revision = actual_safeguards = None
        message = ""
        verifier_s = 0.0
        reply: ModelReply | None = None
        reason = "rejected"
        if cancelled():
            reason = "cancelled"
        elif (ledger.accounted_usd + route.tariff.cost(route.ceiling) > budget
              or ledger.accounted_tokens + route.ceiling.tokens > max_tokens):
            reason = "budget"
        else:
            try:
                allowed = authorize(request, route) is True
            except Exception as exc:
                allowed = False
                message = f"authorization error: {type(exc).__name__}"
            if not allowed:
                reason = "denied"
            elif cancelled():
                reason = "cancelled"
            else:
                # After dispatch we cannot assume a failed call was free.
                usage = None
                try:
                    reply = backend(request, route)
                    if not isinstance(reply, ModelReply) or not isinstance(reply.usage, Usage):
                        raise TypeError("adapter returned an invalid reply")
                    usage = reply.usage
                    actual_model, actual_revision = reply.model, reply.revision
                    actual_safeguards = reply.safeguard_mode
                    if (reply.model != route.model or reply.revision != route.revision
                            or reply.safeguard_mode != route.safeguard_mode
                            or not usage.fits(route.ceiling)
                            or not isinstance(reply.text, str)
                            or type(reply.complete) is not bool or type(reply.refused) is not bool):
                        reason = "contract_error"
                    elif reply.refused:
                        reason = "refused"
                    elif not reply.complete:
                        reason = "incomplete"
                    elif cancelled():
                        reason = "cancelled"
                    else:
                        verify_started = perf_counter()
                        try:
                            verdict = checker(request, reply.text)
                            if type(verdict) is not bool:
                                raise TypeError("checker must return bool")
                            reason = "accepted" if verdict else "rejected"
                        except Exception as exc:
                            reason = "checker_error"
                            message = type(exc).__name__
                        finally:
                            verifier_s = perf_counter() - verify_started
                        if cancelled():
                            reason = "cancelled"
                except BackendFailure as exc:
                    usage = exc.usage
                    reason = "backend_error"
                    message = str(exc)
                except Exception as exc:
                    reason = "backend_error"
                    message = type(exc).__name__
        attempt_id = f"{request.task_id}:{len(ids)}"
        ledger.append(TaskAttempt(
            attempt_id=attempt_id, task_id=request.task_id, domain=request.domain,
            requested_model=route.model, model_revision=route.revision,
            actual_model=actual_model, actual_revision=actual_revision,
            effort=route.effort, safeguard_mode=route.safeguard_mode,
            actual_safeguard_mode=actual_safeguards, grader_revision=request.grader_revision,
            tariff=route.tariff, outcome=reason, usage=usage, reserved_usage=route.ceiling,
            wall_s=perf_counter() - started, verifier_s=verifier_s,
            parent_attempt_id=ids[-1] if ids else None, message=message,
        ))
        ids.append(attempt_id)
        if reason == "accepted":
            return RoutingResult(True, reply.text, reason, tuple(ids))
        if reason != "rejected":
            break
    return RoutingResult(False, None, reason, tuple(ids))
