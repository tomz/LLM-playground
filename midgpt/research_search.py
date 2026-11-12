"""Bounded local candidate search with host-owned verification and explicit cost.

The legacy callback benchmark remains available. The CLI uses the local model
adapter in research_model, never executes candidate code, and needs no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from research_model import Generation

Generator = Callable[[str, int], Iterable[str]]
Verifier = Callable[[str], tuple[bool, float]]
VERSION = "numeric-search-v1"
CHECKER_VERSION = "strict-json-sum-v1"
MAX_OUTPUT_CHARS = 16384


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    valid: bool
    generation_tokens: int
    verifier_s: float
    generation_s: float = 0.0
    status: str = "checked"
    error: str | None = None
    retry: int = 0
    budget_tokens: int = 0
    prompt_tokens: int = 0
    forward_input_tokens: int = 0
    token_ids: tuple[int, ...] = ()
    token_count_kind: str = "character-estimate"


@dataclass(frozen=True)
class SearchReport:
    attempts: list[CandidateResult]
    metadata: dict = field(default_factory=dict)
    stop_reason: str = "candidate_budget"
    elapsed_s: float = 0.0
    setup_s: float = 0.0

    @property
    def valid_rate(self) -> float:
        return sum(row.valid for row in self.attempts) / max(1, len(self.attempts))

    @property
    def generation_tokens(self) -> int:
        return sum(row.generation_tokens for row in self.attempts)

    @property
    def verifier_s(self) -> float:
        return sum(row.verifier_s for row in self.attempts)

    @property
    def first_valid(self) -> CandidateResult | None:
        return next((row for row in self.attempts if row.valid), None)

    @property
    def accepted_tasks(self) -> int:
        """One report represents ONE task, not one task per valid candidate."""
        return int(self.first_valid is not None)

    def to_dict(self) -> dict:
        """JSON-safe totals; undefined cost per accepted task is null, not zero."""
        result = asdict(self)
        total_s = self.setup_s + self.elapsed_s
        result["summary"] = {
            "attempts": len(self.attempts),
            "retries": max(0, len(self.attempts) - 1),
            "valid_candidates": sum(row.valid for row in self.attempts),
            "valid_rate": self.valid_rate,
            "accepted_tasks": self.accepted_tasks,
            "generation_tokens": self.generation_tokens,
            "budget_tokens": sum(row.budget_tokens for row in self.attempts),
            "forward_input_tokens": sum(row.forward_input_tokens for row in self.attempts),
            "generation_s": sum(row.generation_s for row in self.attempts),
            "verifier_s": self.verifier_s,
            "total_s": total_s,
            "tokens_per_accepted_task": (
                self.generation_tokens if self.accepted_tasks
                and all(row.token_count_kind == "exact" for row in self.attempts) else None
            ),
            "seconds_per_accepted_task": total_s if self.accepted_tasks else None,
            "token_accounting_complete": all(row.token_count_kind == "exact" for row in self.attempts),
        }
        return result


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:1000]


def _check(candidate: str, verifier: Verifier) -> tuple[bool, float, str | None]:
    start = time.perf_counter()
    try:
        valid, measured_s = verifier(candidate)
        if type(valid) is not bool or not math.isfinite(measured_s) or measured_s < 0:
            raise ValueError("checker must return (bool, finite nonnegative seconds)")
        return valid, max(measured_s, time.perf_counter() - start), None
    except Exception as exc:
        return False, time.perf_counter() - start, _error(exc)


def benchmark_search(prompt: str, generator: Generator, verifier: Verifier,
                     attempts: int) -> SearchReport:
    """Legacy callback API; counts are estimates, callbacks are trusted code.

    No extra item is pulled after the candidate limit. An exception while
    creating/advancing the iterable is recorded and terminates that stream.
    This API cannot impose a token or time limit inside arbitrary callbacks.
    """
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    rows: list[CandidateResult] = []
    start = time.perf_counter()
    iterator = None
    stop = "candidate_budget"
    for index in range(attempts):
        generated_at = time.perf_counter()
        try:
            if iterator is None:
                iterator = iter(generator(prompt, attempts))
            candidate = next(iterator)
            if not isinstance(candidate, str):
                raise TypeError("generator must yield strings")
        except StopIteration:
            stop = "generator_exhausted"
            break
        except Exception as exc:
            rows.append(CandidateResult("", False, 0, 0.0,
                                        time.perf_counter() - generated_at,
                                        "generation_error", _error(exc), index,
                                        token_count_kind="unknown"))
            stop = "generation_error"
            break
        generation_s = time.perf_counter() - generated_at
        valid, verifier_s, error = _check(candidate, verifier)
        tokens = max(1, (len(candidate) + 3) // 4)
        rows.append(CandidateResult(candidate, valid, tokens, verifier_s,
                                    generation_s, "checker_error" if error else "checked",
                                    error, index, tokens))
    return SearchReport(rows, {"adapter": "legacy-callback", "version": VERSION},
                        stop, time.perf_counter() - start)


@dataclass(frozen=True)
class SearchBudget:
    candidates: int = 3
    max_new_tokens: int = 32
    total_tokens: int = 96

    def __post_init__(self) -> None:
        for name, upper in (("candidates", 128), ("max_new_tokens", 512), ("total_tokens", 8192)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1, {upper}]")


@dataclass(frozen=True)
class NumericTask:
    left: int = 19
    right: int = 23

    def __post_init__(self) -> None:
        if any(type(n) is not int or abs(n) > 1_000_000 for n in (self.left, self.right)):
            raise ValueError("operands must be integers with absolute value <= 1000000")

    @property
    def prompt(self) -> str:
        return (f'Add {self.left} and {self.right}. Return only a JSON object with one '
                'key "sum" and an integer value. No explanation.\n')

    def verify(self, candidate: str) -> tuple[bool, float]:
        """Parse data only: reject extra keys, duplicate keys, floats and bools."""
        if len(candidate) > MAX_OUTPUT_CHARS:
            return False, 0.0

        def unique(pairs: list[tuple[str, object]]) -> dict:
            obj: dict = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError("duplicate key")
                obj[key] = value
            return obj

        try:
            obj = json.loads(candidate, object_pairs_hook=unique)
            valid = (type(obj) is dict and set(obj) == {"sum"}
                     and type(obj["sum"]) is int and obj["sum"] == self.left + self.right)
            return valid, 0.0
        except (ValueError, RecursionError):
            return False, 0.0


def run_search(task: NumericTask, generate: Callable[[str, int, int], Generation],
               budget: SearchBudget, *, model_info: dict, seed: int = 0,
               verifier: Verifier | None = None, checker_version: str = CHECKER_VERSION,
               setup_s: float = 0.0) -> SearchReport:
    """Run independent retries; all branches count, even after acceptance.

    The adapter receives (prompt, token allowance, per-attempt seed). Exceptions
    reserve the full allowance since unreported generation may have happened.
    Only the supplied local adapter enforces its computation budget internally;
    arbitrary Python callbacks are trusted, not sandboxed or preempted.
    """
    if verifier is not None and checker_version == CHECKER_VERSION:
        raise ValueError("custom verifier requires an explicit checker_version")
    if not math.isfinite(setup_s) or setup_s < 0:
        raise ValueError("setup_s must be finite and nonnegative")
    checker = task.verify if verifier is None else verifier
    metadata = {
        "version": VERSION, "domain": "integer-addition/strict-json",
        "task": asdict(task), "prompt": task.prompt,
        "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
        "model": model_info, "checker_version": checker_version,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "budget": asdict(budget), "seed": seed,
        "effort": "independent-candidate-retries; no critique or hidden reasoning budget",
        "stop_on_success": False,
    }
    metadata["experiment_id"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()
    rows: list[CandidateResult] = []
    remaining = budget.total_tokens
    start = time.perf_counter()
    stop = "candidate_budget"
    for retry in range(budget.candidates):
        if remaining <= 0:
            stop = "token_budget"
            break
        allowance = min(budget.max_new_tokens, remaining)
        generated_at = time.perf_counter()
        result = Generation("")
        unknown = False
        try:
            result = generate(task.prompt, allowance, seed + retry)
            if not isinstance(result, Generation):
                raise TypeError("adapter must return Generation")
            if (not isinstance(result.text, str) or len(result.text) > MAX_OUTPUT_CHARS
                    or len(result.token_ids) > allowance
                    or any(type(t) is not int or t < 0 for t in result.token_ids)
                    or type(result.prompt_tokens) is not int or result.prompt_tokens < 0
                    or type(result.forward_input_tokens) is not int or result.forward_input_tokens < 0
                    or (result.error is not None and not isinstance(result.error, str))):
                raise ValueError("adapter violated output/accounting contract")
        except Exception as exc:
            result = Generation("", error=_error(exc))
            unknown = True
        generation_s = time.perf_counter() - generated_at
        charge = allowance if result.error is not None else len(result.token_ids)
        remaining -= charge
        valid, verifier_s, error = False, 0.0, result.error
        status = "generation_error"
        if error is None:
            valid, verifier_s, error = _check(result.text, checker)
            status = "checker_error" if error else ("accepted" if valid else "invalid")
        rows.append(CandidateResult(
            result.text, valid, len(result.token_ids), verifier_s, generation_s,
            status, error, retry, charge, result.prompt_tokens,
            result.forward_input_tokens, result.token_ids,
            "unknown" if unknown else "exact",
        ))
    if remaining <= 0:
        stop = "token_budget"
    return SearchReport(rows, metadata, stop, time.perf_counter() - start, setup_s)


def main(argv: list[str] | None = None) -> int:
    """Run one bounded task and emit its complete JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", "--ckpt", type=Path)
    source.add_argument("--tiny-smoke", action="store_true", help="Untrained CPU byte GPT; not a quality eval")
    parser.add_argument("--gpt2-vocab", type=Path, help="Local GPT-2 vocab.json; no downloads")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--total-tokens", type=int, default=96)
    parser.add_argument("--left", type=int, default=19)
    parser.add_argument("--right", type=int, default=23)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Optional JSON artifact; stdout otherwise")
    args = parser.parse_args(argv)
    try:
        budget = SearchBudget(args.candidates, args.max_new_tokens, args.total_tokens)
        task = NumericTask(args.left, args.right)
        if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 10:
            raise ValueError("temperature must be finite and in [0, 10]")
        if not 0 <= args.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        from research_model import LocalModelAdapter
        import torch
        torch.set_num_threads(1)

        setup_at = time.perf_counter()
        adapter = (LocalModelAdapter.tiny(args.seed, args.temperature) if args.tiny_smoke
                   else LocalModelAdapter.from_checkpoint(args.checkpoint, args.gpt2_vocab, args.temperature))
        report = run_search(task, adapter.generate, budget, model_info=adapter.provenance,
                            seed=args.seed, setup_s=time.perf_counter() - setup_at)
        payload = json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        else:
            # stdout is machine-readable experiment data, not a log message.
            print(payload, end="")
        return 0
    except Exception as exc:
        logging.error("Experiment setup/output failed: %s", _error(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
