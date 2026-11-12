"""Offline routing/accounting demonstration using synthetic, non-model replies.

Rates and responses are fixtures, not a vendor performance or pricing benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontier_platform.serving.task_cost import Tariff, TaskLedger, Usage
from frontier_platform.serving.task_routing import ModelReply, Route, TaskRequest, run_task


def run_demo() -> tuple[dict, dict[str, TaskLedger]]:
    """Compare a fixed premium route against cheap-then-premium routing."""
    routes = [Route(
        name, "fixture-v1", "low", "public",
        Tariff(name + "-fixture", name, "synthetic tariff, not vendor pricing",
               "2026-11-01", price, price), Usage(10, output_tokens=10),
    ) for name, price in [("small", "1"), ("large", "5")]]
    tasks = [("easy-1", "2", False), ("easy-2", "4", False), ("hard-1", "6", True)]
    reports = {}
    ledgers = {}
    for policy in ("fixed_large", "cascade"):
        ledger = TaskLedger()
        for task_id, expected, hard in tasks:
            request = TaskRequest(task_id, "fixture-arithmetic", expected, "2026-11-11", "exact-fixture-v1")
            def backend(req: TaskRequest, route: Route) -> ModelReply:
                answer = "wrong" if hard and route.model == "small" else req.prompt
                return ModelReply(answer, route.model, route.revision, route.safeguard_mode,
                                  Usage(10, output_tokens=10))
            run_task(request, routes if policy == "cascade" else routes[1:], backend,
                     lambda req, text: text == req.prompt,
                     authorize=lambda req, route: True, ledger=ledger, max_usd="1", max_tokens=200,
                     max_attempts=6)
        reports[policy] = ledger.summary()
        ledgers[policy] = ledger
    return {"kind": "synthetic_contract_demo", "policies": reports}, ledgers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("out/task-routing"))
    args = parser.parse_args()
    report, ledgers = run_demo()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ledger in ledgers.items():
        ledger.write_jsonl(args.out_dir / f"{name}.jsonl")
    path = args.out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Synthetic routing fixture (not model results): %s", path)


if __name__ == "__main__":
    main()
