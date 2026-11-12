"""Bounded offline specialization experiment, NOT an LFM3/LLM benchmark.

Question: does supervised parity specialization improve held-out parity while
preserving a positive-number classifier? Compare identical base weights with
full SFT and a rank-one LoRA readout. The tiny fixed character classifier makes
accounting executable on CPU; its scores do not predict language-model gains.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .run_humaneval import write_json_summary

LOG = logging.getLogger(__name__)
VOCAB = " positiveven:+-0123456789"
# Deduplicate the vocabulary, retaining a fixed padding index.
VOCAB = "".join(dict.fromkeys(VOCAB))
WIDTH = 16
FORMAT = "tiny-character-classifier/v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Budget:
    seeds: tuple[int, ...] = (0, 1, 2)
    base_steps: int = 60
    train_steps: int = 80
    batch_size: int = 16
    learning_rate: float = 0.02
    branch_seconds: float = 15.0

    def __post_init__(self) -> None:
        for name, upper in (("base_steps", 200), ("train_steps", 200), ("batch_size", 32)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in 1..{upper}")
        if not self.seeds or len(self.seeds) > 10 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("use 1..10 distinct seeds")
        if any(type(s) is not int or not 0 <= s <= 2**31 - 1 for s in self.seeds):
            raise ValueError("seeds must be nonnegative 31-bit integers")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be in (0, .1]")
        if not math.isfinite(self.branch_seconds) or not 0 < self.branch_seconds <= 30:
            raise ValueError("branch_seconds must be in (0, 30]")


@dataclass(frozen=True)
class Example:
    task_id: str
    prompt: str
    label: int


def dataset() -> dict[str, list[Example]]:
    """Fixed disjoint identities; labels encode arithmetic, never model output."""
    def parity(n: int) -> Example:
        return Example(f"even/{n}", f"even:{n:03d}", int(n % 2 == 0))

    def positive(n: int) -> Example:
        return Example(f"positive/{n}", f"positive:{n:+04d}", int(n > 0))

    return {
        "train": [parity(n) for n in range(80)],
        "heldout": [parity(n) for n in range(80, 100)],
        "base_train": [positive(n) for n in range(-40, 40)],
        "regression": [positive(n) for n in list(range(-50, -40)) + list(range(40, 50))],
    }


def validate_splits(splits: dict[str, list[Example]]) -> None:
    """Reject ID or canonical-prompt leakage, including within a split."""
    if set(splits) != {"train", "heldout", "base_train", "regression"}:
        raise ValueError("exactly four named splits are required")
    ids: set[str] = set()
    prompts: set[str] = set()
    for examples in splits.values():
        if not examples or len(examples) > 128:
            raise ValueError("split size must be 1..128")
        for ex in examples:
            key = " ".join(ex.prompt.split()).casefold()
            if ex.task_id in ids or key in prompts:
                raise ValueError("split leakage: duplicate ID or prompt")
            if type(ex.label) is not int or ex.label not in (0, 1):
                raise ValueError("binary labels required")
            encode(ex.prompt)
            ids.add(ex.task_id)
            prompts.add(key)


def encode(prompt: str) -> list[int]:
    if not 0 < len(prompt) <= WIDTH or any(c not in VOCAB for c in prompt):
        raise ValueError("prompt exceeds fixed local vocabulary/length budget")
    return [VOCAB.index(c) for c in prompt.ljust(WIDTH)]


class TinyClassifier(nn.Module):
    """Fixed, small neural classifier trained with supervised cross entropy."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), 8)
        self.hidden = nn.Linear(WIDTH * 8, 32)
        self.head = nn.Linear(32, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden(self.embedding(tokens).flatten(1)))
        return self.head(hidden)


class LowRankHead(nn.Module):
    """W frozen; W_eff = W + B A, rank=1 and alpha/r=1. Bias stays frozen."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(1, base.in_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(base.out_features, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.base(hidden) + (hidden @ self.a.T) @ self.b.T


def state_hash(model: TinyClassifier) -> str:
    state = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        state.update(key.encode())
        state.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return state.hexdigest()


def save_model(model: TinyClassifier, directory: Path) -> None:
    """Save a merged tiny checkpoint; never pickle a Python model object."""
    merged = copy.deepcopy(model)
    if isinstance(merged.head, LowRankHead):
        adapter = merged.head
        with torch.no_grad():
            adapter.base.weight.add_(adapter.b @ adapter.a)
        merged.head = adapter.base
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "model.pt"
    torch.save(merged.state_dict(), weights)
    write_json_summary({"format": FORMAT, "vocabulary": VOCAB, "width": WIDTH,
                        "weights_sha256": sha(weights.read_bytes())}, str(directory / "config.json"))


def load_model(directory: Path) -> TinyClassifier:
    """Only this tiny local format; no HF IDs, remote code or download fallback."""
    if not directory.is_dir():
        raise ValueError("model_dir must be an existing local tiny model directory")
    config_path, weights = directory / "config.json", directory / "model.pt"
    if config_path.stat().st_size > 4096 or weights.stat().st_size > 1024 * 1024:
        raise ValueError("local checkpoint exceeds size budget")
    config = json.loads(config_path.read_text())
    if (config.get("format"), config.get("vocabulary"), config.get("width")) != (FORMAT, VOCAB, WIDTH):
        raise ValueError("unsupported local model format")
    if sha(weights.read_bytes()) != config["weights_sha256"]:
        raise ValueError("local weights hash mismatch")
    model = TinyClassifier()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("local model contains nonfinite weights")
    return model


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("branch wall-clock budget exhausted")


def train(model: TinyClassifier, examples: list[Example], steps: int, budget: Budget,
          seed: int, deadline: float, ledger: dict) -> None:
    """One fixed schedule, no tuning, retries, or selection on held-out data."""
    tokens = torch.tensor([encode(ex.prompt) for ex in examples])
    labels = torch.tensor([ex.label for ex in examples])
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=budget.learning_rate)
    model.train()
    for step in range(steps):
        check_deadline(deadline)
        indices = torch.randint(len(examples), (budget.batch_size,), generator=generator)
        event = {"step": step, "status": "started", "examples": budget.batch_size}
        ledger["training"].append(event)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(tokens[indices]), labels[indices])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        event.update(status="completed", loss=float(loss.detach()))
        check_deadline(deadline)


def metrics(attempts: list[dict]) -> dict:
    """Failures count as incorrect; calibration reports explicit valid coverage."""
    valid = [a for a in attempts if a["status"] == "ok"]
    n = len(attempts)
    brier = statistics.mean((a["probability"] - a["label"]) ** 2 for a in valid) if valid else None
    ece = 0.0
    for i in range(10):
        group = [a for a in valid if min(9, int(a["confidence"] * 10)) == i]
        if group:
            ece += len(group) / len(valid) * abs(statistics.mean(a["confidence"] for a in group)
                                               - statistics.mean(a["correct"] for a in group))
    times = sorted(a["seconds"] for a in valid)
    return {"n": n, "accuracy": sum(a["correct"] for a in attempts) / n if n else None,
            "failed": n - len(valid), "calibration_n": len(valid), "brier": brier,
            "ece_10_bins": ece if valid else None,
            "latency_seconds": {"p50": statistics.median(times),
                                "p95": times[math.ceil(.95 * len(times)) - 1],
                                "stdev": statistics.stdev(times) if len(times) > 1 else 0} if times else None}


def evaluate(model: TinyClassifier | None, splits: dict[str, list[Example]], deadline: float,
             error: str | None = None) -> list[dict]:
    """Keep every planned prediction, including errors and skipped branches."""
    attempts = []
    if model is not None:
        model.eval()
    for split in ("heldout", "regression"):
        for ex in splits[split]:
            started = time.monotonic()
            attempt = {**asdict(ex), "split": split, "status": "failed", "correct": False,
                       "probability": None, "confidence": None, "prediction": None}
            try:
                if error:
                    raise RuntimeError(error)
                check_deadline(deadline)
                if model is None:
                    raise RuntimeError("no trained model")
                with torch.no_grad():
                    probs = model(torch.tensor([encode(ex.prompt)])).softmax(-1)[0]
                check_deadline(deadline)
                if not torch.isfinite(probs).all():
                    raise ValueError("nonfinite prediction")
                probability = float(probs[1])
                prediction = int(probability >= .5)
                attempt.update(status="ok", prediction=prediction, probability=probability,
                               confidence=max(probability, 1 - probability), correct=prediction == ex.label)
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"[:512]
            attempt["seconds"] = time.monotonic() - started
            attempts.append(attempt)
    return attempts


def run_comparison(output: Path, budget: Budget, *, model_dir: Path | None = None,
                   splits: dict[str, list[Example]] | None = None) -> dict:
    """Run all predeclared seeds and arms on CPU, writing every branch ledger."""
    splits = dataset() if splits is None else splits
    validate_splits(splits)
    # Validate local inputs before creating artifacts. This format is deliberately
    # not a general HF loader: arbitrary large models are outside this budget.
    local = load_model(model_dir) if model_dir is not None else None
    output.mkdir(parents=True, exist_ok=False)
    manifest = {name: [asdict(ex) for ex in examples] for name, examples in splits.items()}
    write_json_summary(manifest, str(output / "dataset.json"))
    torch.set_num_threads(1)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=2, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = "unknown"
    summary = {
        "schema": "educational-specialization/v1", "claim": "tiny CPU classifier, not an LLM benchmark",
        "question": "Does parity specialization improve held-out parity and preserve sign classification?",
        "config": asdict(budget), "dataset_sha256": sha(json.dumps(manifest, sort_keys=True).encode()),
        "source_sha256": sha(Path(__file__).read_bytes()), "commit": commit,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "cpu": platform.processor(), "threads": 1},
        "local_model": str(model_dir.resolve()) if model_dir else None,
        "cost_units": "CPU wall seconds, optimizer steps, padded character tokens; no monetary estimate",
        "branches": [], "base_preparation": [], "aggregate": {},
    }
    with (output / "branches.jsonl").open("w") as stream:
        for seed in budget.seeds:
            torch.manual_seed(seed)
            base = copy.deepcopy(local) if local is not None else TinyClassifier()
            preparation = {"seed": seed, "training": [], "status": "ok"}
            base_error = None
            started = time.monotonic()
            try:
                if local is None:
                    train(base, splits["base_train"], budget.base_steps, budget, seed,
                          started + budget.branch_seconds, preparation)
            except Exception as exc:
                base_error = f"base preparation failed: {type(exc).__name__}: {exc}"
                preparation.update(status="failed", error=base_error)
            preparation["seconds"] = time.monotonic() - started
            summary["base_preparation"].append(preparation)
            for arm in ("baseline", "sft", "lora"):
                started = time.monotonic()
                deadline = started + budget.branch_seconds
                model = copy.deepcopy(base)
                branch = {"seed": seed, "arm": arm, "training": [], "status": "ok",
                          "initial_sha256": state_hash(base)}
                error = base_error
                try:
                    if error:
                        raise RuntimeError(error)
                    torch.manual_seed(seed)
                    if arm == "lora":
                        model.requires_grad_(False)
                        model.head = LowRankHead(model.head)
                    branch["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    if arm != "baseline":
                        train(model, splits["train"], budget.train_steps, budget, seed, deadline, branch)
                    branch["final_sha256"] = state_hash(model)
                    save_model(model, output / "models" / f"seed-{seed}" / arm)
                    # One unscored warm-up, independent of all held-out labels.
                    model.eval()
                    with torch.no_grad():
                        model(torch.tensor([encode(splits["train"][0].prompt)]))
                    check_deadline(deadline)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:512]
                    branch.update(status="failed", error=error)
                branch["attempts"] = evaluate(model, splits, deadline, error)
                if any(a["status"] != "ok" for a in branch["attempts"]):
                    branch["status"] = "failed"
                branch["metrics"] = {split: metrics([a for a in branch["attempts"] if a["split"] == split])
                                     for split in ("heldout", "regression")}
                branch["cost"] = {"seconds": time.monotonic() - started,
                                  "completed_steps": sum(e["status"] == "completed" for e in branch["training"]),
                                  "attempted_steps": len(branch["training"]),
                                  "training_character_tokens": len(branch["training"]) * budget.batch_size * WIDTH,
                                  "planned_predictions": len(branch["attempts"]),
                                  "api_calls": 0}
                stream.write(json.dumps(branch, allow_nan=False) + "\n")
                stream.flush()
                summary["branches"].append(branch)
                write_json_summary(summary, str(output / "run.json"))
    for arm in ("baseline", "sft", "lora"):
        branches = [b for b in summary["branches"] if b["arm"] == arm]
        summary["aggregate"][arm] = {}
        for split in ("heldout", "regression"):
            scores = [b["metrics"][split]["accuracy"] for b in branches]
            summary["aggregate"][arm][split] = {
                "accuracy_mean": statistics.mean(scores),
                "accuracy_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                "seeds": len(scores), "failed_branches": sum(b["status"] != "ok" for b in branches),
            }
    write_json_summary(summary, str(output / "run.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/specialization_smoke.json"))
    parser.add_argument("--output", type=Path, required=True, help="new directory, never overwritten")
    parser.add_argument("--model-dir", type=Path, help="existing tiny local checkpoint, not a Hub model ID")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if "seeds" in config:
        config["seeds"] = tuple(config["seeds"])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_comparison(args.output, Budget(**config), model_dir=args.model_dir)
    LOG.info("%s", json.dumps(result["aggregate"], indent=2))
    raise SystemExit(1 if any(b["status"] != "ok" for b in result["branches"]) else 0)


if __name__ == "__main__":
    main()
