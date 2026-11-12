"""Auditable base-HumanEval evaluation through an explicit local Docker backend.

This is NOT EvalPlus/HumanEval+. A subset is labeled as a subset. Every generated
sample and every unknown infrastructure verdict is retained; unknowns suppress
the headline score. No model output is ever executed on the host. The separately
named eval.trusted_subprocess module exists only for reviewed legacy fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import logging
import math
from pathlib import Path
import re
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cf_provenance import digest, environment_versions, file_sha256, require_revision, write_json
from eval.python_jail import DockerPythonExecutor, PythonLimits, PythonResult

LOGGER = logging.getLogger(__name__)
BASE_DATASET = "openai/openai_humaneval"
BASE_IDS = frozenset(f"HumanEval/{index}" for index in range(164))


def extract_code(text: str) -> str:
    """Extract the first fenced Python block, or the first plausible code suffix."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    start = re.search(r"^(?:async\s+def|def|class|import|from)\s", text, re.MULTILINE)
    return text[start.start():] if start else text


def build_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    """Assemble text only; both candidate and tests execute in the container."""
    if not entry_point.isidentifier() or keyword.iskeyword(entry_point):
        raise ValueError("entry_point must be a Python identifier")
    code = extract_code(completion)
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code) is None:
        code = prompt + "\n" + code
    return code + "\n\n" + test + f"\n\ncheck({entry_point})\n"


def run_one(program: str, timeout: float = 5.0, *, image: str | None = None,
            executor: DockerPythonExecutor | None = None) -> PythonResult:
    """Safe public execution API. Missing explicit image/backend is an error."""
    if executor is None:
        executor = DockerPythonExecutor(image, limits=PythonLimits(wall_seconds=timeout))
    return executor.run(program)


def run_many(programs: list[str], timeout: float = 5.0, *, image: str | None = None,
             executor: DockerPythonExecutor | None = None) -> list[PythonResult]:
    """Sequential, independently isolated candidate execution; no host fallback."""
    if executor is None:
        executor = DockerPythonExecutor(image, limits=PythonLimits(wall_seconds=timeout))
    return executor.run_many(programs)


def _eos_ids(tok) -> list[int]:
    """Collect explicit EOS/turn-end IDs without treating unknown tokens as EOS."""
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        tid = tok.convert_tokens_to_ids(marker)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def validate_problems(rows: list[dict]) -> list[dict]:
    """Validate complete task identities before any expensive model loading."""
    if not rows:
        raise ValueError("evaluation requires at least one problem")
    seen: set[str] = set()
    for row in rows:
        for field in ("task_id", "prompt", "test", "entry_point"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"each problem needs a nonempty string {field}")
        if row["task_id"] in seen:
            raise ValueError(f"duplicate task_id: {row['task_id']}")
        seen.add(row["task_id"])
        if not row["entry_point"].isidentifier() or keyword.iskeyword(row["entry_point"]):
            raise ValueError("entry_point must be a Python identifier")
    return rows


def load_problem_set(*, problems_path: str | None = None, dataset_revision: str | None = None,
                     benchmark: str = "humaneval-base") -> tuple[list[dict], dict]:
    """Load a pinned base dataset or an operator-supplied local JSON/JSONL suite.

    Local suites are content-hashed, not magically authenticated as canonical
    HumanEval. EvalPlus has different tests/oracles and is explicitly rejected;
    importing the package or renaming this dataset would not be an integration.
    """
    if benchmark not in {"humaneval-base", "local-python"}:
        raise ValueError("this runner supports base HumanEval/local Python, not EvalPlus/HumanEval+")
    if problems_path:
        path = Path(problems_path)
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("local problems must be a list of task objects (or one per JSONL line)")
        source = {"kind": "local-file", "sha256": file_sha256(path), "canonical_verified": False}
    else:
        if benchmark != "humaneval-base":
            raise ValueError("local-python requires --problems")
        revision = require_revision(dataset_revision, label="HumanEval dataset")
        from datasets import load_dataset
        rows = list(load_dataset(BASE_DATASET, split="test", revision=revision))
        source = {"kind": "huggingface", "name": BASE_DATASET, "revision": revision,
                  "split": "test", "canonical_verified": True}
    rows = validate_problems(rows)
    manifest = {"benchmark": benchmark, "source": source, "suite_sha256": digest(rows),
                "available_task_ids": [row["task_id"] for row in rows],
                "expected_base_task_ids": sorted(BASE_IDS) if benchmark == "humaneval-base" else None}
    return rows, manifest


def _generation_config(n_samples: int, max_new_tokens: int, temperature: float, top_p: float,
                       seed: int | None, prompt_style: str) -> dict:
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite/nonnegative and top_p in (0, 1]")
    if n_samples > 1 and temperature == 0:
        raise ValueError("multiple samples require temperature > 0; never silently change sampling")
    if prompt_style not in {"chat", "raw"}:
        raise ValueError("prompt_style must be chat or raw")
    return {"n_samples": n_samples, "max_new_tokens": max_new_tokens, "temperature": temperature,
            "top_p": top_p, "seed": seed, "prompt_style": prompt_style,
            "do_sample": temperature > 0, "num_return_sequences": n_samples}


def summarize_records(records: list[dict], *, suite: dict, model: dict, generation: dict,
                      evaluator: dict, selected_ids: list[str]) -> dict:
    """No headline score for missing/unassessed samples; retain transparent bounds."""
    n_samples = generation["n_samples"]
    completed_ids = {row["task_id"] for row in records}
    samples = [sample for row in records for sample in row["samples"]]
    unknown = sum(sample["verdict"]["passed"] is None for sample in samples)
    missing = (len(selected_ids) - len(completed_ids)) * n_samples
    solved = sum(any(sample["verdict"]["passed"] is True for sample in row["samples"]) for row in records)
    uncertain_tasks = sum(not any(s["verdict"]["passed"] is True for s in row["samples"])
                          and any(s["verdict"]["passed"] is None for s in row["samples"]) for row in records)
    uncertain_tasks += len(selected_ids) - len(completed_ids)
    complete = not unknown and not missing and len(samples) == len(selected_ids) * n_samples
    n = len(selected_ids)
    metric = f"pass@{n_samples}"
    all_supplied = set(selected_ids) == set(suite["available_task_ids"])
    canonical_full = (suite["benchmark"] == "humaneval-base" and set(selected_ids) == BASE_IDS
                      and suite["source"].get("canonical_verified", False))
    scope = "full-base-suite" if canonical_full else ("all-supplied-tasks" if all_supplied else "subset")
    return {"schema": "coder-finetune.code-eval/v1", "benchmark": suite["benchmark"], "scope": scope,
            "is_evalplus": False, "suite": suite, "task_ids": selected_ids,
            "model": model, "generation": generation, "evaluator": evaluator,
            "versions": environment_versions(), "n": n, "passes": solved, "n_samples": n_samples,
            "metric": metric, metric: solved / n if n and complete else None, "complete": complete,
            "unknown_samples": unknown + missing,
            "score_bounds": {"lower": solved / n if n else None,
                             "upper": (solved + uncertain_tasks) / n if n else None},
            "estimator": "fraction of tasks with >=1 passing sample; k equals the number generated",
            "records": records}


def evaluate_problems(problems: list[dict], generate: Callable, executor: DockerPythonExecutor, *,
                      suite: dict, model: dict, generation: dict, on_record: Callable | None = None) -> dict:
    """Test every sample, preserving generation errors and executor unknowns."""
    validate_problems(problems)
    records: list[dict] = []
    n_samples = generation["n_samples"]
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be positive")
    selected_ids = [problem["task_id"] for problem in problems]
    for problem in problems:
        started = time.monotonic()
        generation_error = None
        try:
            generated = generate(problem)
            if not isinstance(generated, list) or len(generated) != n_samples:
                raise ValueError("generator returned the wrong sample count")
        except Exception as exc:
            generation_error = type(exc).__name__
            generated = [None] * n_samples
        generation_seconds = time.monotonic() - started
        samples = []
        for index, value in enumerate(generated):
            if isinstance(value, str):
                value = {"completion": value, "token_ids": None, "stop_reason": "unknown"}
            if value is None or not isinstance(value, dict) or not isinstance(value.get("completion"), str):
                samples.append({"sample_id": index, "completion": None, "token_ids": None,
                                "stop_reason": "generation_error", "error_type": generation_error or "InvalidSample",
                                "verdict": {"passed": None, "outcome": "generation_error"}})
                continue
            completion = value["completion"]
            program = build_program(problem["prompt"], completion, problem["test"], problem["entry_point"])
            try:
                verdict = executor.run(program).to_dict()
            except Exception as exc:
                verdict = {"passed": None, "outcome": "runner_error", "error_type": type(exc).__name__}
            samples.append({"sample_id": index, "completion": completion,
                            "token_ids": value.get("token_ids"), "stop_reason": value.get("stop_reason", "unknown"),
                            "extracted_code": extract_code(completion),
                            "program_sha256": hashlib.sha256(program.encode()).hexdigest(), "verdict": verdict})
        record = {"task_id": problem["task_id"], "prompt": problem["prompt"],
                  "entry_point": problem["entry_point"], "test_sha256": digest(problem["test"]),
                  "problem_sha256": digest(problem), "generation_seconds": generation_seconds, "samples": samples}
        records.append(record)
        if on_record:
            on_record(record)
    return summarize_records(records, suite=suite, model=model, generation=generation,
                             evaluator=executor.identity(), selected_ids=selected_ids)


def quick_eval(model_path: str, n_problems: int | None = None, n_samples: int = 1,
               max_new_tokens: int = 384, temperature: float = 0.2,
               seed: int | None = None, save_completions: str | None = None,
               json_out: str | None = None, *, python_image: str | None = None,
               executor: DockerPythonExecutor | None = None, problems_path: str | None = None,
               dataset_revision: str | None = None, model_revision: str | None = None,
               tokenizer_revision: str | None = None, benchmark: str = "humaneval-base",
               prompt_style: str = "chat", top_p: float = 0.95,
               local_files_only: bool = True) -> float | None:
    """Run all supplied tasks by default; report a score only if fully assessed.

    A 20-task quick run remains available through n_problems=20, explicitly
    labeled subset. Output artifacts are always written, not conditional on
    whether a sample passes. This API never substitutes a host executor.
    """
    generation = _generation_config(n_samples, max_new_tokens, temperature, top_p, seed, prompt_style)
    if n_problems is not None and (type(n_problems) is not int or n_problems < 1):
        raise ValueError("n_problems must be positive or omitted for the full supplied suite")
    if executor is None:
        executor = DockerPythonExecutor(python_image)
    executor.require_available()
    problems, suite = load_problem_set(problems_path=problems_path, dataset_revision=dataset_revision, benchmark=benchmark)
    if n_problems is not None:
        if n_problems > len(problems):
            raise ValueError("n_problems exceeds the supplied suite; do not silently clamp the request")
        problems = problems[:n_problems]

    import torch
    from infer.generate import _load
    if seed is not None:
        torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok, model = _load(model_path, device, dtype, revision=model_revision,
                       tokenizer_revision=tokenizer_revision, local_files_only=local_files_only)
    eos_ids = _eos_ids(tok)
    generation.update({"eos_token_ids": eos_ids, "pad_token_id": tok.pad_token_id,
                       "model_generation_config": model.generation_config.to_dict()})

    def generate(problem: dict) -> list[dict]:
        text = problem["prompt"]
        if prompt_style == "chat":
            messages = [{"role": "system", "content": "You are a Python coding assistant. Complete the function. Return only Python code."},
                        {"role": "user", "content": text}]
            # Missing/unsupported templates are errors, never a hidden raw fallback.
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=(prompt_style == "raw")).to(device)
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tok.pad_token_id, "eos_token_id": eos_ids or tok.eos_token_id,
                  "num_return_sequences": n_samples}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            output = model.generate(**ids, **kwargs)
        length = ids["input_ids"].shape[1]
        samples = []
        for row in output:
            tokens = row[length:].tolist()
            first_eos = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
            if first_eos is not None:
                tokens = tokens[:first_eos + 1]
            samples.append({"completion": tok.decode(tokens, skip_special_tokens=True), "token_ids": tokens,
                            "stop_reason": "eos" if first_eos is not None else "length"})
        return samples

    completions_path = Path(save_completions or "out/eval/samples.jsonl")
    summary_path = json_out or "out/eval/summary.json"
    completions_path.parent.mkdir(parents=True, exist_ok=True)
    with completions_path.open("w", encoding="utf-8") as stream:
        def save_record(record: dict) -> None:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        summary = evaluate_problems(problems, generate, executor, suite=suite,
                                    model=model._cf_provenance, generation=generation, on_record=save_record)
    write_json_summary(summary, summary_path)
    LOGGER.info("%s %s: %s=%s; unknown samples=%s; report=%s", benchmark, summary["scope"],
                summary["metric"], summary[summary["metric"]], summary["unknown_samples"], summary_path)
    return summary[summary["metric"]]


def build_eval_summary(model_path: str, n: int, passes: int, n_samples: int,
                       temperature: float, seed: int | None) -> dict:
    """Legacy aggregate helper; explicitly NOT a complete benchmark artifact."""
    if n < 0 or not 0 <= passes <= n or n_samples < 1:
        raise ValueError("invalid aggregate counts")
    label = f"pass@{n_samples}"
    return {"model": model_path, "benchmark": "humaneval-base", "scope": "unverified-aggregate",
            "n": n, "passes": passes, label: passes / n if n else None, "metric": label,
            "n_samples": n_samples, "temperature": temperature, "seed": seed,
            "complete_benchmark_artifact": False}


def write_json_summary(summary: dict, path: str) -> None:
    """Publish the complete machine-readable report atomically."""
    write_json(path, summary)


def write_completions_jsonl(records: list[dict], path: str) -> None:
    """Persist every task/sample/verdict, not only passing or executed samples."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--python-image", required=True, help="explicitly trusted local sha256 image ID; never pulled")
    parser.add_argument("--python-path", default="/usr/local/bin/python3")
    parser.add_argument("--docker-socket", default="/var/run/docker.sock")
    parser.add_argument("--problems", help="local JSON/JSONL problem set, content-hashed in the report")
    parser.add_argument("--dataset-revision", help="immutable HF commit SHA when loading canonical base HumanEval")
    parser.add_argument("--benchmark", choices=["humaneval-base", "local-python"], default="humaneval-base")
    parser.add_argument("--n-problems", type=int, default=None, help="omit for all supplied tasks; a subset is never labeled full")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-style", choices=["chat", "raw"], default="chat")
    parser.add_argument("--allow-model-download", action="store_true", help="opt in to loading pinned model/tokenizer from HF")
    parser.add_argument("--save-completions", default="out/eval/samples.jsonl")
    parser.add_argument("--json-out", default="out/eval/summary.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    executor = DockerPythonExecutor(args.python_image, python_path=args.python_path, docker_socket=args.docker_socket)
    quick_eval(args.model, args.n_problems, args.n_samples, args.max_new_tokens, args.temperature,
               seed=args.seed, save_completions=args.save_completions, json_out=args.json_out,
               executor=executor, problems_path=args.problems, dataset_revision=args.dataset_revision,
               model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
               benchmark=args.benchmark, prompt_style=args.prompt_style, top_p=args.top_p,
               local_files_only=not args.allow_model_download)


if __name__ == "__main__":
    main()
