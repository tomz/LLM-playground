"""Smoke tests for scripts/plot_sim.py — pure-Python, no matplotlib needed."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _fake_events(path: Path, n: int = 50) -> None:
    """Write a minimal events.jsonl that plot_sim.py can consume."""
    lines = [
        {"kind": "program.start", "name": "fake", "n_params": 1e9,
         "total_tokens": 1e9, "pretrain_gpus": 8, "pretrain_gpu_type": "H100"},
        {"kind": "pretrain.start", "n_params": 1e9, "total_tokens": 1e9,
         "total_steps": n, "seconds_per_step": 1.0, "modeled_seconds_per_step": 1.0,
         "throughput_source": "fake", "cluster_gpus": 8, "mfu": 0.5,
         "estimated_hours": 1.0},
    ]
    for i in range(n):
        lines.append({
            "kind": "pretrain.log", "step": i * 10,
            "loss": 4.0 - 2.0 * i / n,
            "day": i * 0.01, "failures": i // 20, "healthy_gpus": 8,
            "dollars": 10.0,
        })
    with open(path, "w") as f:
        for ev in lines:
            f.write(json.dumps(ev) + "\n")


@pytest.fixture
def fake_run(tmp_path):
    _fake_events(tmp_path / "events.jsonl")
    return tmp_path


def test_plot_sim_writes_svg_without_matplotlib(fake_run):
    """SVG path is hand-rolled — must succeed even if matplotlib is missing."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "plot_sim.py"),
         str(fake_run), "--no-png", "--quiet"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    svg = fake_run / "loss.svg"
    assert svg.exists() and svg.stat().st_size > 200
    content = svg.read_text()
    assert "<svg" in content and "</svg>" in content
    assert "polyline" in content   # at least one drawn series


def test_plot_sim_ascii_sparkline_in_stdout(fake_run):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "plot_sim.py"),
         str(fake_run), "--no-png"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert "LOSS CURVE" in result.stdout
    assert "█" in result.stdout    # ascii sparkline rendered something


def test_plot_sim_errors_on_missing_events(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "plot_sim.py"),
         str(tmp_path), "--no-png", "--quiet"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr
