import json
import time
from platform.infra.cluster import Cluster, Node
from platform.infra.scheduler import LocalScheduler, JobSpec
from platform.infra.observability import Metrics
import pytest


def _mk_nodes():
    return [
        Node("h1", "H100", 8, "r1", "sw1", "podA"),
        Node("h2", "H100", 8, "r1", "sw1", "podA"),
        Node("h3", "H100", 8, "r2", "sw2", "podB"),
    ]


def test_cluster_allocate_picks_same_superpod_first():
    c = Cluster(_mk_nodes())
    nodes = c.allocate(16, prefer_same_superpod=True)
    pods = {n.superpod for n in nodes}
    assert pods == {"podA"}
    assert sum(n.gpu_count for n in nodes) >= 16


def test_cluster_allocate_raises_when_insufficient():
    c = Cluster(_mk_nodes())
    with pytest.raises(RuntimeError):
        c.allocate(1000)


def test_cluster_quarantine():
    c = Cluster(_mk_nodes())
    c.quarantine("h1", "bad")
    assert len(c.healthy_nodes()) == 2


def test_local_scheduler_submit_status_cancel():
    s = LocalScheduler()
    spec = JobSpec(name="sleep", image="", command=["sleep", "5"], nodes=1, gpus_per_node=0)
    jid = s.submit(spec)
    assert s.status(jid) == "RUNNING"
    s.cancel(jid)
    time.sleep(0.2)
    assert s.status(jid) in ("CANCELLED", "FAILED", "COMPLETED")


def test_metrics_emit_writes_jsonl_when_env_set(tmp_path, monkeypatch):
    p = tmp_path / "m.jsonl"
    monkeypatch.setenv("PLATFORM_METRICS_PATH", str(p))
    Metrics.emit("loss", 1.5, {"step": "10"})
    Metrics.emit("loss", 1.2)
    Metrics.flush()
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["name"] == "loss" and rec["value"] == 1.5


def test_metrics_emit_noop_without_env(monkeypatch):
    monkeypatch.delenv("PLATFORM_METRICS_PATH", raising=False)
    Metrics.emit("x", 1.0)  # must not raise
