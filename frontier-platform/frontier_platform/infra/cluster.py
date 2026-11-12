"""Cluster description + topology-aware placement."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Node:
    hostname: str
    gpu_type: str
    gpu_count: int
    rack: str
    leaf_switch: str
    superpod: str
    healthy: bool = True


class Cluster:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def healthy_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.healthy]

    def allocate(self, n_gpus: int, prefer_same_superpod: bool = True) -> list[Node]:
        healthy = self.healthy_nodes()
        total = sum(n.gpu_count for n in healthy)
        if total < n_gpus:
            raise RuntimeError(f"insufficient GPUs: need {n_gpus}, have {total}")
        if prefer_same_superpod:
            by_pod: dict[str, list[Node]] = defaultdict(list)
            for n in healthy:
                by_pod[n.superpod].append(n)
            # Try each superpod (largest first) before falling back to cross-pod.
            pods = sorted(by_pod.items(), key=lambda kv: -sum(x.gpu_count for x in kv[1]))
            for _, pod_nodes in pods:
                picked: list[Node] = []
                got = 0
                for n in sorted(pod_nodes, key=lambda x: -x.gpu_count):
                    if got >= n_gpus:
                        break
                    picked.append(n)
                    got += n.gpu_count
                if got >= n_gpus:
                    return picked
        # Cross-pod fallback: greedy by gpu_count desc.
        picked = []
        got = 0
        for n in sorted(healthy, key=lambda x: -x.gpu_count):
            if got >= n_gpus:
                break
            picked.append(n)
            got += n.gpu_count
        if got < n_gpus:
            raise RuntimeError(f"insufficient GPUs: need {n_gpus}, have {got}")
        return picked

    def quarantine(self, hostname: str, reason: str) -> None:
        for n in self.nodes:
            if n.hostname == hostname:
                n.healthy = False
                return
