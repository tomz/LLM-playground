"""Cluster description + topology-aware placement."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Node:
    hostname: str
    gpu_type: str         # 'H100' | 'H200' | 'B200'
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
        raise NotImplementedError
    def quarantine(self, hostname: str, reason: str) -> None:
        for n in self.nodes:
            if n.hostname == hostname:
                n.healthy = False
                return
