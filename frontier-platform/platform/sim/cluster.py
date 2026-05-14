"""Virtual GPU fleet with realistic failure model.

MTBF assumption: per-GPU MTBF ~ 10 years ≈ 87600 h. On 4096 GPUs we expect
~one failure every ~21 hours. We sample inter-failure times from Exp(λ).
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field

GPU_MTBF_HOURS = 87600.0    # ~10 years
NODE_RECOVERY_MIN = 8.0     # how long to swap in a hot spare + restart

GPU_SPECS = {
    # peak BF16 TFLOP/s, HBM GB, $/GPU-hr (rented)
    "H100":  {"tflops": 989.0, "hbm": 80,  "price": 2.0},
    "H200":  {"tflops": 989.0, "hbm": 141, "price": 2.5},
    "B200":  {"tflops": 2250.0, "hbm": 192, "price": 4.5},
    "A100":  {"tflops": 312.0, "hbm": 80,  "price": 1.2},
}


@dataclass
class Cluster:
    name: str
    n_nodes: int
    gpus_per_node: int = 8
    gpu_type: str = "H100"
    hot_spare_frac: float = 0.03
    failures: int = 0
    downtime_seconds: float = 0.0
    quarantined: int = 0
    healthy_nodes: int = field(init=False)

    def __post_init__(self):
        self.healthy_nodes = self.n_nodes

    @property
    def total_gpus(self) -> int:
        return self.n_nodes * self.gpus_per_node

    @property
    def healthy_gpus(self) -> int:
        return self.healthy_nodes * self.gpus_per_node

    @property
    def peak_tflops(self) -> float:
        return self.healthy_gpus * GPU_SPECS[self.gpu_type]["tflops"]

    @property
    def hourly_cost(self) -> float:
        return self.total_gpus * GPU_SPECS[self.gpu_type]["price"]

    def sample_failures(self, dt_hours: float, rng: random.Random) -> int:
        """Poisson-approx number of node failures in dt hours."""
        # node MTBF = GPU MTBF / gpus_per_node (any GPU dying takes the node)
        node_mtbf = GPU_MTBF_HOURS / self.gpus_per_node
        rate_per_node_per_hour = 1.0 / node_mtbf
        expected = rate_per_node_per_hour * self.healthy_nodes * dt_hours
        # Poisson sample
        L = math.exp(-expected); k = 0; p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p < L:
                return k - 1

    def tick(self, dt_seconds: float, rng: random.Random) -> dict:
        """Advance cluster state. Returns event summary."""
        dt_h = dt_seconds / 3600.0
        fails = self.sample_failures(dt_h, rng)
        recovered = 0
        if fails:
            self.failures += fails
            # each failure costs the whole job NODE_RECOVERY_MIN of stalled time
            self.downtime_seconds += fails * NODE_RECOVERY_MIN * 60
            spare = max(1, int(self.n_nodes * self.hot_spare_frac))
            # we replace from spares unless exhausted
            if fails <= spare:
                recovered = fails
            else:
                self.healthy_nodes -= (fails - spare)
                self.quarantined += (fails - spare)
                recovered = spare
        return {"failures": fails, "recovered": recovered}
