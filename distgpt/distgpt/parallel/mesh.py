"""Explicit (pipeline, data, tensor) coordinates and group ownership."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


def resolve_dimensions(world_size: int, dp: int, tp: int, pp: int) -> tuple[int, int, int]:
    """Resolve dp=-1 and reject invalid layouts without removable assertions."""
    for name, value in (("world_size", world_size), ("tp", tp), ("pp", pp)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if dp == -1:
        if world_size % (tp * pp):
            raise ValueError("world size must be divisible by tp*pp")
        dp = world_size // (tp * pp)
    if type(dp) is not int or dp < 1 or dp * tp * pp != world_size:
        raise ValueError(f"dp*tp*pp ({dp}*{tp}*{pp}) != world_size ({world_size})")
    return dp, tp, pp


def build_mesh(world_size: int, dp: int, tp: int, pp: int, device_type: str = "cuda"):
    """Construct the named mesh; single-process development needs no group."""
    dp, tp, pp = resolve_dimensions(world_size, dp, tp, pp)
    if not dist.is_initialized():
        if world_size != 1:
            raise ValueError("a multi-rank mesh requires an initialized process group")
        return None, dp, tp, pp
    mesh = init_device_mesh(device_type, (pp, dp, tp), mesh_dim_names=("pp", "dp", "tp"))
    return mesh, dp, tp, pp


@dataclass(frozen=True)
class ParallelContext:
    """Single source of truth for coordinates, DP reductions and world decisions."""

    rank: int
    dp: int
    tp: int
    pp: int
    mesh: DeviceMesh | None = None

    def __post_init__(self) -> None:
        resolve_dimensions(self.dp * self.tp * self.pp, self.dp, self.tp, self.pp)
        if not 0 <= self.rank < self.dp * self.tp * self.pp:
            raise ValueError("rank outside parallel topology")

    @property
    def dp_rank(self) -> int:
        return (self.rank // self.tp) % self.dp

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp

    @property
    def pp_rank(self) -> int:
        return self.rank // (self.dp * self.tp)

    @property
    def dimensions(self) -> dict:
        return {"dp": self.dp, "tp": self.tp, "pp": self.pp}

    def global_rank(self, *, pp: int, dp: int, tp: int) -> int:
        """Translate coordinates into the GLOBAL rank required by broadcast(src)."""
        if not (0 <= pp < self.pp and 0 <= dp < self.dp and 0 <= tp < self.tp):
            raise ValueError("coordinates outside parallel topology")
        return (pp * self.dp + dp) * self.tp + tp

    def mean_dp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Average replicas only, never TP peers or zero-loss PP intermediates."""
        result = tensor.clone()
        if self.dp > 1:
            if self.mesh is None:
                raise ValueError("DP reduction requires a mesh")
            dist.all_reduce(result, group=self.mesh["dp"].get_group())
            result /= self.dp
        return result

    def any_rank(self, flag: bool, device: str | torch.device) -> bool:
        """Make one collective control decision (spike, nonfinite gradient, etc.)."""
        result = torch.tensor(int(flag), device=device)
        if dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.MAX)
        return bool(result.item())
