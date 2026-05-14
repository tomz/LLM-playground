"""3D device mesh: data x tensor x pipeline."""
from __future__ import annotations
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def build_mesh(world_size: int, dp: int, tp: int, pp: int, device_type: str = "cuda"):
    """Resolve dp=-1 to fill, validate, return DeviceMesh with named dims."""
    if dp == -1:
        assert world_size % (tp * pp) == 0, f"world {world_size} not divisible by tp*pp = {tp*pp}"
        dp = world_size // (tp * pp)
    assert dp * tp * pp == world_size, f"dp*tp*pp ({dp}*{tp}*{pp}) != world_size ({world_size})"
    if not dist.is_initialized():
        # single-process dev mode: return None and let callers no-op
        return None, dp, tp, pp
    mesh = init_device_mesh(device_type, (pp, dp, tp), mesh_dim_names=("pp", "dp", "tp"))
    return mesh, dp, tp, pp
