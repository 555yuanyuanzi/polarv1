from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class DistributedState:
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = -1
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def init_distributed_mode(enable: bool = True) -> DistributedState:
    if not enable:
        return DistributedState(device=_default_device())

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        dist.barrier()
        return DistributedState(True, rank, world_size, local_rank, device)
    return DistributedState(device=_default_device())


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(state: DistributedState) -> bool:
    return state.rank == 0


def synchronize() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_tensor(value: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced /= dist.get_world_size()
    return reduced


def reduce_dict(metrics: dict[str, Any], average: bool = True) -> dict[str, Any]:
    reduced: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            reduced[key] = reduce_tensor(value, average=average)
        else:
            reduced[key] = value
    return reduced
