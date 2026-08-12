from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def all_true(self, value: bool) -> bool:
        if not self.distributed:
            return value
        tensor = torch.tensor(int(value), device=self.device, dtype=torch.int32)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item())

    def window_box_normalizer(self, local_num_boxes: int) -> float:
        tensor = torch.tensor(float(local_num_boxes), device=self.device)
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return max(float(tensor.item()) / self.world_size, 1.0)


def initialize_distributed(*, allow_cpu: bool = False) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    local_world_size = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE",
            str(torch.cuda.device_count() if torch.cuda.is_available() else 1),
        )
    )
    os.environ.setdefault("LOCAL_SIZE", str(local_world_size))

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, but only "
                f"{torch.cuda.device_count()} GPU(s) are visible"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif allow_cpu:
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise RuntimeError(
            "CUDA is not available. Check Docker GPU access or pass --allow-cpu "
            "only for configuration tests."
        )

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=6),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_sums(
    values: dict[str, float], context: DistributedContext
) -> dict[str, float]:
    if not context.distributed:
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor(
        [values[key] for key in keys], dtype=torch.float64, device=context.device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist())}
