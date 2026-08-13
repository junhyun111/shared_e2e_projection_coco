import torch

from projection_coco.distributed import DistributedContext, reduce_sums


def test_reduce_sums_converts_device_tensors_once_at_the_boundary():
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )

    result = reduce_sums(
        {
            "loss": torch.tensor(1.25, dtype=torch.float64),
            "steps": torch.tensor(3.0, dtype=torch.float64),
        },
        context,
    )

    assert result == {"loss": 1.25, "steps": 3.0}


def test_all_true_accepts_scalar_tensor():
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )

    assert context.all_true(torch.tensor(True)) is True
    assert context.all_true(torch.tensor(False)) is False
