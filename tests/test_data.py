from pathlib import Path

import torch

from projection_coco.config import TrainConfig
from projection_coco.data import (
    DataBundle,
    make_train_loader,
    make_val_loader,
    set_train_loader_epoch,
)
from projection_coco.distributed import DistributedContext
from util.misc import NestedTensor


class _Pinnable:
    def __init__(self, *, pinned: bool = False):
        self.pinned = pinned

    def pin_memory(self):
        return _Pinnable(pinned=True)


def _config(tmp_path: Path, *, num_workers: int) -> TrainConfig:
    return TrainConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "artifacts",
        num_workers=num_workers,
    )


def _bundle(size: int = 32) -> DataBundle:
    dataset = list(range(size))
    return DataBundle(dataset, dataset, coco_api=None)


def _context() -> DistributedContext:
    return DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )


def test_loader_enables_persistent_prefetch_only_with_workers(tmp_path):
    with_workers = _config(tmp_path, num_workers=2)
    train_loader = make_train_loader(with_workers, _bundle(), _context())
    val_loader = make_val_loader(with_workers, _bundle(), _context())

    assert train_loader.persistent_workers is True
    assert train_loader.prefetch_factor == 4
    assert val_loader.persistent_workers is True
    assert val_loader.prefetch_factor == 4

    without_workers = _config(tmp_path, num_workers=0)
    loader = make_train_loader(without_workers, _bundle(), _context())
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None


def test_single_process_sampler_is_reseeded_for_each_epoch(tmp_path):
    config = _config(tmp_path, num_workers=0)
    loader = make_train_loader(config, _bundle(), _context())

    set_train_loader_epoch(loader, config, epoch=7)
    first_order = list(loader.sampler)
    set_train_loader_epoch(loader, config, epoch=7)
    repeated_order = list(loader.sampler)
    set_train_loader_epoch(loader, config, epoch=8)
    next_order = list(loader.sampler)

    assert first_order == repeated_order
    assert first_order != next_order


def test_nested_tensor_pins_image_and_mask_storage():
    nested = NestedTensor(_Pinnable(), _Pinnable())

    pinned = nested.pin_memory()

    assert pinned.tensors.pinned is True
    assert pinned.mask.pinned is True
