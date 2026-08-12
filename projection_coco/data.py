from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from .config import TrainConfig
from .distributed import DistributedContext
from .upstream import ensure_upstream_imports


ensure_upstream_imports()
from datasets import build_dataset, get_coco_api_from_dataset  # noqa: E402
from util.misc import collate_fn  # noqa: E402


@dataclass(frozen=True)
class DataBundle:
    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset
    coco_api: object


def _deterministic_subset(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    generator = np.random.default_rng(seed)
    indices = sorted(
        int(index)
        for index in generator.choice(len(dataset), size=limit, replace=False)
    )
    return Subset(dataset, indices)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_data(config: TrainConfig, context: DistributedContext) -> DataBundle:
    args = config.official_args(context.device)
    train_dataset = build_dataset("train", args)
    val_dataset = build_dataset("val", args)
    coco_api = get_coco_api_from_dataset(val_dataset)
    train_dataset = _deterministic_subset(
        train_dataset, config.train_limit, config.seed
    )
    val_dataset = _deterministic_subset(
        val_dataset, config.val_limit, config.seed + 1
    )
    return DataBundle(train_dataset, val_dataset, coco_api)


def make_train_loader(
    config: TrainConfig,
    bundle: DataBundle,
    context: DistributedContext,
    *,
    epoch: int,
):
    if context.distributed:
        sampler = DistributedSampler(
            bundle.train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
        sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(
            bundle.train_dataset,
            generator=torch.Generator().manual_seed(config.seed + epoch * 100_003),
        )
    generator = torch.Generator().manual_seed(
        config.seed + context.rank + epoch * 100_003
    )
    return DataLoader(
        bundle.train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=context.device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def make_val_loader(
    config: TrainConfig, bundle: DataBundle, context: DistributedContext
):
    if context.distributed:
        sampler = DistributedSampler(
            bundle.val_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = SequentialSampler(bundle.val_dataset)
    return DataLoader(
        bundle.val_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=context.device.type == "cuda",
        worker_init_fn=_seed_worker,
    )
