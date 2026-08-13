from __future__ import annotations

import time

import torch
from tqdm.auto import tqdm

from .distributed import DistributedContext
from .upstream import ensure_upstream_imports

import io
from contextlib import redirect_stdout


ensure_upstream_imports()
from datasets.coco_eval import CocoEvaluator  # noqa: E402


@torch.no_grad()
def evaluate_coco(
    model,
    postprocessors: dict,
    val_loader,
    coco_api,
    context: DistributedContext,
) -> dict[str, float]:
    model.eval()
    evaluator = CocoEvaluator(coco_api, ("bbox",))
    start = time.perf_counter()
    iterator = tqdm(
        val_loader,
        desc="COCO validation",
        leave=False,
        disable=not context.is_main,
        mininterval=0.5,
    )
    for samples, targets in iterator:
        samples = samples.to(context.device, non_blocking=True)
        device_targets = [
            {key: value.to(context.device) for key, value in target.items()}
            for target in targets
        ]
        result = model(samples, None)
        outputs = result["detector_outputs"]
        original_sizes = torch.stack(
            [target["orig_size"] for target in device_targets], dim=0
        )
        predictions = postprocessors["bbox"](outputs, original_sizes)
        evaluator.update(
            {
                target["image_id"].item(): prediction
                for target, prediction in zip(device_targets, predictions)
            }
        )

    evaluator.synchronize_between_processes()
    evaluator.accumulate()

    if context.is_main:
        evaluator.summarize()
    else:
        with redirect_stdout(io.StringIO()):
            evaluator.summarize()

    stats = evaluator.coco_eval["bbox"].stats
    return {
        "map": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "mar100": float(stats[8]),
        "val_seconds": time.perf_counter() - start,
    }
