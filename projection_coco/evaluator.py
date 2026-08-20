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
    *,
    inference_precision: str = "fp32",
) -> dict[str, float]:
    if inference_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("inference_precision must be fp32, fp16 or bf16")
    if inference_precision != "fp32" and context.device.type != "cuda":
        raise ValueError("Mixed-precision inference requires CUDA")
    autocast_dtype = {
        "fp32": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[inference_precision]
    model.eval()
    evaluator = CocoEvaluator(coco_api, ("bbox",))
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
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
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=inference_precision != "fp32",
        ):
            result = model(samples, None)
        detector_outputs = result["detector_outputs"]
        outputs = {
            **detector_outputs,
            "pred_logits": detector_outputs["pred_logits"].float(),
            "pred_boxes": detector_outputs["pred_boxes"].float(),
        }
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

    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    stats = evaluator.coco_eval["bbox"].stats
    return {
        "map": float(stats[0]),
        "map50": float(stats[1]),
        "map75": float(stats[2]),
        "map_small": float(stats[3]),
        "map_medium": float(stats[4]),
        "map_large": float(stats[5]),
        "mar1": float(stats[6]),
        "mar10": float(stats[7]),
        "mar100": float(stats[8]),
        "mar_small": float(stats[9]),
        "mar_medium": float(stats[10]),
        "mar_large": float(stats[11]),
        "val_seconds": time.perf_counter() - start,
    }
