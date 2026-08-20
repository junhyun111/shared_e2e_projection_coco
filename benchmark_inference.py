from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from projection_coco.config import TrainConfig, configure_torch_cache
from projection_coco.data import make_val_loader, prepare_val_data
from projection_coco.detector import build_official_components
from projection_coco.distributed import (
    cleanup_distributed,
    initialize_distributed,
    runtime_metadata,
)
from projection_coco.methods import make_research_model
from projection_coco.upstream import upstream_commit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark research-checkpoint inference. Data loading and COCO "
            "post-processing are excluded; H2D transfer and model forward are included."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--torch-cache", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--inference-precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    return parser


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _batches(loader, count: int):
    iterator = iter(loader)
    for _ in range(count):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)


@torch.inference_mode()
def _run_model(model, samples, device: torch.device, precision: str) -> int:
    samples = samples.to(device, non_blocking=True)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(
        device_type="cuda", dtype=dtype, enabled=precision != "fp32"
    ):
        result = model(samples, None)
    # Retain a dependency on the output before the explicit CUDA synchronize.
    result["detector_outputs"]["pred_logits"].sum()
    return int(samples.tensors.shape[0])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * percentile), len(ordered) - 1)
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.measure_steps <= 0 or args.warmup_steps < 0:
        raise ValueError("batch-size/measure-steps must be positive and warmup non-negative")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("upstream_commit") != upstream_commit():
        raise ValueError("Checkpoint upstream commit does not match this checkout")
    saved_config = checkpoint.get("config")
    if not saved_config:
        raise ValueError("Checkpoint does not contain its training configuration")

    configure_torch_cache(args.torch_cache)
    context = initialize_distributed(allow_cpu=False)
    model = None
    try:
        config = TrainConfig.from_dict(
            saved_config,
            data_root=args.data_root,
            output_root=checkpoint_path.parent,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_limit=None,
            val_limit=None,
        )
        bundle = prepare_val_data(config, context)
        detector, criterion, _ = build_official_components(
            config, context.device, pretrained_backbone=False
        )
        del criterion
        model = make_research_model(detector, config, context.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.eval()
        loader = make_val_loader(config, bundle, context)

        warmup_and_measure = _batches(
            loader, args.warmup_steps + args.measure_steps
        )
        for _ in range(args.warmup_steps):
            samples, _ = next(warmup_and_measure)
            _run_model(model, samples, context.device, args.inference_precision)
        torch.cuda.synchronize(context.device)
        context.barrier()
        torch.cuda.reset_peak_memory_stats(context.device)

        latencies: list[float] = []
        local_images = 0
        for _ in range(args.measure_steps):
            samples, _ = next(warmup_and_measure)
            torch.cuda.synchronize(context.device)
            step_start = time.perf_counter()
            local_images += _run_model(
                model, samples, context.device, args.inference_precision
            )
            torch.cuda.synchronize(context.device)
            latencies.append(time.perf_counter() - step_start)

        totals = torch.tensor(
            [float(local_images), float(sum(latencies)), float(len(latencies))],
            device=context.device,
            dtype=torch.float64,
        )
        max_values = torch.tensor(
            [
                sum(latencies),
                torch.cuda.max_memory_allocated(context.device) / 2**20,
            ],
            device=context.device,
            dtype=torch.float64,
        )
        if context.distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
            gathered: list[list[float] | None] = [None] * context.world_size
            dist.all_gather_object(gathered, latencies)
            all_latencies = [
                value
                for rank_values in gathered
                if rank_values is not None
                for value in rank_values
            ]
        else:
            all_latencies = latencies

        if context.is_main:
            images, latency_sum, batches = totals.cpu().tolist()
            max_compute, peak_cuda_mb = max_values.cpu().tolist()
            report = {
                "measurement": "H2D transfer + model forward; loader/postprocess excluded",
                "checkpoint": str(checkpoint_path),
                "runtime": runtime_metadata(),
                "precision": args.inference_precision,
                "world_size": context.world_size,
                "batch_size_per_gpu": args.batch_size,
                "warmup_steps": args.warmup_steps,
                "measure_steps_per_gpu": args.measure_steps,
                "images": int(images),
                "aggregate_fps": images / max(max_compute, 1e-12),
                "mean_batch_latency_ms": latency_sum / max(batches, 1.0) * 1000.0,
                "p50_batch_latency_ms": _percentile(all_latencies, 0.50) * 1000.0,
                "p95_batch_latency_ms": _percentile(all_latencies, 0.95) * 1000.0,
                "peak_cuda_mb_max_rank": peak_cuda_mb,
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))
        context.barrier()
    finally:
        if model is not None:
            model.close()
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
