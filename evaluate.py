from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from projection_coco.config import TrainConfig, configure_torch_cache
from projection_coco.data import make_val_loader, prepare_data
from projection_coco.detector import build_official_components
from projection_coco.distributed import cleanup_distributed, initialize_distributed
from projection_coco.evaluator import evaluate_coco
from projection_coco.methods import make_research_model
from projection_coco.upstream import upstream_commit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a research checkpoint on COCO."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--torch-cache", type=Path, default=os.environ.get("TORCH_HOME")
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("config")
    if not saved_config:
        raise ValueError("Checkpoint does not contain its training configuration")
    if checkpoint.get("upstream_commit") != upstream_commit():
        raise ValueError("Checkpoint upstream commit does not match this checkout")
    default_output_root = (
        checkpoint_path.parents[3]
        if len(checkpoint_path.parents) > 3
        else checkpoint_path.parent
    )
    output_root = args.output_root or default_output_root
    overrides = {
        "data_root": args.data_root,
        "output_root": output_root,
        "num_workers": args.num_workers,
        "train_limit": None,
        "val_limit": None,
    }
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    config = TrainConfig.from_dict(saved_config, **overrides)
    configure_torch_cache(args.torch_cache)
    context = initialize_distributed(allow_cpu=args.allow_cpu)
    try:
        bundle = prepare_data(config, context)
        detector, _, postprocessors = build_official_components(
            config, context.device, pretrained_backbone=False
        )
        model = make_research_model(detector, config, context.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        val_loader = make_val_loader(config, bundle, context)
        metrics = evaluate_coco(
            model, postprocessors, val_loader, bundle.coco_api, context
        )
        if context.is_main:
            print(json.dumps(metrics, indent=2))
        model.close()
    finally:
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
