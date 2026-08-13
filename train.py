from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from projection_coco.config import (
    METHODS,
    TrainConfig,
    configure_torch_cache,
)
from projection_coco.data import prepare_data
from projection_coco.distributed import cleanup_distributed, initialize_distributed
from projection_coco.engine import train


PROJECT_DIR = Path(__file__).resolve().parent


def optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train official Deformable DETR or a Shared-E2E ablation on COCO."
    )
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("COCO_ROOT", PROJECT_DIR / "data" / "coco")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ROOT", PROJECT_DIR / "artifacts")),
    )
    parser.add_argument(
        "--torch-cache", type=Path, default=os.environ.get("TORCH_HOME")
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-global-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--feature-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=optional_positive_int, default=None)
    parser.add_argument("--val-limit", type=optional_positive_int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--gradient-log-every", type=int, default=100)
    parser.add_argument(
        "--performance-log-every",
        type=int,
        default=0,
        help="Synchronize and report data/optimizer-step timing every N steps (0 disables)",
    )
    parser.add_argument(
        "--resume", default=None, help="Checkpoint path or 'auto' for latest.pt"
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental deviation from the official FP32 recipe",
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--cache-mode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip-initial-eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_torch_cache(args.torch_cache)
    config = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_global_batch_size=args.target_global_batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        aux_weight=args.aux_weight,
        feature_level=args.feature_level,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        eval_every=args.eval_every,
        save_every=args.save_every,
        gradient_log_every=args.gradient_log_every,
        performance_log_every=args.performance_log_every,
        amp=args.amp,
        deterministic=args.deterministic,
        cache_mode=args.cache_mode,
        skip_initial_eval=args.skip_initial_eval,
    )
    context = initialize_distributed(allow_cpu=args.allow_cpu)
    try:
        bundle = prepare_data(config, context)
        context.barrier()
        train(config, bundle, context, resume_from=args.resume)
        context.barrier()
    finally:
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
