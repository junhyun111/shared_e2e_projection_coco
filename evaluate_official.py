from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

from projection_coco.config import TrainConfig, configure_torch_cache, seed_everything
from projection_coco.data import make_val_loader, prepare_val_data
from projection_coco.detector import build_official_components
from projection_coco.distributed import cleanup_distributed, initialize_distributed
from projection_coco.evaluation_report import write_official_evaluation_report
from projection_coco.evaluator import evaluate_coco
from projection_coco.methods import make_research_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "r50_deformable_detr-checkpoint.pth"


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official Deformable DETR R50 checkpoint on full "
            "COCO 2017 val and write JSON/PNG reports."
        )
    )
    parser.add_argument("checkpoint", type=Path, nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env-file", type=Path, default=PROJECT_DIR / ".env")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--torch-cache", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    _load_env_file(args.env_file.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data_root = args.data_root
    if data_root is None:
        data_root = Path(os.environ.get("COCO_ROOT", PROJECT_DIR / "data" / "coco"))
    data_root = data_root.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_root = Path(os.environ.get("OUTPUT_ROOT", PROJECT_DIR / "artifacts"))
        output_dir = output_root / "official_checkpoint_eval"
    output_dir = output_dir.expanduser().resolve()
    return checkpoint_path, data_root, output_dir


def _validate_inputs(checkpoint_path: Path, data_root: Path) -> None:
    required = (
        checkpoint_path,
        data_root / "val2017",
        data_root / "annotations" / "instances_val2017.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required evaluation input is missing:\n{formatted}")


def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(
            "Expected an official Deformable DETR checkpoint containing a 'model' state dict"
        )
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path, data_root, output_dir = _resolve_paths(args)
    _validate_inputs(checkpoint_path, data_root)
    configure_torch_cache(
        args.torch_cache
        or os.environ.get("TORCH_CACHE")
        or os.environ.get("TORCH_HOME")
    )
    config = TrainConfig(
        data_root=data_root,
        output_root=output_dir.parent,
        method="baseline",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_limit=None,
        val_limit=None,
        cache_mode=False,
    )

    context = initialize_distributed(allow_cpu=args.allow_cpu)
    model = None
    try:
        seed_everything(config.seed, deterministic=False)
        bundle = prepare_val_data(config, context)
        detector, criterion, postprocessors = build_official_components(
            config, context.device, pretrained_backbone=False
        )
        del criterion
        checkpoint = _load_checkpoint(checkpoint_path)
        detector.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        gc.collect()
        model = make_research_model(detector, config, context.device)
        val_loader = make_val_loader(config, bundle, context)

        if context.is_main:
            print(
                f"[official-eval] checkpoint={checkpoint_path} "
                f"COCO-val={len(bundle.val_dataset)} world_size={context.world_size}"
            )
        metrics = evaluate_coco(
            model, postprocessors, val_loader, bundle.coco_api, context
        )
        if context.is_main:
            json_path, png_path, report = write_official_evaluation_report(
                metrics,
                output_dir,
                checkpoint_path=checkpoint_path,
                data_root=data_root,
                num_images=len(bundle.val_dataset),
                world_size=context.world_size,
                batch_size=config.batch_size,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"[official-eval] JSON: {json_path}")
            print(f"[official-eval] PNG:  {png_path}")
        context.barrier()
    finally:
        if model is not None:
            model.close()
        cleanup_distributed(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
