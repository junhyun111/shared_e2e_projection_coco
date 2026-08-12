from __future__ import annotations

import os
from pathlib import Path

import torch

from .config import TrainConfig, model_fingerprint, seed_everything
from .detector import build_official_components
from .distributed import DistributedContext
from .upstream import upstream_commit


def _atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def ensure_detector_initialization(
    config: TrainConfig, context: DistributedContext
):
    config.create_output_dirs()
    path = config.initialization_path
    if context.is_main and not path.is_file():
        seed_everything(config.seed, config.deterministic)
        detector, _, _ = build_official_components(
            config, context.device, pretrained_backbone=True
        )
        fingerprint = model_fingerprint(detector)
        _atomic_save(
            {
                "format_version": 1,
                "upstream_commit": upstream_commit(),
                "seed": config.seed,
                "detector_recipe": config.detector_recipe_dict(),
                "detector_recipe_fingerprint": config.detector_recipe_fingerprint,
                "detector_fingerprint": fingerprint,
                "model_state_dict": _cpu_state_dict(detector),
            },
            path,
        )
        del detector
        if context.device.type == "cuda":
            torch.cuda.empty_cache()
    context.barrier()

    if not path.is_file():
        raise FileNotFoundError(f"Detector initialization was not created: {path}")
    initialization = torch.load(path, map_location="cpu")
    if initialization.get("upstream_commit") != upstream_commit():
        raise ValueError("Initialization was created from a different upstream commit")
    if (
        initialization.get("detector_recipe_fingerprint")
        != config.detector_recipe_fingerprint
    ):
        raise ValueError("Initialization detector recipe does not match this run")
    if int(initialization.get("seed", -1)) != config.seed:
        raise ValueError("Initialization seed does not match this run")

    seed_everything(config.seed, config.deterministic)
    detector, criterion, postprocessors = build_official_components(
        config, context.device, pretrained_backbone=False
    )
    detector.load_state_dict(initialization["model_state_dict"], strict=True)
    fingerprint = model_fingerprint(detector)
    if fingerprint != initialization.get("detector_fingerprint"):
        raise RuntimeError("Loaded detector fingerprint does not match initialization")
    return detector, criterion, postprocessors, fingerprint
