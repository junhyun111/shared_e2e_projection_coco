from __future__ import annotations

import torch

from .config import TrainConfig
from .criterion import WindowNormalizedSetCriterion
from .upstream import ensure_upstream_imports


def build_official_components(
    config: TrainConfig,
    device: torch.device,
    *,
    pretrained_backbone: bool,
):
    ensure_upstream_imports()
    import models
    import models.backbone as backbone_module

    original_is_main_process = backbone_module.is_main_process
    backbone_module.is_main_process = lambda: pretrained_backbone
    try:
        detector, official_criterion, postprocessors = models.build_model(
            config.official_args(device)
        )
    finally:
        backbone_module.is_main_process = original_is_main_process

    detector.to(device)
    criterion = WindowNormalizedSetCriterion.from_official(official_criterion).to(
        device
    )
    return detector, criterion, postprocessors

