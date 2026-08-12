from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TrainConfig


@dataclass(frozen=True)
class OptimizerSummary:
    normal_names: tuple[str, ...]
    backbone_names: tuple[str, ...]
    linear_projection_names: tuple[str, ...]

    def as_dict(self) -> dict[str, int]:
        return {
            "normal_parameters": len(self.normal_names),
            "backbone_parameters": len(self.backbone_names),
            "linear_projection_parameters": len(self.linear_projection_names),
        }


def make_optimizer(model: torch.nn.Module, config: TrainConfig):
    groups: dict[str, list[torch.nn.Parameter]] = {
        "normal": [],
        "backbone": [],
        "linear_projection": [],
    }
    names: dict[str, list[str]] = {key: [] for key in groups}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "reference_points" in name or "sampling_offsets" in name:
            group = "linear_projection"
        elif "detector.backbone.0" in name:
            group = "backbone"
        else:
            group = "normal"
        groups[group].append(parameter)
        names[group].append(name)

    grouped_ids = [
        id(parameter) for values in groups.values() for parameter in values
    ]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError(
            "A trainable parameter was assigned to multiple optimizer groups"
        )
    if set(grouped_ids) != set(trainable_ids):
        raise RuntimeError("Optimizer groups do not cover every trainable parameter")
    if not all(groups.values()):
        empty = [name for name, values in groups.items() if not values]
        raise RuntimeError(f"Empty official optimizer group(s): {', '.join(empty)}")

    optimizer = torch.optim.AdamW(
        [
            {"params": groups["normal"], "lr": config.lr, "name": "normal"},
            {
                "params": groups["backbone"],
                "lr": config.backbone_lr,
                "name": "backbone",
            },
            {
                "params": groups["linear_projection"],
                "lr": config.lr * config.linear_proj_lr_mult,
                "name": "linear_projection",
            },
        ],
        weight_decay=config.weight_decay,
    )
    summary = OptimizerSummary(
        normal_names=tuple(names["normal"]),
        backbone_names=tuple(names["backbone"]),
        linear_projection_names=tuple(names["linear_projection"]),
    )
    return optimizer, summary
