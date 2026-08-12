from __future__ import annotations

import copy

import torch

from .upstream import ensure_upstream_imports


ensure_upstream_imports()
from models.deformable_detr import SetCriterion  # noqa: E402


class WindowNormalizedSetCriterion(SetCriterion):
    """Official criterion with an optional accumulation-window normalizer."""

    @classmethod
    def from_official(cls, criterion: SetCriterion) -> "WindowNormalizedSetCriterion":
        return cls(
            criterion.num_classes,
            criterion.matcher,
            criterion.weight_dict,
            criterion.losses,
            focal_alpha=criterion.focal_alpha,
        )

    def forward(
        self,
        outputs: dict,
        targets: list[dict],
        *,
        num_boxes_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs_without_aux = {
            key: value
            for key, value in outputs.items()
            if key not in {"aux_outputs", "enc_outputs"}
        }
        indices = self.matcher(outputs_without_aux, targets)

        if num_boxes_override is None:
            num_boxes = sum(len(target["labels"]) for target in targets)
            num_boxes_tensor = torch.as_tensor(
                [num_boxes],
                dtype=torch.float,
                device=next(iter(outputs.values())).device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(num_boxes_tensor)
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1
            num_boxes = max(float(num_boxes_tensor.item()) / world_size, 1.0)
        else:
            num_boxes = max(float(num_boxes_override), 1.0)

        losses: dict[str, torch.Tensor] = {}
        for loss_name in self.losses:
            losses.update(
                self.get_loss(loss_name, outputs, targets, indices, num_boxes)
            )

        for index, auxiliary_outputs in enumerate(outputs.get("aux_outputs", [])):
            auxiliary_indices = self.matcher(auxiliary_outputs, targets)
            for loss_name in self.losses:
                if loss_name == "masks":
                    continue
                kwargs = {"log": False} if loss_name == "labels" else {}
                layer_losses = self.get_loss(
                    loss_name,
                    auxiliary_outputs,
                    targets,
                    auxiliary_indices,
                    num_boxes,
                    **kwargs,
                )
                losses.update(
                    {f"{key}_{index}": value for key, value in layer_losses.items()}
                )

        if "enc_outputs" in outputs:
            encoder_targets = copy.deepcopy(targets)
            for target in encoder_targets:
                target["labels"] = torch.zeros_like(target["labels"])
            encoder_outputs = outputs["enc_outputs"]
            encoder_indices = self.matcher(encoder_outputs, encoder_targets)
            for loss_name in self.losses:
                if loss_name == "masks":
                    continue
                kwargs = {"log": False} if loss_name == "labels" else {}
                encoder_losses = self.get_loss(
                    loss_name,
                    encoder_outputs,
                    encoder_targets,
                    encoder_indices,
                    num_boxes,
                    **kwargs,
                )
                losses.update(
                    {f"{key}_enc": value for key, value in encoder_losses.items()}
                )
        return losses


def weighted_detection_loss(
    loss_dict: dict[str, torch.Tensor], weight_dict: dict[str, float]
) -> torch.Tensor:
    weighted = [
        value * weight_dict[key]
        for key, value in loss_dict.items()
        if key in weight_dict
    ]
    if not weighted:
        raise RuntimeError("Detection criterion returned no weighted losses")
    return sum(weighted)

