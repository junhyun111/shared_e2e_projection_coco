from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TrainConfig, seed_everything
from .geometry import cxcywh_to_xyxy, generalized_box_iou, inverse_sigmoid


class ResidualPatchAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.fc2(F.gelu(self.fc1(self.norm(features))))


class BaselineModel(nn.Module):
    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(
        self,
        samples,
        targets: list[dict] | None = None,
        *,
        aux_normalizer: float | None = None,
    ) -> dict:
        del targets, aux_normalizer
        return {
            "detector_outputs": self.detector(samples),
            "aux_loss": None,
            "aux_l1": None,
            "aux_giou": None,
            "aux_total": 0,
            "aux_used": 0,
            "aux_collisions": 0,
            "feature_stats": {},
        }

    def close(self) -> None:
        return None


class AuxiliaryModel(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        *,
        feature_level: int,
        use_adapter: bool,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.feature_level = feature_level
        self.adapter = (
            ResidualPatchAdapter(detector.transformer.d_model)
            if use_adapter
            else nn.Identity()
        )
        self._encoder_cache: dict[str, torch.Tensor] = {}
        self._hook_handle = detector.transformer.encoder.register_forward_hook(
            self._capture_encoder
        )

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def shared_bbox_head(self):
        return self.detector.bbox_embed[-1]

    @property
    def encoder_representation(self) -> torch.Tensor:
        if not self._encoder_cache:
            raise RuntimeError("Encoder cache is empty; run a detector forward first")
        return self._encoder_cache["memory"]

    def _capture_encoder(self, module, inputs, output) -> None:
        del module
        if len(inputs) < 6:
            raise RuntimeError("Unsupported official Deformable DETR encoder signature")
        self._encoder_cache = {
            "memory": output,
            "spatial_shapes": inputs[1],
            "level_start_index": inputs[2],
            "padding_mask": inputs[5],
        }

    def _valid_level_layout(self, batch_index: int):
        cache = self._encoder_cache
        level = self.feature_level
        full_height, full_width = [
            int(value) for value in cache["spatial_shapes"][level].tolist()
        ]
        start = int(cache["level_start_index"][level].item())
        padding_mask = cache["padding_mask"][
            batch_index, start : start + full_height * full_width
        ].reshape(full_height, full_width)
        valid_mask = ~padding_mask
        valid_height = int(valid_mask.any(dim=1).sum().item())
        valid_width = int(valid_mask.any(dim=0).sum().item())
        if valid_height <= 0 or valid_width <= 0:
            raise RuntimeError("Empty valid feature region")
        return start, full_width, valid_height, valid_width

    def select_aux_samples(self, targets: list[dict]) -> dict:
        memory = self.encoder_representation
        features = []
        references = []
        target_boxes = []
        total = 0
        collisions = 0

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"]
            object_count = len(boxes)
            total += object_count
            if object_count == 0:
                continue
            start, full_width, valid_height, valid_width = self._valid_level_layout(
                batch_index
            )
            centers = boxes[:, :2].clamp(min=0.0, max=1.0 - 1e-7)
            columns = torch.floor(centers[:, 0] * valid_width).long().clamp(
                0, valid_width - 1
            )
            rows = torch.floor(centers[:, 1] * valid_height).long().clamp(
                0, valid_height - 1
            )
            cells = rows * valid_width + columns
            keep = torch.zeros(object_count, dtype=torch.bool, device=boxes.device)
            for cell in cells.unique():
                indices = torch.where(cells == cell)[0]
                cell_row = torch.div(cell, valid_width, rounding_mode="floor")
                cell_column = cell % valid_width
                cell_center = boxes.new_tensor(
                    [
                        (float(cell_column.item()) + 0.5) / valid_width,
                        (float(cell_row.item()) + 0.5) / valid_height,
                    ]
                )
                distances = (centers[indices] - cell_center).square().sum(dim=-1)
                keep[indices[distances.argmin()]] = True

            kept_indices = torch.where(keep)[0]
            collisions += object_count - int(kept_indices.numel())
            for object_index in kept_indices.tolist():
                row = int(rows[object_index])
                column = int(columns[object_index])
                flat_index = start + row * full_width + column
                features.append(memory[batch_index, flat_index])
                references.append(
                    memory.new_tensor(
                        [
                            (column + 0.5) / valid_width,
                            (row + 0.5) / valid_height,
                        ]
                    )
                )
                target_boxes.append(boxes[object_index])

        if not features:
            empty = memory.new_empty
            return {
                "features": empty((0, memory.shape[-1])),
                "references": empty((0, 2)),
                "targets": empty((0, 4)),
                "total": total,
                "collisions": collisions,
            }
        return {
            "features": torch.stack(features),
            "references": torch.stack(references),
            "targets": torch.stack(target_boxes),
            "total": total,
            "collisions": collisions,
        }

    def auxiliary_loss(self, targets: list[dict], normalizer: float) -> dict:
        selected = self.select_aux_samples(targets)
        raw_features = selected["features"]
        used = len(raw_features)
        if used == 0:
            zero = (
                self.encoder_representation.sum() * 0.0
                + self.adapter(raw_features).sum()
            )
            return {
                "loss": zero,
                "l1": zero,
                "giou": zero,
                "selected": selected,
                "stats": {},
            }

        adapted = self.adapter(raw_features)
        delta = self.shared_bbox_head(adapted)
        center_logits = delta[..., :2] + inverse_sigmoid(selected["references"])
        predicted = torch.cat((center_logits, delta[..., 2:]), dim=-1).sigmoid()
        target_boxes = selected["targets"]
        loss_l1 = (
            F.l1_loss(predicted, target_boxes, reduction="none").sum()
            / normalizer
        )
        giou = generalized_box_iou(
            cxcywh_to_xyxy(predicted), cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = (1.0 - torch.diag(giou)).sum() / normalizer
        return {
            "loss": 5.0 * loss_l1 + 2.0 * loss_giou,
            "l1": loss_l1,
            "giou": loss_giou,
            "selected": selected,
            "stats": {
                "raw_feature_norm": raw_features.detach().norm(dim=-1).mean(),
                "adapted_feature_norm": adapted.detach().norm(dim=-1).mean(),
            },
        }

    def forward(
        self,
        samples,
        targets: list[dict] | None = None,
        *,
        aux_normalizer: float | None = None,
    ) -> dict:
        self._encoder_cache = {}
        detector_outputs = self.detector(samples)
        result = {
            "detector_outputs": detector_outputs,
            "aux_loss": None,
            "aux_l1": None,
            "aux_giou": None,
            "aux_total": 0,
            "aux_used": 0,
            "aux_collisions": 0,
            "feature_stats": {},
        }
        if not self.training or targets is None:
            return result
        if aux_normalizer is None or aux_normalizer <= 0:
            raise ValueError("Auxiliary training requires a positive window normalizer")
        auxiliary = self.auxiliary_loss(targets, aux_normalizer)
        selected = auxiliary["selected"]
        result.update(
            {
                "aux_loss": auxiliary["loss"],
                "aux_l1": auxiliary["l1"],
                "aux_giou": auxiliary["giou"],
                "aux_total": selected["total"],
                "aux_used": len(selected["features"]),
                "aux_collisions": selected["collisions"],
                "feature_stats": auxiliary["stats"],
            }
        )
        return result


def make_research_model(detector: nn.Module, config: TrainConfig, device: torch.device):
    if config.method == "baseline":
        model = BaselineModel(detector)
    else:
        seed_everything(config.seed + 1_000_003, config.deterministic)
        model = AuxiliaryModel(
            detector,
            feature_level=config.feature_level,
            use_adapter=config.uses_adapter,
        )
    return model.to(device)
