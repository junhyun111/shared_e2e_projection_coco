from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TrainConfig, seed_everything
from .geometry import cxcywh_to_xyxy, generalized_box_iou, inverse_sigmoid


def _stable_group_argmin(
    group_ids: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Return the first minimum-value index for each group, in input order."""
    if group_ids.numel() == 0:
        return group_ids.new_empty((0,))
    distance_order = torch.argsort(values, stable=True)
    group_order = torch.argsort(group_ids[distance_order], stable=True)
    ordered_indices = distance_order[group_order]
    ordered_groups = group_ids[ordered_indices]
    first_in_group = torch.ones_like(ordered_groups, dtype=torch.bool)
    first_in_group[1:] = ordered_groups[1:] != ordered_groups[:-1]
    return ordered_indices[first_in_group].sort().values


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

    def _level_layout(self):
        cache = self._encoder_cache
        memory = cache["memory"]
        level = self.feature_level
        full_height, full_width = cache["spatial_shapes"][level].unbind()
        start = cache["level_start_index"][level]
        level_size = full_height * full_width

        positions = torch.arange(
            memory.shape[1], device=memory.device, dtype=torch.long
        )
        level_offsets = positions - start
        in_level = (level_offsets >= 0) & (level_offsets < level_size)
        level_rows = torch.div(
            level_offsets, full_width, rounding_mode="floor"
        )
        level_columns = torch.remainder(level_offsets, full_width)
        valid = (~cache["padding_mask"]) & in_level.unsqueeze(0)
        # NestedTensor padding is bottom/right-only, so the largest valid
        # coordinate plus one is the valid height/width for each image.
        valid_heights = (
            torch.where(valid, level_rows.unsqueeze(0), -1).amax(dim=1) + 1
        )
        valid_widths = (
            torch.where(valid, level_columns.unsqueeze(0), -1).amax(dim=1) + 1
        )
        return start, full_width, valid_heights, valid_widths

    def select_aux_samples(self, targets: list[dict]) -> dict:
        memory = self.encoder_representation
        if len(targets) != memory.shape[0]:
            raise ValueError("Target count must match the encoder batch size")
        object_counts = [target["boxes"].shape[0] for target in targets]
        total = sum(object_counts)
        if total == 0:
            empty = memory.new_empty
            return {
                "features": empty((0, memory.shape[-1])),
                "references": empty((0, 2)),
                "targets": empty((0, 4)),
                "total": 0,
                "collisions": 0,
            }

        boxes = torch.cat([target["boxes"] for target in targets], dim=0)
        batch_indices = torch.repeat_interleave(
            torch.arange(len(targets), device=memory.device),
            torch.tensor(object_counts, device=memory.device),
            output_size=total,
        )
        start, full_width, valid_heights, valid_widths = self._level_layout()
        object_heights = valid_heights[batch_indices]
        object_widths = valid_widths[batch_indices]
        centers = boxes[:, :2].clamp(min=0.0, max=1.0 - 1e-7)
        columns = torch.floor(
            centers[:, 0] * object_widths.to(dtype=centers.dtype)
        ).long()
        rows = torch.floor(
            centers[:, 1] * object_heights.to(dtype=centers.dtype)
        ).long()
        columns = torch.minimum(columns.clamp_min(0), object_widths - 1)
        rows = torch.minimum(rows.clamp_min(0), object_heights - 1)

        flat_indices = start + rows * full_width + columns
        reference_dtype = boxes.dtype
        cell_centers = torch.stack(
            (
                (columns.to(reference_dtype) + 0.5)
                / object_widths.to(reference_dtype),
                (rows.to(reference_dtype) + 0.5)
                / object_heights.to(reference_dtype),
            ),
            dim=-1,
        )
        distances = (centers - cell_centers).square().sum(dim=-1)
        group_ids = batch_indices * memory.shape[1] + flat_indices
        kept_indices = _stable_group_argmin(group_ids, distances)
        kept_batches = batch_indices[kept_indices]
        kept_flat_indices = flat_indices[kept_indices]

        return {
            "features": memory[kept_batches, kept_flat_indices],
            "references": cell_centers[kept_indices].to(dtype=memory.dtype),
            "targets": boxes[kept_indices],
            "total": total,
            "collisions": total - kept_indices.numel(),
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
