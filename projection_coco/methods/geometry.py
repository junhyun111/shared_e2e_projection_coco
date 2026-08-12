from __future__ import annotations

import torch


def inverse_sigmoid(tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    tensor = tensor.clamp(0, 1)
    return torch.log(tensor.clamp(min=eps) / (1 - tensor).clamp(min=eps))


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1, area2 = box_area(boxes1), box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = (right_bottom - left_top).clamp(min=0)
    intersection = size[..., 0] * size[..., 1]
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp(min=1e-7)

    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_right_bottom = torch.maximum(
        boxes1[:, None, 2:], boxes2[None, :, 2:]
    )
    enclosing_size = (enclosing_right_bottom - enclosing_left_top).clamp(min=0)
    enclosing = (enclosing_size[..., 0] * enclosing_size[..., 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing

