from __future__ import annotations

import torch


def project_conflicting_gradient(
    classification_gradient: torch.Tensor,
    auxiliary_gradient: torch.Tensor,
    epsilon: float = 1e-12,
):
    classification_float = classification_gradient.detach().float()
    auxiliary_float = auxiliary_gradient.detach().float()
    dot = (classification_float * auxiliary_float).sum()
    classification_norm_squared = classification_float.square().sum()
    classification_norm = classification_norm_squared.sqrt()
    auxiliary_norm = auxiliary_float.square().sum().sqrt()
    cosine = dot / (classification_norm * auxiliary_norm + epsilon)
    applied = dot < 0
    coefficient = dot / (classification_norm_squared + epsilon)
    conflicting_projection = auxiliary_float - coefficient * classification_float
    projected_float = torch.where(
        applied, conflicting_projection, auxiliary_float
    )
    projected = projected_float.to(dtype=auxiliary_gradient.dtype)
    projected_float = projected.detach().float()
    projected_norm = projected_float.square().sum().sqrt()
    projected_dot = (classification_float * projected_float).sum()
    removed_ratio = torch.where(
        auxiliary_norm <= epsilon,
        torch.zeros_like(auxiliary_norm),
        1.0 - projected_norm / auxiliary_norm.clamp_min(epsilon),
    )
    return projected, {
        "cls_aux_cosine_raw": cosine,
        "cls_aux_dot_raw": dot,
        "cls_aux_dot_projected": projected_dot,
        "cls_grad_norm": classification_norm,
        "aux_grad_norm": auxiliary_norm,
        "aux_grad_norm_projected": projected_norm,
        "projection_applied": applied,
        "projection_removed_ratio": removed_ratio,
    }


def representation_projected_gradients(model, loss_dict: dict, result: dict):
    if "loss_ce" not in loss_dict:
        raise KeyError("Projection requires final-decoder loss_ce")
    if result["aux_loss"] is None:
        raise RuntimeError("Projection requires the auxiliary branch")
    representation = model.encoder_representation
    classification_gradient = torch.autograd.grad(
        loss_dict["loss_ce"], representation, retain_graph=True, allow_unused=False
    )[0]
    auxiliary_gradient = torch.autograd.grad(
        result["aux_loss"], representation, retain_graph=True, allow_unused=False
    )[0]
    projected_gradient, stats = project_conflicting_gradient(
        classification_gradient, auxiliary_gradient
    )
    return representation, auxiliary_gradient, projected_gradient, stats


def register_representation_gradient_correction(
    representation: torch.Tensor,
    raw_auxiliary_gradient: torch.Tensor,
    projected_auxiliary_gradient: torch.Tensor,
    *,
    auxiliary_weight: float,
    grad_scale: float,
):
    correction = (
        (
            projected_auxiliary_gradient.detach().float()
            - raw_auxiliary_gradient.detach().float()
        )
        * float(auxiliary_weight)
        * float(grad_scale)
    )

    def add_correction(total_gradient):
        return total_gradient + correction.to(dtype=total_gradient.dtype)

    return representation.register_hook(add_correction)
