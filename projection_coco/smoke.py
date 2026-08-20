from __future__ import annotations

import math
from pathlib import Path

import torch


HISTORY_FINITE_KEYS = (
    "total_loss",
    "detector_loss",
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "aux_loss",
    "aux_l1",
    "aux_giou",
    "projection_conflict_rate",
    "cls_aux_cosine_raw_mean",
    "projection_removed_ratio_mean",
    "peak_cuda_mb",
)

GRADIENT_FINITE_KEYS = (
    "cls_aux_cosine_raw",
    "cls_aux_dot_raw",
    "cls_aux_dot_projected",
    "cls_grad_norm",
    "aux_grad_norm",
    "aux_grad_norm_projected",
    "projection_removed_ratio",
    "grad_scale",
)


def _require_finite(mapping: dict, keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if key not in mapping:
            raise ValueError(f"{label} is missing {key}")
        if not math.isfinite(float(mapping[key])):
            raise ValueError(f"{label}.{key} is not finite: {mapping[key]}")


def validate_projected_amp_smoke_checkpoint(
    checkpoint: dict,
    *,
    expected_world_size: int = 2,
) -> dict:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain its training configuration")
    if checkpoint.get("method") != "projected" or config.get("method") != "projected":
        raise ValueError("Smoke checkpoint must use method=projected")
    if not config.get("amp") or config.get("amp_dtype") != "float16":
        raise ValueError("Smoke checkpoint must use FP16 AMP")
    if int(checkpoint.get("world_size", 0)) != expected_world_size:
        raise ValueError(
            f"Smoke checkpoint must use world_size={expected_world_size}"
        )
    if config.get("projection_scope") is not None:
        raise ValueError("projection_scope must be stored as method metadata, not config")

    definition = checkpoint.get("method_definition")
    if not isinstance(definition, dict):
        raise ValueError("Checkpoint is missing method_definition")
    if definition.get("projection_scope") != (
        "per_rank_micro_batch_encoder_representation"
    ):
        raise ValueError("Unexpected projection scope")
    if definition.get("projection_reference_loss") != "final_decoder_loss_ce":
        raise ValueError("Unexpected projection reference loss")

    history = checkpoint.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Smoke checkpoint has no history")
    final_row = history[-1]
    _require_finite(final_row, HISTORY_FINITE_KEYS, "history")
    if int(final_row.get("optimizer_steps", 0)) <= 0:
        raise ValueError("Smoke run performed no optimizer steps")
    skip_rate = float(final_row.get("optimizer_step_skip_rate", 1.0))
    if skip_rate >= 1.0:
        raise ValueError("All optimizer steps were skipped")

    gradients = checkpoint.get("gradients")
    if not isinstance(gradients, list) or not gradients:
        raise ValueError("Smoke checkpoint has no projection gradient records")
    for index, row in enumerate(gradients):
        _require_finite(row, GRADIENT_FINITE_KEYS, f"gradients[{index}]")

    scaler_state = checkpoint.get("scaler_state_dict")
    if not isinstance(scaler_state, dict) or not scaler_state:
        raise ValueError("FP16 smoke checkpoint has no GradScaler state")

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("Smoke checkpoint has no model state")
    nonfinite_parameters = [
        name
        for name, tensor in model_state.items()
        if isinstance(tensor, torch.Tensor)
        and (tensor.is_floating_point() or tensor.is_complex())
        and not bool(torch.isfinite(tensor).all())
    ]
    if nonfinite_parameters:
        raise ValueError(
            "Model contains non-finite parameters: "
            + ", ".join(nonfinite_parameters[:5])
        )

    runtime = checkpoint.get("runtime", {})
    devices = runtime.get("devices", []) if isinstance(runtime, dict) else []
    if len(devices) < expected_world_size:
        raise ValueError("Runtime metadata does not list both visible GPUs")

    return {
        "status": "passed",
        "method": "projected",
        "precision": "fp16",
        "world_size": expected_world_size,
        "batch_recipe": config.get("batch_recipe"),
        "batch_size_per_gpu": config.get("batch_size"),
        "target_global_batch_size": config.get("target_global_batch_size"),
        "aux_weight": config.get("aux_weight"),
        "optimizer_steps": int(final_row["optimizer_steps"]),
        "optimizer_step_skip_rate": float(
            final_row["optimizer_step_skip_rate"]
        ),
        "projection_records": len(gradients),
        "peak_cuda_mb": float(final_row["peak_cuda_mb"]),
        "recipe_fingerprint": checkpoint.get("recipe_fingerprint"),
    }


def load_and_validate_projected_amp_smoke(
    path: str | Path, *, expected_world_size: int = 2
) -> dict:
    path = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return validate_projected_amp_smoke_checkpoint(
        checkpoint, expected_world_size=expected_world_size
    )
