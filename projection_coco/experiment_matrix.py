from __future__ import annotations

from pathlib import Path

import torch

from .config import TrainConfig


PAPER_METHODS = ("baseline", "aux_only", "projected")


def summarize_experiment_checkpoint(
    checkpoint: dict, *, checkpoint_path: str | Path | None = None
) -> dict:
    config_values = checkpoint.get("config")
    if not isinstance(config_values, dict):
        raise ValueError("Checkpoint does not contain its training configuration")
    config = TrainConfig.from_dict(config_values)
    method = checkpoint.get("method")
    if method != config.method:
        raise ValueError("Checkpoint method and saved configuration disagree")

    comparison_fingerprint = config.comparison_fingerprint
    stored_comparison = checkpoint.get("comparison_fingerprint")
    if (
        stored_comparison is not None
        and stored_comparison != comparison_fingerprint
    ):
        raise ValueError("Checkpoint comparison fingerprint is inconsistent")

    summary = {
        "method": config.method,
        "checkpoint": (
            str(Path(checkpoint_path).expanduser().resolve())
            if checkpoint_path is not None
            else None
        ),
        "epoch": int(checkpoint.get("epoch", 0)),
        "configured_epochs": config.epochs,
        "precision": config.precision,
        "batch_recipe": config.batch_recipe,
        "batch_size_per_gpu": config.batch_size,
        "target_global_batch_size": config.target_global_batch_size,
        "world_size": int(checkpoint.get("world_size", 1)),
        "aux_weight": config.aux_weight if config.uses_auxiliary else None,
        "feature_level": config.feature_level if config.uses_auxiliary else None,
        "recipe_fingerprint": config.recipe_fingerprint,
        "comparison_fingerprint": comparison_fingerprint,
        "initialization_fingerprint": checkpoint.get(
            "initialization_fingerprint"
        ),
        "upstream_commit": checkpoint.get("upstream_commit"),
    }
    return summary


def validate_experiment_matrix(summaries: list[dict]) -> dict:
    by_method: dict[str, dict] = {}
    for summary in summaries:
        method = summary.get("method")
        if method in by_method:
            raise ValueError(f"Duplicate checkpoint for method={method}")
        by_method[method] = summary

    if set(by_method) != set(PAPER_METHODS):
        missing = sorted(set(PAPER_METHODS) - set(by_method))
        extra = sorted(set(by_method) - set(PAPER_METHODS))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise ValueError(
            "Experiment matrix must contain baseline, aux_only, and projected "
            f"exactly once ({'; '.join(details)})"
        )

    comparison_fields = (
        "comparison_fingerprint",
        "initialization_fingerprint",
        "upstream_commit",
        "world_size",
    )
    reference = by_method["baseline"]
    for method, summary in by_method.items():
        if summary.get("epoch") != summary.get("configured_epochs"):
            raise ValueError(
                f"Checkpoint for method={method} is incomplete: "
                f"epoch={summary.get('epoch')} of "
                f"{summary.get('configured_epochs')}"
            )
    for field in comparison_fields:
        values = {method: row.get(field) for method, row in by_method.items()}
        if len(set(values.values())) != 1:
            formatted = ", ".join(
                f"{method}={value}" for method, value in values.items()
            )
            raise ValueError(f"Experiment matrix differs in {field}: {formatted}")
        if reference.get(field) is None:
            raise ValueError(f"Experiment matrix is missing {field}")

    aux_only = by_method["aux_only"]
    projected = by_method["projected"]
    for field in ("aux_weight", "feature_level"):
        if aux_only.get(field) != projected.get(field):
            raise ValueError(
                f"aux_only and projected differ in {field}: "
                f"{aux_only.get(field)} != {projected.get(field)}"
            )

    return {
        "status": "passed",
        "methods": list(PAPER_METHODS),
        "precision": reference["precision"],
        "batch_recipe": reference["batch_recipe"],
        "batch_size_per_gpu": reference["batch_size_per_gpu"],
        "target_global_batch_size": reference["target_global_batch_size"],
        "world_size": reference["world_size"],
        "epochs": reference["epoch"],
        "comparison_fingerprint": reference["comparison_fingerprint"],
        "initialization_fingerprint": reference[
            "initialization_fingerprint"
        ],
        "upstream_commit": reference["upstream_commit"],
        "aux_weight": aux_only["aux_weight"],
        "feature_level": aux_only["feature_level"],
        "checkpoints": {
            method: by_method[method].get("checkpoint")
            for method in PAPER_METHODS
        },
    }


def load_experiment_checkpoint_summary(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return summarize_experiment_checkpoint(
        checkpoint, checkpoint_path=path
    )


def load_and_validate_experiment_matrix(paths: list[str | Path]) -> dict:
    summaries = [load_experiment_checkpoint_summary(path) for path in paths]
    return validate_experiment_matrix(summaries)
