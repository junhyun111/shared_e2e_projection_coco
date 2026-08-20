from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .distributed import DistributedContext, runtime_metadata
from .upstream import upstream_commit


def _atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_resume_path(resume_from: str | Path | None, config: TrainConfig):
    if resume_from is None:
        return None
    if str(resume_from).lower() == "auto":
        return config.latest_checkpoint if config.latest_checkpoint.is_file() else None
    return Path(resume_from).expanduser().resolve()


def _checkpoint_recipe_dict(checkpoint: dict, config: TrainConfig) -> dict | None:
    stored_recipe = checkpoint.get("recipe")
    if isinstance(stored_recipe, dict):
        return stored_recipe
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        return None
    try:
        restored = TrainConfig.from_dict(
            saved_config,
            data_root=config.data_root,
            output_root=config.output_root,
            run_name=None,
        )
    except (TypeError, ValueError):
        return None
    return restored.recipe_dict()


def _recipe_differences(saved: dict, expected: dict) -> list[str]:
    differences = []
    for key in sorted(set(saved) | set(expected)):
        if saved.get(key) != expected.get(key):
            differences.append(key)
    return differences


def load_training_checkpoint(
    resume_from: str | Path | None,
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    context: DistributedContext,
    initialization_fingerprint: str,
):
    path = resolve_resume_path(resume_from, config)
    if path is None:
        return 1, 0.0, [], [], None
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("method") != config.method:
        raise ValueError("Checkpoint method does not match this run")
    saved_recipe = _checkpoint_recipe_dict(checkpoint, config)
    if saved_recipe is not None:
        differences = _recipe_differences(saved_recipe, config.recipe_dict())
        if differences:
            raise ValueError(
                "Checkpoint training recipe does not match this run; differing "
                f"fields: {', '.join(differences)}"
            )
    elif checkpoint.get("recipe_fingerprint") != config.recipe_fingerprint:
        raise ValueError("Checkpoint training recipe does not match this run")
    if checkpoint.get("upstream_commit") != upstream_commit():
        raise ValueError("Checkpoint upstream commit does not match this run")
    if checkpoint.get("initialization_fingerprint") != initialization_fingerprint:
        raise ValueError("Checkpoint detector initialization does not match this run")
    if int(checkpoint.get("world_size", 1)) != context.world_size:
        raise ValueError("Resume requires the same world size")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("elapsed_train", 0.0)),
        list(checkpoint.get("history", [])),
        list(checkpoint.get("gradients", [])),
        path,
    )


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    context: DistributedContext,
    *,
    initialization_fingerprint: str,
    epoch: int,
    elapsed_train: float,
    history: list[dict],
    gradients: list[dict],
) -> None:
    context.barrier()
    if context.is_main:
        state = {
            "format_version": 3,
            "recipe_schema_version": 2,
            "method": config.method,
            "upstream_commit": upstream_commit(),
            "runtime": runtime_metadata(),
            "recipe_fingerprint": config.recipe_fingerprint,
            "recipe": config.recipe_dict(),
            "comparison_fingerprint": config.comparison_fingerprint,
            "method_definition": config.method_definition,
            "initialization_fingerprint": initialization_fingerprint,
            "config": config.as_dict(),
            "world_size": context.world_size,
            "accumulation_steps": config.accumulation_steps(context.world_size),
            "epoch": epoch,
            "elapsed_train": elapsed_train,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": _rng_state(),
            "history": history,
            "gradients": gradients,
        }
        _atomic_save(state, config.latest_checkpoint)
        if epoch % config.save_every == 0 or epoch == config.epochs:
            _atomic_save(
                state, config.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            )
    context.barrier()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
