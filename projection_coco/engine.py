from __future__ import annotations

import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    write_csv,
)
from .config import TrainConfig, seed_everything
from .criterion import weighted_detection_loss
from .data import (
    DataBundle,
    make_train_loader,
    make_val_loader,
    set_train_loader_epoch,
)
from .distributed import DistributedContext, reduce_sums, runtime_metadata
from .evaluator import evaluate_coco
from .initialization import ensure_detector_initialization
from .methods import (
    make_research_model,
    register_representation_gradient_correction,
    representation_projected_gradients,
)
from .optimizer import make_optimizer


def _autocast_context(config: TrainConfig, device: torch.device):
    return torch.autocast(
        device_type="cuda",
        dtype=config.autocast_dtype,
        enabled=config.amp and device.type == "cuda",
    )


def _float_detection_outputs(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _float_detection_outputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_float_detection_outputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_float_detection_outputs(item) for item in value)
    return value


def _stats_are_finite(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    finite = [
        torch.isfinite(value.detach().float()).all()
        for value in stats.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]
    if not finite:
        raise RuntimeError("Projection statistics contain no floating-point values")
    return torch.stack(finite).all()


def _full_windows(loader, size: int):
    data_start = time.perf_counter()
    iterator = iter(loader)
    while True:
        window = list(islice(iterator, size))
        data_seconds = time.perf_counter() - data_start
        if len(window) != size:
            return
        yield window, data_seconds
        data_start = time.perf_counter()


def _move_targets(targets: list[dict], device: torch.device) -> list[dict]:
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in targets
    ]


def _empty_epoch_sums(device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "micro_batches",
        "optimizer_steps",
        "optimizer_steps_skipped",
        "total_loss",
        "detector_loss",
        "loss_ce",
        "loss_bbox",
        "loss_giou",
        "aux_loss",
        "aux_l1",
        "aux_giou",
        "aux_batches",
        "total_objects",
        "used_objects",
        "collision_targets",
        "raw_feature_norm",
        "adapted_feature_norm",
        "projection_steps",
        "projection_applied",
        "cls_aux_cosine_raw",
        "projection_removed_ratio",
    )
    return {
        key: torch.zeros((), device=device, dtype=torch.float64) for key in keys
    }


def _add_epoch_sum(
    sums: dict[str, torch.Tensor], key: str, value: torch.Tensor | float | int
) -> None:
    if isinstance(value, torch.Tensor):
        sums[key].add_(value.detach().to(dtype=torch.float64))
    else:
        sums[key].add_(value)


def _projection_stats_for_log(
    stats: dict[str, torch.Tensor],
) -> dict[str, float | bool]:
    keys = tuple(stats)
    values = torch.stack(
        [stats[key].detach().to(dtype=torch.float64) for key in keys]
    ).cpu().tolist()
    logged = {key: float(value) for key, value in zip(keys, values)}
    logged["projection_applied"] = bool(logged["projection_applied"])
    return logged


def _initial_history_row(config: TrainConfig, metrics: dict) -> dict:
    return {
        "method": config.method,
        "seed": config.seed,
        "batch_recipe": config.batch_recipe,
        "precision": config.precision,
        "batch_size_per_gpu": config.batch_size,
        "target_global_batch_size": config.target_global_batch_size,
        "recipe_fingerprint": config.recipe_fingerprint,
        "comparison_fingerprint": config.comparison_fingerprint,
        "projection_scope": config.projection_scope,
        "projection_reference_loss": config.projection_reference_loss,
        "epoch": 0,
        "total_loss": np.nan,
        "detector_loss": np.nan,
        "loss_ce": np.nan,
        "loss_bbox": np.nan,
        "loss_giou": np.nan,
        "aux_loss": np.nan,
        "aux_l1": np.nan,
        "aux_giou": np.nan,
        "aux_coverage": np.nan,
        "collision_rate": np.nan,
        "raw_feature_norm": np.nan,
        "adapted_feature_norm": np.nan,
        "projection_conflict_rate": np.nan,
        "cls_aux_cosine_raw_mean": np.nan,
        "projection_removed_ratio_mean": np.nan,
        "optimizer_steps": 0,
        "dropped_micro_batches": 0,
        "epoch_train_seconds": 0.0,
        "peak_cuda_mb": np.nan,
        **metrics,
    }


def train(
    config: TrainConfig,
    bundle: DataBundle,
    context: DistributedContext,
    *,
    resume_from: str | Path | None = None,
) -> None:
    existing_outputs = config.latest_checkpoint.is_file() or config.history_path.is_file()
    if resume_from is None and existing_outputs:
        raise FileExistsError(
            f"Run output already exists at {config.run_dir}; use --resume auto or "
            "choose a different --run-name"
        )
    if (
        str(resume_from).lower() == "auto"
        and not config.latest_checkpoint.is_file()
        and config.history_path.is_file()
    ):
        raise FileNotFoundError(
            f"History exists but latest checkpoint is missing at {config.run_dir}"
        )
    config.create_output_dirs()
    accumulation_steps = config.accumulation_steps(context.world_size)
    detector, criterion, postprocessors, initialization_fingerprint = (
        ensure_detector_initialization(config, context)
    )
    base_model = make_research_model(detector, config, context.device)
    optimizer, optimizer_summary = make_optimizer(base_model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_drop_epoch,
        gamma=config.lr_drop_gamma,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=config.uses_grad_scaler and context.device.type == "cuda"
    )
    start_epoch, elapsed_train, history, gradient_history, resume_path = (
        load_training_checkpoint(
            resume_from,
            base_model,
            optimizer,
            scheduler,
            scaler,
            config,
            context,
            initialization_fingerprint,
        )
    )

    training_model = base_model
    if context.distributed:
        training_model = DistributedDataParallel(
            base_model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    train_loader = make_train_loader(config, bundle, context)
    val_loader = make_val_loader(config, bundle, context)

    if context.is_main:
        print(config.to_json(context.world_size))
        print(f"[runtime] {runtime_metadata()}")
        print(
            f"[upstream] detector_init={initialization_fingerprint} "
            f"optimizer_groups={optimizer_summary.as_dict()}"
        )
        print(
            f"[data] train={len(bundle.train_dataset)} val={len(bundle.val_dataset)}"
        )
        if config.amp:
            print(
                f"[precision] {config.precision}; MSDeformAttn, detection losses, "
                "and auxiliary localization use FP32 safety boundaries"
            )
        if resume_path is not None:
            print(f"[resume] {resume_path} -> epoch {start_epoch}")
            if config.num_workers > 0:
                print(
                    "[warning] Persistent DataLoader worker RNG is not stored in "
                    "checkpoints; resumed random augmentations are not bit-exact."
                )

    if resume_path is None and not config.skip_initial_eval:
        metrics = evaluate_coco(
            base_model,
            postprocessors,
            val_loader,
            bundle.coco_api,
            context,
        )
        if context.is_main:
            history.append(_initial_history_row(config, metrics))
            write_csv(config.history_path, history)
        context.barrier()

    if start_epoch > config.epochs:
        if context.is_main:
            print(f"[done] checkpoint already reached epoch {start_epoch - 1}")
        base_model.close()
        return

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            seed_everything(
                config.seed + context.rank + epoch * 100_003,
                config.deterministic,
            )
            set_train_loader_epoch(train_loader, config, epoch)
            optimizer_steps_per_epoch = len(train_loader) // accumulation_steps
            dropped_micro_batches = len(train_loader) % accumulation_steps
            if optimizer_steps_per_epoch == 0:
                raise RuntimeError(
                    "Training loader is smaller than one complete accumulation window"
                )

            training_model.train()
            criterion.train()
            sums = _empty_epoch_sums(context.device)
            epoch_start = time.perf_counter()
            if context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(context.device)

            windows = _full_windows(train_loader, accumulation_steps)
            progress = tqdm(
                windows,
                total=optimizer_steps_per_epoch,
                desc=f"{config.method} e{epoch}",
                leave=False,
                disable=not context.is_main,
                mininterval=0.5,
            )
            micro_step = 0
            lr_used = optimizer.param_groups[0]["lr"]
            for optimizer_step, (window, data_seconds) in enumerate(progress):
                log_performance = (
                    context.is_main
                    and config.performance_log_every > 0
                    and optimizer_step % config.performance_log_every == 0
                )
                if log_performance and context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                step_start = time.perf_counter() if log_performance else 0.0
                window_finite = torch.ones(
                    (), dtype=torch.bool, device=context.device
                )
                local_num_boxes = sum(
                    len(target["labels"])
                    for _, targets in window
                    for target in targets
                )
                num_boxes = context.window_box_normalizer(local_num_boxes)
                optimizer.zero_grad(set_to_none=True)
                scale_before = scaler.get_scale()

                for micro_index, (samples, cpu_targets) in enumerate(window):
                    is_last_micro = micro_index == accumulation_steps - 1
                    sync_context = (
                        nullcontext()
                        if is_last_micro or not context.distributed
                        else training_model.no_sync()
                    )
                    samples = samples.to(context.device, non_blocking=True)
                    targets = _move_targets(cpu_targets, context.device)
                    projection_stats = None

                    with sync_context:
                        with _autocast_context(config, context.device):
                            result = training_model(
                                samples,
                                targets,
                                aux_normalizer=num_boxes,
                            )
                        # Matching, focal/L1/GIoU losses and their reductions are
                        # intentionally kept in FP32 for research stability.
                        with torch.autocast(device_type="cuda", enabled=False):
                            loss_dict = criterion(
                                _float_detection_outputs(
                                    result["detector_outputs"]
                                ),
                                targets,
                                num_boxes_override=num_boxes,
                            )
                            detector_loss = weighted_detection_loss(
                                loss_dict, criterion.weight_dict
                            )
                            total_loss = detector_loss
                            if config.uses_auxiliary:
                                if result["aux_loss"] is None:
                                    raise RuntimeError(
                                        "Auxiliary method skipped its loss"
                                    )
                                total_loss = (
                                    total_loss + config.aux_weight * result["aux_loss"]
                                )

                        window_finite.logical_and_(
                            torch.isfinite(total_loss.detach())
                        )

                        correction_hook = None
                        if config.uses_projection:
                            (
                                representation,
                                raw_auxiliary_gradient,
                                projected_auxiliary_gradient,
                                projection_stats,
                            ) = representation_projected_gradients(
                                base_model, loss_dict, result
                            )
                            window_finite.logical_and_(
                                _stats_are_finite(projection_stats)
                            )
                            correction_hook = (
                                register_representation_gradient_correction(
                                    representation,
                                    raw_auxiliary_gradient,
                                    projected_auxiliary_gradient,
                                    auxiliary_weight=config.aux_weight,
                                    grad_scale=scaler.get_scale(),
                                )
                            )
                        try:
                            scaler.scale(total_loss).backward()
                        finally:
                            if correction_hook is not None:
                                correction_hook.remove()

                    with torch.no_grad():
                        _add_epoch_sum(sums, "micro_batches", 1)
                        _add_epoch_sum(sums, "total_loss", total_loss)
                        _add_epoch_sum(sums, "detector_loss", detector_loss)
                        _add_epoch_sum(sums, "loss_ce", loss_dict["loss_ce"])
                        _add_epoch_sum(sums, "loss_bbox", loss_dict["loss_bbox"])
                        _add_epoch_sum(sums, "loss_giou", loss_dict["loss_giou"])
                        if config.uses_auxiliary:
                            _add_epoch_sum(sums, "aux_batches", 1)
                            _add_epoch_sum(sums, "aux_loss", result["aux_loss"])
                            _add_epoch_sum(sums, "aux_l1", result["aux_l1"])
                            _add_epoch_sum(sums, "aux_giou", result["aux_giou"])
                            _add_epoch_sum(
                                sums, "total_objects", result["aux_total"]
                            )
                            _add_epoch_sum(
                                sums, "used_objects", result["aux_used"]
                            )
                            _add_epoch_sum(
                                sums,
                                "collision_targets",
                                result["aux_collisions"],
                            )
                            for key, value in result["feature_stats"].items():
                                _add_epoch_sum(sums, key, value)
                        if config.uses_projection:
                            if projection_stats is None:
                                raise RuntimeError("Missing projection statistics")
                            _add_epoch_sum(sums, "projection_steps", 1)
                            _add_epoch_sum(
                                sums,
                                "projection_applied",
                                projection_stats["projection_applied"],
                            )
                            _add_epoch_sum(
                                sums,
                                "cls_aux_cosine_raw",
                                projection_stats["cls_aux_cosine_raw"],
                            )
                            _add_epoch_sum(
                                sums,
                                "projection_removed_ratio",
                                projection_stats["projection_removed_ratio"],
                            )
                    if config.uses_projection:
                        if (
                            context.is_main
                            and micro_step % config.gradient_log_every == 0
                        ):
                            gradient_history.append(
                                {
                                    "method": config.method,
                                    "seed": config.seed,
                                    "epoch": epoch,
                                    "optimizer_step": optimizer_step,
                                    "micro_step": micro_step,
                                    "rank": context.rank,
                                    "batch_recipe": config.batch_recipe,
                                    "precision": config.precision,
                                    "recipe_fingerprint": config.recipe_fingerprint,
                                    "projection_scope": config.projection_scope,
                                    "projection_reference_loss": (
                                        config.projection_reference_loss
                                    ),
                                    **_projection_stats_for_log(projection_stats),
                                    "aux_weight": config.aux_weight,
                                    "grad_scale": scaler.get_scale(),
                                }
                            )
                    micro_step += 1

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(), config.grad_clip
                )
                window_finite.logical_and_(torch.isfinite(grad_norm.detach()))
                if not context.all_true(window_finite):
                    raise FloatingPointError(
                        f"Non-finite loss, projection statistic, or gradient at epoch={epoch} "
                        f"step={optimizer_step}"
                    )
                scaler.step(optimizer)
                scaler.update()
                skipped = bool(
                    config.uses_grad_scaler and scaler.get_scale() < scale_before
                )
                with torch.no_grad():
                    _add_epoch_sum(sums, "optimizer_steps", 1)
                    _add_epoch_sum(sums, "optimizer_steps_skipped", int(skipped))
                if log_performance:
                    if context.device.type == "cuda":
                        torch.cuda.synchronize(context.device)
                    step_seconds = time.perf_counter() - step_start
                    print(
                        f"[perf] epoch={epoch} step={optimizer_step + 1} "
                        f"data_wait={data_seconds:.3f}s "
                        f"compute={step_seconds:.3f}s"
                    )

            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            epoch_seconds = time.perf_counter() - epoch_start
            elapsed_train += epoch_seconds
            global_sums = reduce_sums(sums, context)
            scheduler.step()

            metrics = {
                "map": np.nan,
                "map50": np.nan,
                "map75": np.nan,
                "map_small": np.nan,
                "map_medium": np.nan,
                "map_large": np.nan,
                "mar100": np.nan,
                "val_seconds": np.nan,
            }
            if epoch % config.eval_every == 0 or epoch == config.epochs:
                metrics = evaluate_coco(
                    base_model,
                    postprocessors,
                    val_loader,
                    bundle.coco_api,
                    context,
                )

            if context.is_main:
                global_steps = max(global_sums["optimizer_steps"], 1.0)
                auxiliary_batches = max(global_sums["aux_batches"], 1.0)
                projection_steps = max(global_sums["projection_steps"], 1.0)
                row = {
                    "method": config.method,
                    "seed": config.seed,
                    "batch_recipe": config.batch_recipe,
                    "precision": config.precision,
                    "batch_size_per_gpu": config.batch_size,
                    "target_global_batch_size": config.target_global_batch_size,
                    "recipe_fingerprint": config.recipe_fingerprint,
                    "comparison_fingerprint": config.comparison_fingerprint,
                    "projection_scope": config.projection_scope,
                    "projection_reference_loss": config.projection_reference_loss,
                    "epoch": epoch,
                    "lr": lr_used,
                    "total_loss": global_sums["total_loss"] / global_steps,
                    "detector_loss": global_sums["detector_loss"] / global_steps,
                    "loss_ce": global_sums["loss_ce"] / global_steps,
                    "loss_bbox": global_sums["loss_bbox"] / global_steps,
                    "loss_giou": global_sums["loss_giou"] / global_steps,
                    "aux_loss": (
                        global_sums["aux_loss"] / global_steps
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "aux_l1": (
                        global_sums["aux_l1"] / global_steps
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "aux_giou": (
                        global_sums["aux_giou"] / global_steps
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "aux_coverage": (
                        global_sums["used_objects"]
                        / max(global_sums["total_objects"], 1.0)
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "collision_rate": (
                        global_sums["collision_targets"]
                        / max(global_sums["total_objects"], 1.0)
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "raw_feature_norm": (
                        global_sums["raw_feature_norm"] / auxiliary_batches
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "adapted_feature_norm": (
                        global_sums["adapted_feature_norm"] / auxiliary_batches
                        if config.uses_auxiliary
                        else np.nan
                    ),
                    "projection_conflict_rate": (
                        global_sums["projection_applied"] / projection_steps
                        if config.uses_projection
                        else np.nan
                    ),
                    "cls_aux_cosine_raw_mean": (
                        global_sums["cls_aux_cosine_raw"] / projection_steps
                        if config.uses_projection
                        else np.nan
                    ),
                    "projection_removed_ratio_mean": (
                        global_sums["projection_removed_ratio"] / projection_steps
                        if config.uses_projection
                        else np.nan
                    ),
                    "optimizer_steps": int(global_steps / context.world_size),
                    "optimizer_step_skip_rate": (
                        global_sums["optimizer_steps_skipped"] / global_steps
                    ),
                    "dropped_micro_batches": dropped_micro_batches,
                    "epoch_train_seconds": epoch_seconds,
                    "peak_cuda_mb": (
                        torch.cuda.max_memory_allocated(context.device) / 2**20
                        if context.device.type == "cuda"
                        else np.nan
                    ),
                    **metrics,
                }
                history.append(row)
                write_csv(config.history_path, history)
                write_csv(config.gradients_path, gradient_history)
                print(
                    f"[epoch {epoch:03d}] total={row['total_loss']:.4f} "
                    f"det={row['detector_loss']:.4f} "
                    f"aux={row['aux_loss']:.4f} map={row['map']:.4f}"
                )

            save_training_checkpoint(
                base_model,
                optimizer,
                scheduler,
                scaler,
                config,
                context,
                initialization_fingerprint=initialization_fingerprint,
                epoch=epoch,
                elapsed_train=elapsed_train,
                history=history if context.is_main else [],
                gradients=gradient_history if context.is_main else [],
            )
    finally:
        base_model.close()
