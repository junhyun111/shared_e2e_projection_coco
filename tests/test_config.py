from pathlib import Path

import pytest

from projection_coco.config import TrainConfig


def make_config(tmp_path: Path, **overrides) -> TrainConfig:
    values = {
        "data_root": tmp_path / "coco",
        "output_root": tmp_path / "artifacts",
    }
    values.update(overrides)
    return TrainConfig(**values)


def test_accumulation_steps_resolve_reported_batch(tmp_path):
    config = make_config(tmp_path, batch_size=2, target_global_batch_size=32)
    assert config.accumulation_steps(world_size=2) == 8


def test_default_batch_matches_two_gpu_training_recipe(tmp_path):
    config = make_config(tmp_path)
    assert config.batch_size == 4
    assert config.batch_recipe == "coco_gb32_optimized"
    assert config.accumulation_steps(world_size=2) == 4


def test_gb16_reference_uses_two_accumulation_steps_on_two_gpus(tmp_path):
    config = make_config(
        tmp_path,
        batch_recipe="coco_gb16_reference",
        target_global_batch_size=16,
    )

    assert config.accumulation_steps(world_size=2) == 2


def test_named_batch_recipe_rejects_mismatched_global_batch(tmp_path):
    with pytest.raises(ValueError, match="requires target_global_batch_size=16"):
        make_config(
            tmp_path,
            batch_recipe="coco_gb16_reference",
            target_global_batch_size=32,
        )


def test_accumulation_requires_exact_divisibility(tmp_path):
    config = make_config(tmp_path, batch_size=3, target_global_batch_size=32)
    with pytest.raises(ValueError, match="not divisible"):
        config.accumulation_steps(world_size=2)


def test_performance_logging_is_diagnostic_not_recipe(tmp_path):
    regular = make_config(tmp_path)
    profiled = make_config(tmp_path, performance_log_every=100)

    assert regular.recipe_fingerprint == profiled.recipe_fingerprint


def test_inactive_amp_dtype_does_not_change_fp32_recipe(tmp_path):
    fp16_setting = make_config(tmp_path, amp=False, amp_dtype="float16")
    bf16_setting = make_config(tmp_path, amp=False, amp_dtype="bfloat16")

    assert fp16_setting.precision == "float32"
    assert fp16_setting.recipe_fingerprint == bf16_setting.recipe_fingerprint


def test_amp_dtype_changes_mixed_precision_recipe(tmp_path):
    fp16 = make_config(tmp_path, amp=True, amp_dtype="float16")
    bf16 = make_config(tmp_path, amp=True, amp_dtype="bfloat16")

    assert fp16.precision == "float16"
    assert fp16.uses_grad_scaler is True
    assert bf16.precision == "bfloat16"
    assert bf16.uses_grad_scaler is False
    assert fp16.recipe_fingerprint != bf16.recipe_fingerprint


def test_run_name_isolates_outputs_without_changing_recipe(tmp_path):
    regular = make_config(tmp_path)
    isolated = make_config(tmp_path, run_name="fp16_batch4_global32")

    assert isolated.run_dir == regular.run_dir / "fp16_batch4_global32"
    assert isolated.recipe_fingerprint == regular.recipe_fingerprint


def test_baseline_recipe_ignores_inactive_auxiliary_settings(tmp_path):
    first = make_config(tmp_path, method="baseline", aux_weight=2.0, feature_level=0)
    second = make_config(tmp_path, method="baseline", aux_weight=2.5, feature_level=1)

    assert first.recipe_fingerprint == second.recipe_fingerprint


def test_auxiliary_recipe_includes_aux_weight(tmp_path):
    first = make_config(tmp_path, method="projected", aux_weight=2.0)
    second = make_config(tmp_path, method="projected", aux_weight=2.5)

    assert first.recipe_fingerprint != second.recipe_fingerprint


def test_comparison_fingerprint_matches_across_methods(tmp_path):
    baseline = make_config(tmp_path, method="baseline")
    auxiliary = make_config(tmp_path, method="aux_only")
    projected = make_config(tmp_path, method="projected")

    assert baseline.comparison_fingerprint == auxiliary.comparison_fingerprint
    assert auxiliary.comparison_fingerprint == projected.comparison_fingerprint


def test_old_config_infers_named_batch_recipe(tmp_path):
    values = make_config(tmp_path).as_dict()
    values.pop("batch_recipe")

    restored = TrainConfig.from_dict(values)

    assert restored.batch_recipe == "coco_gb32_optimized"


def test_projection_definition_is_explicit(tmp_path):
    config = make_config(tmp_path, method="projected")

    assert config.projection_scope == "per_rank_micro_batch_encoder_representation"
    assert config.projection_reference_loss == "final_decoder_loss_ce"


@pytest.mark.parametrize("run_name", ["../escape", "has space", "", "-leading"])
def test_run_name_rejects_unsafe_paths(tmp_path, run_name):
    with pytest.raises(ValueError, match="run_name"):
        make_config(tmp_path, run_name=run_name)


def test_amp_dtype_is_validated(tmp_path):
    with pytest.raises(ValueError, match="amp_dtype"):
        make_config(tmp_path, amp=True, amp_dtype="float8")


@pytest.mark.parametrize(
    ("method", "uses_auxiliary", "uses_adapter", "uses_projection"),
    [
        ("baseline", False, False, False),
        ("aux_no_adapter", True, False, False),
        ("aux_only", True, True, False),
        ("projected", True, True, True),
    ],
)
def test_method_components(
    tmp_path, method, uses_auxiliary, uses_adapter, uses_projection
):
    config = make_config(tmp_path, method=method)
    assert config.uses_auxiliary is uses_auxiliary
    assert config.uses_adapter is uses_adapter
    assert config.uses_projection is uses_projection


def test_methods_share_detector_recipe(tmp_path):
    baseline = make_config(tmp_path, method="baseline")
    projected = make_config(tmp_path, method="projected")
    assert baseline.detector_recipe_fingerprint == projected.detector_recipe_fingerprint
    assert baseline.recipe_fingerprint != projected.recipe_fingerprint
