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
    assert config.accumulation_steps(world_size=2) == 4


def test_accumulation_requires_exact_divisibility(tmp_path):
    config = make_config(tmp_path, batch_size=3, target_global_batch_size=32)
    with pytest.raises(ValueError, match="not divisible"):
        config.accumulation_steps(world_size=2)


def test_performance_logging_is_diagnostic_not_recipe(tmp_path):
    regular = make_config(tmp_path)
    profiled = make_config(tmp_path, performance_log_every=100)

    assert regular.recipe_fingerprint == profiled.recipe_fingerprint


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
