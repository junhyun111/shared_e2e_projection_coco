from pathlib import Path

from projection_coco.checkpoint import (
    _checkpoint_recipe_dict,
    _recipe_differences,
)
from projection_coco.config import TrainConfig


def make_config(tmp_path: Path, **overrides) -> TrainConfig:
    values = {
        "data_root": tmp_path / "coco",
        "output_root": tmp_path / "artifacts",
    }
    values.update(overrides)
    return TrainConfig(**values)


def test_legacy_baseline_aux_weight_is_normalized_as_inactive(tmp_path):
    current = make_config(tmp_path, method="baseline", aux_weight=2.5)
    legacy_values = current.as_dict()
    legacy_values.pop("batch_recipe")
    legacy_values["aux_weight"] = 2.0
    checkpoint = {"config": legacy_values}

    restored = _checkpoint_recipe_dict(checkpoint, current)

    assert restored == current.recipe_dict()


def test_legacy_projected_aux_weight_difference_is_preserved(tmp_path):
    current = make_config(tmp_path, method="projected", aux_weight=2.5)
    legacy_values = current.as_dict()
    legacy_values.pop("batch_recipe")
    legacy_values["aux_weight"] = 2.0
    checkpoint = {"config": legacy_values}

    restored = _checkpoint_recipe_dict(checkpoint, current)

    assert "aux_weight" in _recipe_differences(restored, current.recipe_dict())
