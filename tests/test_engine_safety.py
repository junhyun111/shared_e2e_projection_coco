from pathlib import Path

import pytest
import torch

from projection_coco.config import TrainConfig
from projection_coco.engine import _stats_are_finite, train


def make_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "artifacts",
        run_name="isolated",
    )


def test_projection_stat_finite_check_rejects_nan():
    stats = {
        "finite": torch.tensor(1.0),
        "nonfinite": torch.tensor(float("nan")),
        "flag": torch.tensor(True),
    }

    assert not bool(_stats_are_finite(stats))


def test_new_run_refuses_to_overwrite_existing_history(tmp_path):
    config = make_config(tmp_path)
    config.history_path.parent.mkdir(parents=True)
    config.history_path.write_text("epoch\n1\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        train(config, bundle=None, context=None, resume_from=None)


def test_auto_resume_refuses_orphaned_history(tmp_path):
    config = make_config(tmp_path)
    config.history_path.parent.mkdir(parents=True)
    config.history_path.write_text("epoch\n1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        train(config, bundle=None, context=None, resume_from="auto")
