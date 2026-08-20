import pytest

from projection_coco.config import TrainConfig
from projection_coco.experiment_matrix import (
    summarize_experiment_checkpoint,
    validate_experiment_matrix,
)


def make_summary(tmp_path, method: str) -> dict:
    config = TrainConfig(
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        method=method,
        target_global_batch_size=16,
        batch_recipe="coco_gb16_reference",
        amp=True,
        amp_dtype="float16",
    )
    return summarize_experiment_checkpoint(
        {
            "method": method,
            "config": config.as_dict(),
            "comparison_fingerprint": config.comparison_fingerprint,
            "initialization_fingerprint": "same-init",
            "upstream_commit": "same-upstream",
            "world_size": 2,
            "epoch": 50,
        }
    )


def test_matrix_accepts_controlled_comparison(tmp_path):
    summaries = [
        make_summary(tmp_path, method)
        for method in ("baseline", "aux_only", "projected")
    ]

    report = validate_experiment_matrix(summaries)

    assert report["status"] == "passed"
    assert report["target_global_batch_size"] == 16
    assert report["aux_weight"] == 2.5


def test_matrix_rejects_mixed_precision(tmp_path):
    summaries = [
        make_summary(tmp_path, method)
        for method in ("baseline", "aux_only", "projected")
    ]
    summaries[2]["comparison_fingerprint"] = "different"

    with pytest.raises(ValueError, match="comparison_fingerprint"):
        validate_experiment_matrix(summaries)


def test_matrix_rejects_aux_weight_mismatch(tmp_path):
    summaries = [
        make_summary(tmp_path, method)
        for method in ("baseline", "aux_only", "projected")
    ]
    summaries[2]["aux_weight"] = 3.0

    with pytest.raises(ValueError, match="aux_weight"):
        validate_experiment_matrix(summaries)


def test_matrix_rejects_duplicate_method(tmp_path):
    baseline = make_summary(tmp_path, "baseline")

    with pytest.raises(ValueError, match="Duplicate"):
        validate_experiment_matrix([baseline, baseline])


def test_matrix_rejects_incomplete_checkpoint(tmp_path):
    summaries = [
        make_summary(tmp_path, method)
        for method in ("baseline", "aux_only", "projected")
    ]
    summaries[1]["epoch"] = 49

    with pytest.raises(ValueError, match="incomplete"):
        validate_experiment_matrix(summaries)
