from pathlib import Path

import pytest

import train
from projection_coco.config import DEFAULT_AUX_WEIGHT, TrainConfig


def test_cli_uses_canonical_aux_weight_when_environment_is_empty(monkeypatch):
    monkeypatch.setenv("AUX_WEIGHT", "")

    args = train.build_parser().parse_args([])

    assert args.aux_weight == DEFAULT_AUX_WEIGHT == 2.5


def test_cli_reads_explicit_aux_weight_from_environment(monkeypatch):
    monkeypatch.setenv("AUX_WEIGHT", "3.0")

    args = train.build_parser().parse_args([])

    assert args.aux_weight == 3.0


def test_cli_resolves_gb16_reference(monkeypatch):
    monkeypatch.delenv("BATCH_RECIPE", raising=False)
    parser = train.build_parser()
    args = parser.parse_args(["--batch-recipe", "coco_gb16_reference"])

    assert train._resolve_batch_recipe(args, parser) == (
        "coco_gb16_reference",
        16,
    )


def test_cli_rejects_named_recipe_batch_mismatch(monkeypatch):
    monkeypatch.delenv("BATCH_RECIPE", raising=False)
    parser = train.build_parser()
    args = parser.parse_args(
        [
            "--batch-recipe",
            "coco_gb16_reference",
            "--target-global-batch-size",
            "32",
        ]
    )

    with pytest.raises(SystemExit):
        train._resolve_batch_recipe(args, parser)


def test_automatic_run_name_contains_recipe_fingerprint(tmp_path: Path):
    config = TrainConfig(
        data_root=tmp_path / "coco",
        output_root=tmp_path / "artifacts",
        method="projected",
        amp=True,
        aux_weight=2.5,
    )

    run_name = train._automatic_run_name(config)

    assert run_name.startswith("gb32_optimized_fp16_b4_")
    assert run_name.endswith(config.recipe_fingerprint[:8])
