import json
from pathlib import Path

from PIL import Image

from projection_coco.evaluation_report import (
    OFFICIAL_REFERENCE,
    write_official_evaluation_report,
)


def test_official_report_writes_json_and_png(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"trusted-test-checkpoint")
    metrics = {**OFFICIAL_REFERENCE, "val_seconds": 12.5}

    json_path, png_path, report = write_official_evaluation_report(
        metrics,
        tmp_path / "results",
        checkpoint_path=checkpoint,
        data_root=tmp_path / "coco",
        num_images=5000,
        world_size=2,
        batch_size=4,
    )

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["observed_percent"]["AP"] == report["observed_percent"]["AP"]
    assert saved["delta_points"]["AP"] == 0.0
    assert saved["ap_within_0_5_points"] is True
    with Image.open(png_path) as image:
        assert image.format == "PNG"
        assert image.size == (1700, 1240)
