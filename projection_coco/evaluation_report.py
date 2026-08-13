from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COCO_METRICS: tuple[tuple[str, str], ...] = (
    ("map", "AP"),
    ("map50", "AP50"),
    ("map75", "AP75"),
    ("map_small", "AP_S"),
    ("map_medium", "AP_M"),
    ("map_large", "AP_L"),
    ("mar1", "AR@1"),
    ("mar10", "AR@10"),
    ("mar100", "AR@100"),
    ("mar_small", "AR_S"),
    ("mar_medium", "AR_M"),
    ("mar_large", "AR_L"),
)

# Official r50_deformable_detr log, epoch 49 (the 50th epoch).
# Values are the unrounded pycocotools COCOeval bbox statistics.
OFFICIAL_REFERENCE: dict[str, float] = {
    "map": 0.4451056887905585,
    "map50": 0.6359162542552462,
    "map75": 0.4872799866444648,
    "map_small": 0.2706573207581538,
    "map_medium": 0.4762535527472109,
    "map_large": 0.5955730848128388,
    "mar1": 0.353169024479641,
    "mar10": 0.5879076322429762,
    "mar100": 0.6297535080079202,
    "mar_small": 0.4228086219915269,
    "mar_medium": 0.6721900994593547,
    "mar_large": 0.8197323368532937,
}

REFERENCE_SOURCE = (
    "Official Deformable-DETR r50_deformable_detr release log, epoch 49"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        filename,
        f"/usr/share/fonts/truetype/dejavu/{filename}",
        f"C:/Windows/Fonts/{'arialbd.ttf' if bold else 'arial.ttf'}",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_png(report: dict, path: Path) -> None:
    width, height = 1700, 1240
    image = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(image)
    title_font = _font(46, bold=True)
    subtitle_font = _font(24)
    header_font = _font(23, bold=True)
    body_font = _font(22)
    small_font = _font(19)

    draw.text(
        (80, 55),
        "Official Deformable DETR - COCO 2017 Validation",
        fill="#f8fafc",
        font=title_font,
    )
    draw.text(
        (82, 120),
        (
            f"bbox COCOeval | {report['num_images']:,} images | "
            f"{report['world_size']} GPU(s) | batch/GPU {report['batch_size']}"
        ),
        fill="#94a3b8",
        font=subtitle_font,
    )

    columns = (90, 345, 600, 850, 1090)
    top = 195
    row_height = 65
    table_bottom = top + row_height * (len(COCO_METRICS) + 1)
    draw.rounded_rectangle(
        (65, top - 15, width - 65, table_bottom + 15),
        radius=18,
        fill="#111c2f",
        outline="#26354d",
        width=2,
    )
    for x, label in zip(
        columns,
        ("Metric", "Measured", "Official log", "Delta", "0 - 100 scale"),
    ):
        draw.text((x, top), label, fill="#cbd5e1", font=header_font)
    draw.line(
        (80, top + 47, width - 80, top + 47), fill="#334155", width=2
    )

    observed = report["observed_percent"]
    reference = report["official_reference_percent"]
    delta = report["delta_points"]
    bar_left, bar_right = columns[-1], width - 100
    bar_width = bar_right - bar_left
    for index, (_, label) in enumerate(COCO_METRICS):
        y = top + row_height * (index + 1)
        if index == 6:
            draw.line((80, y - 10, width - 80, y - 10), fill="#475569", width=3)
        draw.text((columns[0], y), label, fill="#e2e8f0", font=body_font)
        draw.text(
            (columns[1], y),
            f"{observed[label]:.3f}",
            fill="#60a5fa",
            font=body_font,
        )
        draw.text(
            (columns[2], y),
            f"{reference[label]:.3f}",
            fill="#fbbf24",
            font=body_font,
        )
        delta_color = "#86efac" if abs(delta[label]) <= 0.5 else "#fca5a5"
        draw.text(
            (columns[3], y),
            f"{delta[label]:+.3f}",
            fill=delta_color,
            font=body_font,
        )
        bar_y = y + 4
        draw.rounded_rectangle(
            (bar_left, bar_y, bar_right, bar_y + 22),
            radius=8,
            fill="#1e293b",
        )
        measured_x = bar_left + int(
            bar_width * max(0.0, min(observed[label], 100.0)) / 100.0
        )
        draw.rounded_rectangle(
            (bar_left, bar_y, max(bar_left + 2, measured_x), bar_y + 22),
            radius=8,
            fill="#3b82f6",
        )
        reference_x = bar_left + int(
            bar_width * max(0.0, min(reference[label], 100.0)) / 100.0
        )
        draw.line(
            (reference_x, bar_y - 4, reference_x, bar_y + 27),
            fill="#fbbf24",
            width=4,
        )

    footer_y = table_bottom + 45
    ap_delta = delta["AP"]
    status = "MATCH" if abs(ap_delta) <= 0.5 else "CHECK PIPELINE"
    status_color = "#86efac" if status == "MATCH" else "#fca5a5"
    draw.text(
        (80, footer_y),
        f"AP difference: {ap_delta:+.3f} points  |  {status}",
        fill=status_color,
        font=header_font,
    )
    draw.text(
        (80, footer_y + 45),
        f"Reference: {report['reference_source']}",
        fill="#94a3b8",
        font=small_font,
    )
    draw.text(
        (80, footer_y + 76),
        "Blue bar: measured   Yellow marker: official release log",
        fill="#94a3b8",
        font=small_font,
    )
    image.save(path, format="PNG", optimize=True)


def write_official_evaluation_report(
    metrics: dict[str, float],
    output_dir: Path,
    *,
    checkpoint_path: Path,
    data_root: Path,
    num_images: int,
    world_size: int,
    batch_size: int,
) -> tuple[Path, Path, dict]:
    missing = [key for key, _ in COCO_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"Missing COCO metrics: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    observed = {
        label: float(metrics[key]) * 100.0 for key, label in COCO_METRICS
    }
    reference = {
        label: OFFICIAL_REFERENCE[key] * 100.0 for key, label in COCO_METRICS
    }
    delta = {label: observed[label] - reference[label] for _, label in COCO_METRICS}
    checkpoint_path = checkpoint_path.resolve()
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": "COCO 2017 val bbox / pycocotools COCOeval",
        "model_config": {
            "model": "Deformable DETR R50 multi-scale",
            "num_queries": 300,
            "num_feature_levels": 4,
            "two_stage": False,
            "with_box_refine": False,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "data_root": str(data_root.resolve()),
        "num_images": num_images,
        "world_size": world_size,
        "batch_size": batch_size,
        "val_seconds": float(metrics.get("val_seconds", 0.0)),
        "observed_percent": observed,
        "official_reference_percent": reference,
        "delta_points": delta,
        "ap_within_0_5_points": abs(delta["AP"]) <= 0.5,
        "reference_source": REFERENCE_SOURCE,
    }
    json_path = output_dir / "coco_official_metrics.json"
    png_path = output_dir / "coco_official_metrics.png"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _render_png(report, png_path)
    return json_path, png_path, report
