from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "deformable_detr"


def ensure_upstream_imports() -> Path:
    if not (UPSTREAM_ROOT / "models" / "deformable_detr.py").is_file():
        raise RuntimeError(f"Missing vendored Deformable DETR source: {UPSTREAM_ROOT}")
    path = str(UPSTREAM_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return UPSTREAM_ROOT


def upstream_commit() -> str:
    path = ensure_upstream_imports() / "UPSTREAM_COMMIT"
    return path.read_text(encoding="ascii").strip()

