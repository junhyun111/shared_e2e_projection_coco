#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from projection_coco.experiment_matrix import (
    load_and_validate_experiment_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that baseline, aux_only, and projected checkpoints form a "
            "controlled comparison matrix."
        )
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("aux_only", type=Path)
    parser.add_argument("projected", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_and_validate_experiment_matrix(
        [args.baseline, args.aux_only, args.projected]
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
