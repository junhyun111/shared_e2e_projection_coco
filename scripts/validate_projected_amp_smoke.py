from __future__ import annotations

import argparse
import json
from pathlib import Path

from projection_coco.smoke import load_and_validate_projected_amp_smoke


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a projected + FP16 + DDP smoke checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = load_and_validate_projected_amp_smoke(
        args.checkpoint, expected_world_size=args.expected_world_size
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
