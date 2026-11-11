#!/usr/bin/env python3
"""Print the canonical BrainWM human V3 dataset location and inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.paths import describe_v3_shards  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None, help="optional shard or benchmark directory")
    parser.add_argument(
        "--write-receipt",
        type=Path,
        default=None,
        help="write the resolved inventory JSON to this path",
    )
    args = parser.parse_args()
    payload = describe_v3_shards(args.path, project_root=ROOT)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write_receipt is not None:
        target = args.write_receipt.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(target)
    print(encoded, end="")


if __name__ == "__main__":
    main()
