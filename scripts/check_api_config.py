#!/usr/bin/env python3
"""Print non-sensitive nanoBrain third-party API configuration status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nanobrain.api_config import APIConfig, APIConfigError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="dotenv file to inspect (default: project .env)",
    )
    parser.add_argument(
        "--require-configured",
        action="store_true",
        help="return exit code 2 when key, URL, or model is missing",
    )
    args = parser.parse_args()
    try:
        config = APIConfig.from_env(env_file=args.env_file)
        if args.require_configured:
            config.validate(require_credentials=True)
    except APIConfigError as exc:
        print(json.dumps({"configured": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(config.redacted(), indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if config.configured else (2 if args.require_configured else 0)


if __name__ == "__main__":
    raise SystemExit(main())
