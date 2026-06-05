#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_SRC = SCRIPT_DIR.parent / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from midogpp_contract import build_contract, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MIDOG++ annotation-patch dataset contract.")
    parser.add_argument("--config", type=Path, required=True, help="Path to annotation_patch_v1.yaml.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV/JSON files and patch JPEGs.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = build_contract(config, overwrite=bool(args.overwrite))
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
