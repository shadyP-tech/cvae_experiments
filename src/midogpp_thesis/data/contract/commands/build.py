#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import build_contract, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the MIDOG++ annotation-patch dataset contract.")
    parser.add_argument("--config", type=Path, required=True, help="Path to annotation_patch_v1.yaml.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV/JSON files and patch JPEGs.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = build_contract(config, overwrite=bool(args.overwrite))
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
