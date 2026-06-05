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

from midogpp_contract import validate_contract  # noqa: E402
from midogpp_contract.validation import ValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen MIDOG++ annotation-patch contract artifact.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=None)
    args = parser.parse_args()

    try:
        result = validate_contract(args.artifact_root, schema_path=args.schema)
    except ValidationError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
