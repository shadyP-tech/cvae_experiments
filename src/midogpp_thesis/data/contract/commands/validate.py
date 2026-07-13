#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import validate_contract
from ..validation import ValidationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen MIDOG++ annotation-patch contract artifact.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        result = validate_contract(args.artifact_root, schema_path=args.schema)
    except ValidationError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
