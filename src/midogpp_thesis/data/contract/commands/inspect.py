#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..cache_report import (
    CacheReportError,
    build_cache_domain_report,
    format_cache_domain_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect MIDOG++ contract/cache domains and emit read-only config hints.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, default=None)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    try:
        report = build_cache_domain_report(args.artifact_root, cache_report_path=args.cache_report)
    except CacheReportError as exc:
        if args.format == "json":
            print(json.dumps({"schema_version": "midogpp_cache_report_v1", "status": "FAIL", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"MIDOG++ cache/domain inspection failed: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_cache_domain_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
