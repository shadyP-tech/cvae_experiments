#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a seeded cvae_testing run directory containing samples.csv, "
            "split_manifest.json, leakage_report.json, and config_resolved.yaml only."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment config path.")
    parser.add_argument("--seed", type=int, required=True, help="Experiment seed used for split construction.")
    parser.add_argument("--run-id", type=str, required=True, help="Run directory name to materialize.")
    parser.add_argument(
        "--overwrite-manifests",
        action="store_true",
        help="Overwrite existing manifest/report/config files for this run ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the split and print intended paths without writing files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from src.manifest_only import materialize_manifest_only_run

    if args.config.is_absolute():
        config_path = args.config
    else:
        config_path = PROJECT_ROOT / args.config
        repo_relative = PROJECT_ROOT.parent / args.config
        if not config_path.exists() and repo_relative.exists():
            config_path = repo_relative
    result = materialize_manifest_only_run(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        seed=int(args.seed),
        run_id=str(args.run_id),
        overwrite=bool(args.overwrite_manifests),
        dry_run=bool(args.dry_run),
    )
    print(
        json.dumps(
            {
                "status": "dry_run" if args.dry_run else "complete",
                "manifest_only": True,
                "seed": int(args.seed),
                "run_id": str(args.run_id),
                "run_root": str(result.run_root),
                "n_records": result.n_records,
                "split_counts": result.split_counts,
                "outputs": {
                    "samples_manifest": str(result.samples_manifest),
                    "split_manifest": str(result.split_manifest),
                    "leakage_report": str(result.leakage_report),
                    "manifest_only_report": str(result.manifest_only_report),
                    "config_resolved": str(result.config_resolved),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
