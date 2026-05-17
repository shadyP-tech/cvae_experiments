"""Entrypoint for the locked direct support-NELBO downstream experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.protocol import (
    ArtifactSyncError,
    ProtocolError,
    load_locked_v1_config,
    resolve_required_external_artifacts,
)
from cvae_downstream_evaluation.routing import (
    add_deterministic_random_units,
    read_support_selection_units,
    write_support_selection_units,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run direct support-NELBO selected synthetic downstream evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to locked YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, support selections, and required synced artifacts without writing outputs.",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Only validate the locked v1 config. Does not check synced external artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path.cwd()
    config = load_locked_v1_config(Path(args.config))
    if args.validate_config_only:
        print("Config validation passed for locked Camelyon17 downstream v1.")
        return

    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    units = add_deterministic_random_units(read_support_selection_units(support_paths))
    resolve_required_external_artifacts(config, repo_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(units),
                    "artifacts_root": config.artifacts_root,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    artifacts_root = repo_root / config.artifacts_root
    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, units)
    print(f"Wrote support selection units: {support_units_path}")
    print(
        "Generation/classifier execution requires synced checkpoint and embedding "
        "manifests plus the workstation runtime dependencies."
    )


if __name__ == "__main__":
    try:
        main()
    except (ProtocolError, ArtifactSyncError) as exc:
        raise SystemExit(str(exc)) from exc
