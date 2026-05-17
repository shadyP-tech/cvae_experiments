"""Entrypoint for Family C label-conditioned downstream evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.family_c import (  # noqa: E402
    FAMILY_C_EXPERIMENT_NAME,
    load_family_c_downstream_config,
    preflight_family_c_downstream_inputs,
    run_family_c_downstream,
)
from cvae_downstream_evaluation.protocol import ArtifactSyncError, ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Family C label-conditioned synthetic-only downstream evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to Family C downstream YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config and synced Family C routing reports. Heavy checkpoint/cache "
            "artifacts are reported but not required."
        ),
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Only validate the Family C downstream config text and schema.",
    )
    parser.add_argument(
        "--require-heavy-artifacts",
        action="store_true",
        help="During --dry-run, also require checkpoint and embedding cache files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_family_c_downstream_config(Path(args.config))
    if args.validate_config_only:
        print(f"Config validation passed for {FAMILY_C_EXPERIMENT_NAME}.")
        return
    if args.dry_run:
        result = preflight_family_c_downstream_inputs(
            config,
            repo_root=repo_root,
            require_heavy_artifacts=bool(args.require_heavy_artifacts),
        )
        print(json.dumps({"status": "dry_run_passed", **result}, indent=2, sort_keys=True))
        return
    result = run_family_c_downstream(config, repo_root=repo_root, dry_run=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ArtifactSyncError, ProtocolError) as exc:
        raise SystemExit(str(exc)) from exc
