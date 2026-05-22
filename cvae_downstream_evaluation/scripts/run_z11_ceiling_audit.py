"""Entrypoint for the Z1.1 current-setup ceiling audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.ceiling_audit import (  # noqa: E402
    Z11RunLimits,
    discover_support_audit_artifacts,
    load_z11_config,
    run_z11_ceiling_audit,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Z1.1 current-setup ceiling audit.")
    parser.add_argument("--config", required=True, help="Path to z11_current_setup_ceiling_audit.yaml.")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the locked Z1.1 config and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and artifact discovery without writing outputs.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--representations", default=None, help="Comma-separated representation subset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_z11_config(Path(args.config))
    limits = Z11RunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers),
        representations=_parse_str_limit(args.representations),
    )

    if args.validate_config_only:
        print("Config validation passed for Z1.1 current-setup ceiling audit.")
        return

    if args.dry_run:
        artifacts = discover_support_audit_artifacts(config=config, repo_root=repo_root, limits=limits)
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "artifact_candidates": len(artifacts),
                    "artifact_run_dirs": [str(item.run_dir) for item in artifacts],
                    "artifacts_root": config.artifacts_root,
                    "representations": list(limits.representations or config.representations),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = run_z11_ceiling_audit(config=config, repo_root=repo_root, limits=limits)
    print(
        json.dumps(
            {
                "status": "z11_audit_complete",
                "decision_labels": result.decision_labels,
                "outputs": {key: str(value) for key, value in result.output_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_int_limit(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_str_limit(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(str(part.strip()) for part in str(raw).split(",") if part.strip())


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
