"""Entrypoint for the R1.2 pathology foundation embedding screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.pathology_embedding_screen import (  # noqa: E402
    R12RunLimits,
    discover_pathology_cache_artifacts,
    load_r12_config,
    run_r12_pathology_embedding_screen,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run R1.2 pathology foundation embedding screen.")
    parser.add_argument("--config", required=True, help="Path to r12_pathology_embedding_screen.yaml.")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the locked R1.2 config and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and cache discovery without writing outputs.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--backbones", default=None, help="Comma-separated pathology backbone subset.")
    parser.add_argument("--representations", default=None, help="Comma-separated representation subset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_r12_config(Path(args.config))
    limits = R12RunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers),
        backbones=_parse_str_limit(args.backbones),
        representations=_parse_str_limit(args.representations),
    )

    if args.validate_config_only:
        print("Config validation passed for R1.2 pathology foundation embedding screen.")
        return

    if args.dry_run:
        artifacts = discover_pathology_cache_artifacts(config=config, repo_root=repo_root, limits=limits)
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "artifact_candidates": len(artifacts),
                    "cache_candidates": [
                        {
                            "experiment_seed": item.experiment_seed,
                            "backbone_name": item.backbone_name,
                            "train_cache": str(item.train_cache),
                            "test_cache": str(item.test_cache),
                            "train_exists": item.train_cache.exists(),
                            "test_exists": item.test_cache.exists(),
                        }
                        for item in artifacts
                    ],
                    "artifacts_root": config.artifacts_root,
                    "backbones": list(limits.backbones or config.backbones),
                    "representations": list(limits.representations or config.representations),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = run_r12_pathology_embedding_screen(config=config, repo_root=repo_root, limits=limits)
    print(
        json.dumps(
            {
                "status": "r12_pathology_embedding_screen_complete",
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
