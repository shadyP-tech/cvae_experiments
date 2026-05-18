"""Entrypoint for Family E1 direct embedding sampler downstream diagnostics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.family_e1 import (  # noqa: E402
    FamilyE1BuildLimits,
    build_family_e1_all_expert_downstream_matrix,
    build_family_e1_reports,
    discover_family_e1_support_artifacts,
    load_family_e1_config,
    read_family_e1_support_units,
)
from cvae_downstream_evaluation.protocol import ArtifactSyncError, ProtocolError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Family E1 direct embedding sampler downstream diagnostic baseline."
    )
    parser.add_argument("--config", required=True, help="Path to Family E1 locked YAML config.")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the locked Family E1 config and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate synced support artifacts and print the resolved run shape without writing outputs.",
    )
    parser.add_argument(
        "--build-matrix",
        action="store_true",
        help="Fit direct source-train samplers and build the E1 all-expert downstream matrix.",
    )
    parser.add_argument(
        "--build-reports",
        action="store_true",
        help="Build source-transfer selector reports and decision summary from an existing E1 matrix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve existing matrix rows with matching Family E1 primary keys.",
    )
    parser.add_argument(
        "--c2-metrics-json",
        default=None,
        help="Optional JSON file containing c2_selected_center_level_mean_bacc, c2_oracle_center_level_mean_bacc, and c2_oracle_gap_bacc.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-support-sizes", default=None, help="Comma-separated support sizes.")
    parser.add_argument("--limit-support-seeds", default=None, help="Comma-separated support seeds.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path.cwd()
    config = load_family_e1_config(Path(args.config))
    if args.validate_config_only:
        print("Config validation passed for Family E1 direct embedding sampler downstream v1.")
        return

    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = read_family_e1_support_units(support_paths)
    artifacts_root = repo_root / config.artifacts_root
    limits = FamilyE1BuildLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers),
        support_sizes=_parse_int_limit(args.limit_support_sizes),
        support_seeds=_parse_int_limit(args.limit_support_seeds),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds),
    )

    discovered = discover_family_e1_support_artifacts(config=config, repo_root=repo_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": config.artifacts_root,
                    "support_selection_files": len(support_paths),
                    "support_contexts": len(support_units),
                    "support_run_artifacts": len(discovered),
                    "artifacts_root": str(artifacts_root),
                    "pca_before_sampler_enabled": int(config.pca_enabled),
                    "modes": list(config.modes),
                    "budget_per_class": config.budget_per_class,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.build_matrix:
        paths = build_family_e1_all_expert_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            support_units=support_units,
            resume=bool(args.resume),
            limits=limits,
        )
        print(f"Wrote Family E1 matrix: {paths['matrix']}")
        print(f"Wrote Family E1 protocol audit: {paths['protocol_audit']}")

    if args.build_reports:
        c2_metrics = _load_c2_metrics(Path(args.c2_metrics_json)) if args.c2_metrics_json else {}
        paths = build_family_e1_reports(
            artifacts_root=artifacts_root,
            candidate_domains=config.candidate_domains,
            c2_metrics=c2_metrics,
        )
        print(f"Wrote Family E1 selection alignment: {paths['alignment']}")
        print(f"Wrote Family E1 decision summary: {paths['decision_summary']}")

    if not args.build_matrix and not args.build_reports:
        print(
            "Validated Family E1 support contexts. Add --build-matrix on the workstation "
            "to fit direct embedding samplers and score downstream classifiers."
        )


def _load_c2_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("C2 metrics JSON must be an object.")
    return {str(key): float(value) for key, value in payload.items()}


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
    except (ProtocolError, ArtifactSyncError) as exc:
        raise SystemExit(str(exc)) from exc
