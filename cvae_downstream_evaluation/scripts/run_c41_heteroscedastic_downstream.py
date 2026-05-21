"""Run C4.1 heteroscedastic class-conditioned PCA64 downstream experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c41_workstation import (  # noqa: E402
    C41_ARTIFACTS_ROOT,
    assert_selected_expert_invariant,
    build_c41_delta_summary_rows,
    build_c41_downstream_matrix,
    c41_training_profile_from_config,
    load_generator_diagnostics,
    safe_support_selection_units_from_paths,
    write_c41_delta_summary_csv,
)
from cvae_downstream_evaluation.downstream import (  # noqa: E402
    assert_matrix_schema,
    read_candidate_downstream_matrix,
)
from cvae_downstream_evaluation.matrix import MatrixBuildLimits  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402
from cvae_downstream_evaluation.reporting import build_routing_alignment_rows, write_alignment_csv  # noqa: E402
from cvae_downstream_evaluation.routing import (  # noqa: E402
    support_units_from_csv,
    write_support_selection_units,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C4.1 plain-vs-heteroscedastic PCA64 class-conditioned CVAE downstream evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument(
        "--artifacts-root",
        default=C41_ARTIFACTS_ROOT,
        help="Output root for isolated C4.1 artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a small diagnostic run.")
    parser.add_argument("--resume", action="store_true", help="Resume existing checkpoints and matrix rows.")
    parser.add_argument(
        "--training-profile",
        choices=("smoke", "full"),
        default="full",
        help="Training profile for retrained plain and heteroscedastic generators.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda:0"),
        help="Torch device for generation/scoring. Training uses the cvae_testing training helper device policy.",
    )
    parser.add_argument(
        "--allow-legacy-audit-columns",
        action="store_true",
        help=(
            "Drop forbidden oracle/eval audit columns from legacy support-selection artifacts instead of failing. "
            "Default is strict C4.1 protocol rejection."
        ),
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    parser.add_argument(
        "--build-reports-only",
        action="store_true",
        help="Skip model training/matrix building and rebuild reports from an existing matrix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config_path = Path(args.config)
    config = load_locked_v1_config(config_path)
    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = safe_support_selection_units_from_paths(
        support_paths,
        strict_forbidden_columns=not bool(args.allow_legacy_audit_columns),
    )
    training_profile_name = "smoke" if args.smoke else args.training_profile
    training_profile = c41_training_profile_from_config(config_path, profile=training_profile_name)
    artifacts_root = repo_root / str(args.artifacts_root)
    limits = MatrixBuildLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((config.experiment_seeds[0],) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers) or ((config.candidate_domains[0],) if args.smoke else None),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds) or ((config.generation_seeds[0],) if args.smoke else None),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds) or ((config.classifier_seeds[0],) if args.smoke else None),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "C4.1 heteroscedastic decoder CVAE",
                    "artifacts_root": str(artifacts_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "training_profile": training_profile.__dict__,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "strict_support_artifact_columns": not bool(args.allow_legacy_audit_columns),
                    "routing_scores_recomputed_for_heteroscedastic": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, support_units)
    print(f"Wrote C4.1-safe support units: {support_units_path}")

    if not args.build_reports_only:
        matrix_path = build_c41_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            support_units=support_units,
            device=args.device,
            resume=bool(args.resume),
            training_profile=training_profile,
            limits=limits,
        )
        print(f"Wrote/resumed C4.1 downstream matrix: {matrix_path}")

    _build_reports(artifacts_root)


def _build_reports(artifacts_root: Path) -> None:
    support_path = artifacts_root / "tables" / "support_selection_units.csv"
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    diagnostics_path = artifacts_root / "tables" / "generator_distribution_diagnostics.csv"
    assert_matrix_schema(matrix_path)
    selections = support_units_from_csv(support_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    alignment_rows = build_routing_alignment_rows(selections=selections, downstream_rows=downstream_rows)
    assert_selected_expert_invariant(alignment_rows)
    write_alignment_csv(artifacts_root / "tables" / "routing_to_downstream_alignment.csv", alignment_rows)
    diagnostics = load_generator_diagnostics(diagnostics_path)
    delta_rows = build_c41_delta_summary_rows(alignment_rows=alignment_rows, diagnostic_rows=diagnostics)
    write_c41_delta_summary_csv(artifacts_root / "tables" / "c41_delta_summary.csv", delta_rows)
    write_c41_delta_summary_csv(artifacts_root / "tables" / "c41_generation_mode_comparison.csv", delta_rows)
    print(f"Wrote C4.1 alignment and delta reports under: {artifacts_root / 'tables'}")


def _parse_int_limit(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def _parse_str_limit(raw: str | None) -> tuple[str, ...] | None:
    if raw is None or not str(raw).strip():
        return None
    return tuple(str(part.strip()) for part in str(raw).split(",") if part.strip())


if __name__ == "__main__":
    main()
