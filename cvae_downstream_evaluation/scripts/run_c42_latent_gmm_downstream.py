"""Run C4.2 source-class latent GMM prior downstream experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c41_workstation import (  # noqa: E402
    assert_selected_expert_invariant,
    safe_support_selection_units_from_paths,
)
from cvae_downstream_evaluation.c42_latent_gmm import (  # noqa: E402
    C42_LATENT_GMM_GENERATION_MODES,
)
from cvae_downstream_evaluation.c42_workstation import (  # noqa: E402
    C42_ARTIFACTS_ROOT,
    C42_DEFAULT_C41_ROOT,
    build_c42_delta_summary_rows,
    build_c42_downstream_matrix,
    load_csv_rows,
    write_c42_delta_summary_csv,
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
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    C42_POSTERIOR_REPLAY_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C4.2 source-class latent GMM prior ablation with locked C4.1 routing."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument(
        "--artifacts-root",
        default=C42_ARTIFACTS_ROOT,
        help="Output root for isolated C4.2 artifacts.",
    )
    parser.add_argument(
        "--c41-artifacts-root",
        default=C42_DEFAULT_C41_ROOT,
        help="Input C4.1 full artifact root containing plain checkpoints, projections, matrix, and alignment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a small diagnostic run.")
    parser.add_argument("--resume", action="store_true", help="Resume existing latent priors and matrix rows.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda:0"),
        help="Torch device for latent extraction, generation, and scoring.",
    )
    parser.add_argument(
        "--allow-legacy-audit-columns",
        action="store_true",
        help=(
            "Drop forbidden oracle/eval audit columns from legacy support-selection artifacts instead of failing. "
            "Default is strict C4.2 protocol rejection."
        ),
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    parser.add_argument(
        "--covariance-floor",
        type=float,
        default=1.0e-4,
        help="Diagonal covariance floor / sklearn reg_covar for source-class latent GMMs.",
    )
    parser.add_argument(
        "--build-reports-only",
        action="store_true",
        help="Skip latent fitting/matrix building and rebuild reports from an existing matrix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_locked_v1_config(Path(args.config))
    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = safe_support_selection_units_from_paths(
        support_paths,
        strict_forbidden_columns=not bool(args.allow_legacy_audit_columns),
    )
    artifacts_root = repo_root / str(args.artifacts_root)
    c41_artifacts_root = repo_root / str(args.c41_artifacts_root)
    limits = MatrixBuildLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((config.experiment_seeds[0],) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers) or ((config.candidate_domains[0],) if args.smoke else None),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds) or ((config.generation_seeds[0],) if args.smoke else None),
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds) or ((config.classifier_seeds[0],) if args.smoke else None),
    )
    _assert_c41_inputs_exist(c41_artifacts_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "C4.2 source-class latent GMM prior CVAE",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_artifacts_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "generation_modes": (
                        C42_POSTERIOR_REPLAY_GENERATION_MODE,
                        C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
                        *C42_LATENT_GMM_GENERATION_MODES,
                    ),
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "covariance_floor": float(args.covariance_floor),
                    "strict_support_artifact_columns": not bool(args.allow_legacy_audit_columns),
                    "routing_scores_recomputed_for_c42": 0,
                    "c42_method_change": "latent_sampling_prior_only",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, support_units)
    print(f"Wrote C4.2-safe support units: {support_units_path}")

    if not args.build_reports_only:
        matrix_path = build_c42_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            c41_artifacts_root=c41_artifacts_root,
            support_units=support_units,
            device=args.device,
            resume=bool(args.resume),
            limits=limits,
            covariance_floor=float(args.covariance_floor),
        )
        print(f"Wrote/resumed C4.2 downstream matrix: {matrix_path}")

    _build_reports(artifacts_root, c41_artifacts_root)


def _build_reports(artifacts_root: Path, c41_artifacts_root: Path) -> None:
    support_path = artifacts_root / "tables" / "support_selection_units.csv"
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    diagnostics_path = artifacts_root / "tables" / "latent_gmm_prior_diagnostics.csv"
    c41_alignment_path = c41_artifacts_root / "tables" / "routing_to_downstream_alignment.csv"
    if not c41_alignment_path.exists():
        raise ProtocolError(f"C4.2 requires existing C4.1 alignment table: {c41_alignment_path}")

    assert_matrix_schema(matrix_path)
    selections = support_units_from_csv(support_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    alignment_rows = build_routing_alignment_rows(selections=selections, downstream_rows=downstream_rows)
    assert_selected_expert_invariant(alignment_rows)
    alignment_path = artifacts_root / "tables" / "routing_to_downstream_alignment.csv"
    write_alignment_csv(alignment_path, alignment_rows)
    c41_alignment_rows = load_csv_rows(c41_alignment_path)
    diagnostics_rows = load_csv_rows(diagnostics_path)
    delta_rows = build_c42_delta_summary_rows(
        c42_alignment_rows=alignment_rows,
        c41_alignment_rows=c41_alignment_rows,
        diagnostics_rows=diagnostics_rows,
    )
    write_c42_delta_summary_csv(artifacts_root / "tables" / "c42_delta_summary.csv", delta_rows)
    print(f"Wrote C4.2 alignment and delta reports under: {artifacts_root / 'tables'}")


def _assert_c41_inputs_exist(c41_artifacts_root: Path) -> None:
    required = (
        c41_artifacts_root / "tables" / "routing_to_downstream_alignment.csv",
        c41_artifacts_root / "tables" / "all_expert_downstream_matrix.csv",
        c41_artifacts_root / "checkpoints",
        c41_artifacts_root / "projections",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing required C4.1 full artifacts for C4.2: {missing}")


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
