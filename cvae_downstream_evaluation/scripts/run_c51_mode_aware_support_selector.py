"""Run C5.1 mode-aware unlabeled support-distance routing."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c41_workstation import safe_support_selection_units_from_paths  # noqa: E402
from cvae_downstream_evaluation.c51_mode_aware import (  # noqa: E402
    C51_ARTIFACTS_ROOT,
    C51_DEFAULT_C41_ROOT,
    C51_DEFAULT_C42_ROOT,
    build_c51_reports,
    build_c51_support_mode_scores,
)
from cvae_downstream_evaluation.matrix import MatrixBuildLimits  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402
from cvae_downstream_evaluation.routing import write_support_selection_units  # noqa: E402
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    METADATA_METHOD,
    SOURCE_GLOBAL_METHOD,
    SUPPORT_NELBO_METHOD,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C5.1 mode-aware unlabeled support-distance selector over the C4.1/C4.2 generator bank."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=C51_ARTIFACTS_ROOT, help="Output root for C5.1 artifacts.")
    parser.add_argument("--c41-artifacts-root", default=C51_DEFAULT_C41_ROOT, help="Input C4.1 full artifact root.")
    parser.add_argument("--c42-artifacts-root", default=C51_DEFAULT_C42_ROOT, help="Input C4.2 artifact root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the C5.1 execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for one seed/center/support/generation-seed diagnostic run.")
    parser.add_argument("--resume", action="store_true", help="Accepted for CLI symmetry; C5.1 rewrites report tables.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"), help="Torch device for generation.")
    parser.add_argument(
        "--allow-legacy-audit-columns",
        action="store_true",
        help="Drop forbidden oracle/eval audit columns from legacy support-selection artifacts instead of failing.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Accepted for symmetry; selectors ignore classifier seed.")
    parser.add_argument(
        "--build-reports-only",
        action="store_true",
        help="Skip support-distance scoring and rebuild reports from existing C5.1 score rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ = args.resume
    repo_root = Path.cwd()
    config = load_locked_v1_config(Path(args.config))
    artifacts_root = repo_root / str(args.artifacts_root)
    c41_root = repo_root / str(args.c41_artifacts_root)
    c42_root = repo_root / str(args.c42_artifacts_root)
    _assert_inputs_exist(c41_root, c42_root)
    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = safe_support_selection_units_from_paths(
        support_paths,
        strict_forbidden_columns=not bool(args.allow_legacy_audit_columns),
        methods=(SUPPORT_NELBO_METHOD, METADATA_METHOD, SOURCE_GLOBAL_METHOD),
    )
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
                    "experiment": "C5.1 mode-aware unlabeled support-distance routing",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "c42_artifacts_root": str(c42_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds_ignored_by_selector": limits.classifier_seeds,
                    },
                    "primary_selector": "support_distance_rankmean_dino_seed_marginal_top1",
                    "score_space_primary": "dino_original",
                    "target_support_labels_used": 0,
                    "target_eval_labels_used_for_selection": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, support_units)
    print(f"Wrote C5.1-safe support units: {support_units_path}")

    if not args.build_reports_only:
        score_path = build_c51_support_mode_scores(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            c41_artifacts_root=c41_root,
            c42_artifacts_root=c42_root,
            support_units=support_units,
            device=args.device,
            limits=limits,
        )
        print(f"Wrote C5.1 support-mode scores: {score_path}")
    outputs = build_c51_reports(
        artifacts_root=artifacts_root,
        c41_artifacts_root=c41_root,
        c42_artifacts_root=c42_root,
    )
    print(f"Wrote C5.1 reports under: {artifacts_root / 'tables'}")
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


def _assert_inputs_exist(c41_root: Path, c42_root: Path) -> None:
    required = (
        c41_root / "tables" / "all_expert_downstream_matrix.csv",
        c41_root / "checkpoints",
        c41_root / "projections",
        c42_root / "tables" / "all_expert_downstream_matrix.csv",
        c42_root / "latent_priors",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing required C4.1/C4.2 artifacts for C5.1: {missing}")


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
