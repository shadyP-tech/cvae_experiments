"""Run C8.1 unlabeled target-support alignment over C6.3 geometric ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c81_support_alignment import (  # noqa: E402
    C81_ARTIFACTS_ROOT,
    C81_DEFAULT_C41_ROOT,
    C81_DEFAULT_C42_ROOT,
    C81_DEFAULT_C52_ROOT,
    C81_DEFAULT_C62_ROOT,
    C81_DEFAULT_C63_ROOT,
    C81RunLimits,
    PRIMARY_ALIGNMENT_POLICY,
    run_c81_support_alignment,
)
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run C8.1 unlabeled target-support alignment over the fixed C6.3 geometric ensemble bank."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=C81_ARTIFACTS_ROOT, help="Output root for C8.1 artifacts.")
    parser.add_argument("--c41-artifacts-root", default=C81_DEFAULT_C41_ROOT, help="Input C4.1 full artifact root.")
    parser.add_argument("--c42-artifacts-root", default=C81_DEFAULT_C42_ROOT, help="Input C4.2 artifact root.")
    parser.add_argument("--c52-artifacts-root", default=C81_DEFAULT_C52_ROOT, help="Input C5.2 artifact root.")
    parser.add_argument("--c62-artifacts-root", default=C81_DEFAULT_C62_ROOT, help="Input C6.2 artifact root.")
    parser.add_argument("--c63-artifacts-root", default=C81_DEFAULT_C63_ROOT, help="Input C6.3 artifact root.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the C8.1 execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for one seed/center/support and one classifier seed; keeps all generation seeds.")
    parser.add_argument("--resume", action="store_true", help="Accepted for CLI symmetry; C8.1 rewrites report tables.")
    parser.add_argument("--device", default="auto", help="Torch device for generation, e.g. auto, cpu, cuda:0, cuda:1.")
    parser.add_argument("--enable-full-coral-diagnostic", action="store_true", help="Opt in to the diagnostic full-CORAL policy.")
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-support-sizes", default=None, help="Comma-separated support sizes.")
    parser.add_argument("--limit-support-seeds", default=None, help="Comma-separated support seeds.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _ = args.resume
    repo_root = Path.cwd()
    config = load_locked_v1_config(Path(args.config))
    artifacts_root = repo_root / str(args.artifacts_root)
    c41_root = repo_root / str(args.c41_artifacts_root)
    c42_root = repo_root / str(args.c42_artifacts_root)
    c52_root = repo_root / str(args.c52_artifacts_root)
    c62_root = repo_root / str(args.c62_artifacts_root)
    c63_root = repo_root / str(args.c63_artifacts_root)
    _assert_inputs_exist(c41_root, c42_root, c52_root, c62_root, c63_root)
    limits = C81RunLimits(
        experiment_seeds=_parse_int_limit(args.limit_experiment_seeds) or ((config.experiment_seeds[0],) if args.smoke else None),
        heldout_centers=_parse_str_limit(args.limit_heldout_centers) or ((config.candidate_domains[0],) if args.smoke else None),
        support_sizes=_parse_int_limit(args.limit_support_sizes) or ((config.support_sizes[0],) if args.smoke else None),
        support_seeds=_parse_int_limit(args.limit_support_seeds) or ((config.support_seeds[0],) if args.smoke else None),
        generation_seeds=_parse_int_limit(args.limit_generation_seeds) or None,
        classifier_seeds=_parse_int_limit(args.limit_classifier_seeds) or ((config.classifier_seeds[0],) if args.smoke else None),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "experiment": "C8.1 unlabeled target-support adaptation diagnostic",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "c42_artifacts_root": str(c42_root),
                    "c52_artifacts_root": str(c52_root),
                    "c62_artifacts_root": str(c62_root),
                    "c63_artifacts_root": str(c63_root),
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "support_sizes": limits.support_sizes,
                        "support_seeds": limits.support_seeds,
                        "generation_seeds": limits.generation_seeds or config.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "primary_alignment_policy": PRIMARY_ALIGNMENT_POLICY,
                    "full_coral_diagnostic_enabled": bool(args.enable_full_coral_diagnostic),
                    "target_support_x_used_for_alignment": 1,
                    "target_support_labels_used": 0,
                    "target_eval_features_used_for_alignment": 0,
                    "target_eval_features_transformed_by_alignment": 0,
                    "target_eval_labels_used_for_selection": 0,
                    "checkpoints_retrained": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    outputs = run_c81_support_alignment(
        config=config,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
        c41_artifacts_root=c41_root,
        c42_artifacts_root=c42_root,
        c52_artifacts_root=c52_root,
        c62_artifacts_root=c62_root,
        c63_artifacts_root=c63_root,
        device=str(args.device),
        limits=limits,
        enable_full_coral_diagnostic=bool(args.enable_full_coral_diagnostic),
    )
    print(f"Wrote C8.1 reports under: {artifacts_root / 'tables'}")
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2, sort_keys=True))


def _assert_inputs_exist(c41_root: Path, c42_root: Path, c52_root: Path, c62_root: Path, c63_root: Path) -> None:
    required = (
        c41_root / "tables" / "support_selection_units.csv",
        c41_root / "checkpoints",
        c41_root / "projections",
        c42_root / "latent_priors",
        c52_root / "tables" / "c52_selected_route_utility_join.csv",
        c62_root / "tables" / "c62_late_ensemble_downstream_matrix.csv",
        c63_root / "tables" / "c63_geometric_late_ensemble_downstream_matrix.csv",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing required C4.1/C4.2/C5.2/C6.2/C6.3 artifacts for C8.1: {missing}")


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
        raise SystemExit(f"Protocol error: {exc}") from exc

