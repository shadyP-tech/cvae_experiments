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
from cvae_downstream_evaluation.matrix import (
    MatrixBuildLimits,
    build_all_expert_downstream_matrix,
    discover_support_run_artifacts,
    materialize_downstream_manifests,
)
from cvae_downstream_evaluation.downstream import assert_matrix_schema, read_candidate_downstream_matrix
from cvae_downstream_evaluation.reporting import (
    baseline_comparison_rows,
    build_routing_alignment_rows,
    classify_decision,
    stability_rows,
    support_size_stratified_summary,
    write_alignment_csv,
    write_baseline_comparison_csv,
    write_decision_summary,
    write_stability_csv,
    write_support_size_summary_csv,
)
from cvae_downstream_evaluation.routing import (
    add_deterministic_random_units,
    read_support_selection_units,
    support_units_from_csv,
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
    parser.add_argument(
        "--prepare-manifests",
        action="store_true",
        help="Discover frozen support-run artifacts and write downstream manifest CSVs.",
    )
    parser.add_argument(
        "--build-matrix",
        action="store_true",
        help="Generate all_expert_downstream_matrix.csv from frozen artifacts.",
    )
    parser.add_argument(
        "--build-reports",
        action="store_true",
        help="Build routing-to-downstream alignment and decision report tables.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda:0"),
        help="Torch device visible inside the process. With CUDA_VISIBLE_DEVICES=1 use cuda:0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing matrix rows with matching primary keys.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
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
    artifacts_root = repo_root / config.artifacts_root

    if args.prepare_manifests:
        artifacts = discover_support_run_artifacts(config=config, repo_root=repo_root)
        if not args.dry_run:
            materialize_downstream_manifests(artifacts=artifacts, artifacts_root=artifacts_root)
            print(f"Wrote downstream manifests under: {artifacts_root / 'manifests'}")

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

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, units)
    print(f"Wrote support selection units: {support_units_path}")

    if args.build_matrix:
        matrix_path = build_all_expert_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            support_units=units,
            device=args.device,
            resume=bool(args.resume),
            limits=MatrixBuildLimits(
                experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
                heldout_centers=_parse_str_limit(args.limit_heldout_centers),
                generation_seeds=_parse_int_limit(args.limit_generation_seeds),
                classifier_seeds=_parse_int_limit(args.limit_classifier_seeds),
            ),
        )
        print(f"Wrote/resumed downstream matrix: {matrix_path}")

    if args.build_reports:
        _build_reports(artifacts_root)

    if not args.build_matrix and not args.build_reports:
        print(
            "Prepared support units. Add --build-matrix on the workstation to run "
            "synthetic generation/classifier scoring."
        )


def _build_reports(artifacts_root: Path) -> None:
    support_path = artifacts_root / "tables" / "support_selection_units.csv"
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    assert_matrix_schema(matrix_path)
    selections = support_units_from_csv(support_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    alignment_rows = build_routing_alignment_rows(selections=selections, downstream_rows=downstream_rows)
    write_alignment_csv(artifacts_root / "tables" / "routing_to_downstream_alignment.csv", alignment_rows)
    write_baseline_comparison_csv(
        artifacts_root / "tables" / "baseline_comparison.csv",
        baseline_comparison_rows(alignment_rows=alignment_rows, downstream_rows=downstream_rows),
    )
    write_support_size_summary_csv(
        artifacts_root / "tables" / "support_size_stratified_downstream_summary.csv",
        support_size_stratified_summary(alignment_rows),
    )
    write_stability_csv(
        artifacts_root / "tables" / "selection_stability.csv",
        stability_rows(alignment_rows, group="selection_support"),
    )
    write_stability_csv(
        artifacts_root / "tables" / "generation_classifier_stability.csv",
        stability_rows(alignment_rows, group="generation_classifier"),
    )
    write_decision_summary(
        artifacts_root / "reports" / "decision_summary.md",
        classify_decision(alignment_rows),
    )
    print(f"Wrote downstream reports under: {artifacts_root}")


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
