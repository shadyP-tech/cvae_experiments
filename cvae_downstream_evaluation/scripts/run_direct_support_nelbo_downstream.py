"""Entrypoint for the locked direct support-NELBO downstream experiment."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.classifiers import ClassifierSpec, classifier_grid_hash
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
        help="Generate downstream utility matrix from frozen artifacts.",
    )
    parser.add_argument(
        "--diagnostic-matrix",
        action="store_true",
        help="Write tables/diagnostic_downstream_utility.csv instead of the legacy all_expert_downstream_matrix.csv.",
    )
    parser.add_argument(
        "--matrix-path",
        default=None,
        help="Optional explicit matrix output path. Diagnostic paths must be named diagnostic_downstream_utility.*.",
    )
    parser.add_argument(
        "--build-reports",
        action="store_true",
        help="Build routing-to-downstream alignment and decision report tables.",
    )
    parser.add_argument(
        "--report-matrix-path",
        default=None,
        help="Optional matrix path to use when building reports.",
    )
    parser.add_argument(
        "--source-inner-classifier-tuning",
        action="store_true",
        help=(
            "Select a shared classifier spec per heldout center/classifier seed using "
            "source-inner LODO before target evaluation."
        ),
    )
    parser.add_argument(
        "--source-inner-classifier-tuning-path",
        default=None,
        help="Optional output CSV for source-inner classifier tuning rows.",
    )
    parser.add_argument(
        "--classifier-c-grid",
        default="0.1,1.0,10.0",
        help="Comma-separated C values for source-inner classifier tuning.",
    )
    parser.add_argument(
        "--classifier-penalties",
        default="l2",
        help="Comma-separated penalties for source-inner classifier tuning.",
    )
    parser.add_argument(
        "--classifier-solvers",
        default="lbfgs",
        help="Comma-separated solvers for source-inner classifier tuning.",
    )
    parser.add_argument(
        "--classifier-class-weights",
        default="none,balanced",
        help="Comma-separated class weights: none and/or balanced.",
    )
    parser.add_argument(
        "--classifier-max-iters",
        default="2000",
        help="Comma-separated max_iter values for source-inner classifier tuning.",
    )
    parser.add_argument(
        "--classifier-l1-ratios",
        default="",
        help="Comma-separated l1_ratio values used only for elasticnet+saga specs.",
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
        classifier_specs = _classifier_specs_from_args(args) if args.source_inner_classifier_tuning else None
        print(
            json.dumps(
                {
                    "status": "dry_run_passed",
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(units),
                    "artifacts_root": config.artifacts_root,
                    "source_inner_classifier_tuning": bool(args.source_inner_classifier_tuning),
                    "classifier_grid_hash": classifier_grid_hash(classifier_specs) if classifier_specs else "",
                    "classifier_grid_size": len(classifier_specs or ()),
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
        classifier_specs = _classifier_specs_from_args(args) if args.source_inner_classifier_tuning else None
        matrix_path = build_all_expert_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            support_units=units,
            device=args.device,
            resume=bool(args.resume),
            output_path=Path(args.matrix_path) if args.matrix_path else None,
            diagnostic_output=bool(args.diagnostic_matrix),
            source_inner_classifier_specs=classifier_specs,
            source_inner_classifier_tuning_path=(
                Path(args.source_inner_classifier_tuning_path)
                if args.source_inner_classifier_tuning_path
                else None
            ),
            limits=MatrixBuildLimits(
                experiment_seeds=_parse_int_limit(args.limit_experiment_seeds),
                heldout_centers=_parse_str_limit(args.limit_heldout_centers),
                generation_seeds=_parse_int_limit(args.limit_generation_seeds),
                classifier_seeds=_parse_int_limit(args.limit_classifier_seeds),
            ),
        )
        print(f"Wrote/resumed downstream matrix: {matrix_path}")

    if args.build_reports:
        _build_reports(
            artifacts_root,
            matrix_path=_report_matrix_path(
                args=args,
                artifacts_root=artifacts_root,
            ),
        )

    if not args.build_matrix and not args.build_reports:
        print(
            "Prepared support units. Add --build-matrix on the workstation to run "
            "synthetic generation/classifier scoring."
        )


def _build_reports(artifacts_root: Path, *, matrix_path: Path | None = None) -> None:
    support_path = artifacts_root / "tables" / "support_selection_units.csv"
    diagnostic_path = artifacts_root / "tables" / "diagnostic_downstream_utility.csv"
    selected_matrix_path = matrix_path or (
        diagnostic_path if diagnostic_path.exists() else artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    )
    assert_matrix_schema(selected_matrix_path)
    selections = support_units_from_csv(support_path)
    downstream_rows = read_candidate_downstream_matrix(selected_matrix_path)
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


def _classifier_specs_from_args(args: argparse.Namespace) -> tuple[ClassifierSpec, ...]:
    c_values = _parse_float_list(args.classifier_c_grid, "classifier-c-grid")
    penalties = _parse_str_list(args.classifier_penalties)
    solvers = _parse_str_list(args.classifier_solvers)
    class_weights = tuple(_parse_class_weight(value) for value in _parse_str_list(args.classifier_class_weights))
    max_iters = _parse_int_list(args.classifier_max_iters, "classifier-max-iters")
    l1_ratios = _parse_float_list(args.classifier_l1_ratios, "classifier-l1-ratios") if str(args.classifier_l1_ratios).strip() else ()
    specs: list[ClassifierSpec] = []
    for c_value in c_values:
        for penalty in penalties:
            for solver in solvers:
                for class_weight in class_weights:
                    for max_iter in max_iters:
                        if penalty == "elasticnet":
                            for l1_ratio in l1_ratios:
                                specs.append(
                                    ClassifierSpec(
                                        C=float(c_value),
                                        penalty=penalty,
                                        solver=solver,
                                        max_iter=int(max_iter),
                                        class_weight=class_weight,
                                        l1_ratio=float(l1_ratio),
                                    )
                                )
                        else:
                            specs.append(
                                ClassifierSpec(
                                    C=float(c_value),
                                    penalty=penalty,
                                    solver=solver,
                                    max_iter=int(max_iter),
                                    class_weight=class_weight,
                                )
                            )
    if not specs:
        raise ProtocolError("Source-inner classifier tuning grid is empty.")
    return tuple(specs)


def _report_matrix_path(*, args: argparse.Namespace, artifacts_root: Path) -> Path | None:
    if args.report_matrix_path:
        return Path(args.report_matrix_path)
    if args.matrix_path:
        return Path(args.matrix_path)
    if args.source_inner_classifier_tuning:
        grid_hash = classifier_grid_hash(_classifier_specs_from_args(args))
        return artifacts_root / "tables" / f"source_inner_classifier_tuned_{grid_hash}_downstream_matrix.csv"
    return None


def _parse_str_list(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if not values:
        raise ProtocolError("Expected at least one comma-separated value.")
    return values


def _parse_int_list(raw: str, label: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    except ValueError as exc:
        raise ProtocolError(f"Invalid integer in --{label}: {raw!r}") from exc
    if not values:
        raise ProtocolError(f"--{label} must contain at least one value.")
    return values


def _parse_float_list(raw: str, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    except ValueError as exc:
        raise ProtocolError(f"Invalid float in --{label}: {raw!r}") from exc
    if not values:
        raise ProtocolError(f"--{label} must contain at least one value.")
    return values


def _parse_class_weight(raw: str) -> str | None:
    value = str(raw).strip().lower()
    if value in {"none", "null", ""}:
        return None
    if value == "balanced":
        return "balanced"
    raise ProtocolError(f"Unsupported classifier class_weight: {raw!r}")


if __name__ == "__main__":
    try:
        main()
    except (ProtocolError, ArtifactSyncError) as exc:
        raise SystemExit(str(exc)) from exc
