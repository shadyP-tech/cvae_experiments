"""Run F2 source-anchored calibrated residual CVAE downstream diagnostic."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvae_downstream_evaluation.c41_workstation import (  # noqa: E402
    c41_training_profile_from_config,
    safe_support_selection_units_from_paths,
)
from cvae_downstream_evaluation.downstream import read_candidate_downstream_matrix  # noqa: E402
from cvae_downstream_evaluation.f2_calibrated_residual import (  # noqa: E402
    F2_ARTIFACTS_ROOT,
    F2_DEFAULT_C41_ROOT,
    F2_DEFAULT_C61_ROOT,
    F2_DEFAULT_F1_ROOT,
    build_f2_calibration_to_utility_join_rows,
    build_f2_delta_summary_rows,
    build_f2_downstream_matrix,
    build_f2_routing_alignment_rows,
    load_f2_diagnostics,
    write_f2_alignment_csv,
    write_f2_delta_summary_csv,
)
from cvae_downstream_evaluation.matrix import MatrixBuildLimits  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError, load_locked_v1_config  # noqa: E402
from cvae_downstream_evaluation.routing import support_units_from_csv, write_support_selection_units  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run F2 source-anchored calibrated residual CVAE downstream diagnostic."
    )
    parser.add_argument("--config", required=True, help="Path to locked downstream v1 YAML config.")
    parser.add_argument("--artifacts-root", default=F2_ARTIFACTS_ROOT, help="Output root for isolated F2 artifacts.")
    parser.add_argument("--c41-artifacts-root", default=F2_DEFAULT_C41_ROOT, help="Input C4.1 full artifact root.")
    parser.add_argument("--f1-artifacts-root", default=F2_DEFAULT_F1_ROOT, help="Input F1 artifact root for baseline reports.")
    parser.add_argument("--c61-artifacts-root", default=F2_DEFAULT_C61_ROOT, help="Input C6.1 artifact root for report context.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the execution plan.")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a small diagnostic run.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing F2 checkpoints/matrix rows.")
    parser.add_argument("--training-profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument(
        "--allow-legacy-audit-columns",
        action="store_true",
        help="Drop forbidden oracle/eval audit columns from legacy support-selection artifacts instead of failing.",
    )
    parser.add_argument("--limit-experiment-seeds", default=None, help="Comma-separated experiment seeds.")
    parser.add_argument("--limit-heldout-centers", default=None, help="Comma-separated heldout centers.")
    parser.add_argument("--limit-generation-seeds", default=None, help="Comma-separated generation seeds.")
    parser.add_argument("--limit-classifier-seeds", default=None, help="Comma-separated classifier seeds.")
    parser.add_argument("--build-reports-only", action="store_true", help="Rebuild F2 reports from existing matrix.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config_path = Path(args.config)
    config = load_locked_v1_config(config_path)
    c41_root = repo_root / str(args.c41_artifacts_root)
    f1_root = repo_root / str(args.f1_artifacts_root)
    c61_root = repo_root / str(args.c61_artifacts_root)
    _assert_inputs_exist(c41_root=c41_root, f1_root=f1_root)
    support_paths = [Path(path) for path in glob.glob(str(repo_root / config.support_selection_glob))]
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")
    support_units = safe_support_selection_units_from_paths(
        support_paths,
        strict_forbidden_columns=not bool(args.allow_legacy_audit_columns),
    )
    artifacts_root = repo_root / str(args.artifacts_root)
    profile_name = "smoke" if args.smoke else args.training_profile
    training_profile = c41_training_profile_from_config(config_path, profile=profile_name)
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
                    "experiment": "F2 source-anchored calibrated residual CVAE diagnostic",
                    "artifacts_root": str(artifacts_root),
                    "c41_artifacts_root": str(c41_root),
                    "f1_artifacts_root": str(f1_root),
                    "c61_artifacts_root": str(c61_root),
                    "support_selection_files": len(support_paths),
                    "support_selection_units": len(support_units),
                    "training_profile": training_profile.__dict__,
                    "limits": {
                        "experiment_seeds": limits.experiment_seeds,
                        "heldout_centers": limits.heldout_centers,
                        "generation_seeds": limits.generation_seeds,
                        "classifier_seeds": limits.classifier_seeds,
                    },
                    "routing_scores_recomputed_for_f2": 0,
                    "projection_source": "reused_c41_full_source_train_pca64",
                    "generation_conditioning": "source_train_residual_reference_posterior",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    support_units_path = artifacts_root / "tables" / "support_selection_units.csv"
    write_support_selection_units(support_units_path, support_units)
    print(f"Wrote F2-safe support units: {support_units_path}")

    if not args.build_reports_only:
        matrix_path = build_f2_downstream_matrix(
            config=config,
            repo_root=repo_root,
            artifacts_root=artifacts_root,
            c41_artifacts_root=c41_root,
            support_units=support_units,
            device=args.device,
            resume=bool(args.resume),
            training_profile=training_profile,
            limits=limits,
        )
        print(f"Wrote/resumed F2 downstream matrix: {matrix_path}")

    _build_reports(artifacts_root, f1_root, c61_root)


def _build_reports(artifacts_root: Path, f1_root: Path, c61_root: Path) -> None:
    support_path = artifacts_root / "tables" / "support_selection_units.csv"
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    f1_alignment_path = f1_root / "tables" / "routing_to_downstream_alignment.csv"
    selections = support_units_from_csv(support_path)
    downstream_rows = read_candidate_downstream_matrix(matrix_path)
    alignment_rows = build_f2_routing_alignment_rows(selections=selections, downstream_rows=downstream_rows)
    write_f2_alignment_csv(artifacts_root / "tables" / "routing_to_downstream_alignment.csv", alignment_rows)
    duplicate_rows = load_f2_diagnostics(artifacts_root / "tables" / "f2_duplicate_diagnostics.csv")
    residual_rows = load_f2_diagnostics(artifacts_root / "tables" / "f2_residual_moment_diagnostics.csv")
    f1_alignment = _read_csv_dicts(f1_alignment_path)
    delta_rows = build_f2_delta_summary_rows(
        f2_alignment_rows=alignment_rows,
        f1_alignment_rows=f1_alignment,
        duplicate_rows=duplicate_rows,
        residual_rows=residual_rows,
    )
    write_f2_delta_summary_csv(artifacts_root / "tables" / "f2_delta_summary.csv", delta_rows)
    write_f2_delta_summary_csv(artifacts_root / "tables" / "f2_generation_mode_comparison.csv", delta_rows)
    calibration_rows = load_f2_diagnostics(artifacts_root / "tables" / "f2_calibration_diagnostics.csv")
    calibration_join = build_f2_calibration_to_utility_join_rows(
        calibration_rows=calibration_rows,
        delta_rows=delta_rows,
    )
    _write_dict_csv(artifacts_root / "tables" / "f2_calibration_to_utility_join.csv", calibration_join)
    if c61_root.exists():
        _append_c61_context_if_available(
            f2_late_path=artifacts_root / "tables" / "f2_late_ensemble_summary.csv",
            c61_path=c61_root / "tables" / "c61_threshold_audit.csv",
        )
    print(f"Wrote F2 alignment and reports under: {artifacts_root / 'tables'}")


def _assert_inputs_exist(*, c41_root: Path, f1_root: Path) -> None:
    required = (
        c41_root / "tables" / "routing_to_downstream_alignment.csv",
        c41_root / "projections",
        f1_root / "tables" / "routing_to_downstream_alignment.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ProtocolError("F2 requires completed C4.1 and F1 artifacts:\n" + "\n".join(missing))


def _read_csv_dicts(path: Path) -> list[dict[str, object]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _append_c61_context_if_available(*, f2_late_path: Path, c61_path: Path) -> None:
    if not f2_late_path.exists() or not c61_path.exists():
        return
    rows = _read_csv_dicts(f2_late_path)
    if not rows:
        return
    c61_rows = _read_csv_dicts(c61_path)
    late = [
        row for row in c61_rows
        if str(row.get("mixture_policy")) == "late_classifier_ensemble_diagnostic_only"
    ]
    c61_mean = _mean(late, "mean_bacc") if late else float("nan")
    for row in rows:
        row["c61_late_ensemble_mean_bacc"] = c61_mean
        try:
            row["delta_vs_c61_late_ensemble"] = float(row["mean_bacc"]) - c61_mean
        except (TypeError, ValueError):
            row["delta_vs_c61_late_ensemble"] = float("nan")
    _write_dict_csv(f2_late_path, rows)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            value = float("nan")
        if value == value:
            values.append(value)
    return sum(values) / float(len(values)) if values else float("nan")


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
