"""Run MIDOG++ phase-1 scoring from exported source summaries and eval cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.adapters.midogpp_runner import (  # noqa: E402
    MidogppRunContext,
    run_midogpp_phase1_scoring,
    select_midogpp_source_inner_classifier_spec,
)
from cvae_downstream_evaluation.adapters.midogpp_source_summary_backend import (  # noqa: E402
    SourceSummaryMidogppBackend,
    build_midogpp_phase1_run_hashes,
    preflight_midogpp_external_baselines,
    preflight_midogpp_source_summary_inputs,
)
from cvae_downstream_evaluation.classifier_grid import (  # noqa: E402
    add_classifier_grid_arguments,
    classifier_specs_from_args,
    csv_values,
)
from cvae_downstream_evaluation.classifiers import ClassifierSpec, classifier_grid_hash  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp import (  # noqa: E402
    MIDOGPP_ELIGIBLE_CENTERS,
    assert_midogpp_frozen_config_file,
)
from cvae_downstream_evaluation.thresholding import fixed_threshold_spec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score MIDOG++ single-source candidates from exported source summary artifacts."
    )
    parser.add_argument("--summary-manifest", required=True)
    parser.add_argument("--test-cache-root", default=None)
    parser.add_argument("--test-cache-path", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--experiment-seed", type=int, default=42)
    parser.add_argument("--replicate-seed", type=int, default=0)
    parser.add_argument("--heldout-centers", default=",".join(MIDOGPP_ELIGIBLE_CENTERS))
    parser.add_argument("--synthetic-per-class-total", type=int, default=128)
    parser.add_argument("--generation-seed", type=int, default=17)
    parser.add_argument("--latent-sample-seed", type=int, default=17)
    parser.add_argument("--classifier-seed", type=int, default=23)
    parser.add_argument("--source-inner-classifier-tuning", action="store_true")
    parser.add_argument(
        "--threshold-policy",
        choices=("fixed_0_5", "source_inner_selected", "both"),
        default="fixed_0_5",
        help="Classifier decision policy for MIDOG++ phase-1 scoring.",
    )
    add_classifier_grid_arguments(parser)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--protocol-hash", default=None)
    parser.add_argument("--feature-frame-hash", default=None)
    parser.add_argument(
        "--baseline-matrix",
        action="append",
        default=[],
        help="Optional locked diagnostic baseline matrix CSV to include as method-baseline rows.",
    )
    parser.add_argument(
        "--baseline-method",
        action="append",
        default=[],
        help="Optional baseline method name to import from --baseline-matrix. Repeat for multiple methods.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "utility_matrix" / "virchow2_midogpp_all_candidates.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    if bool(args.baseline_matrix) != bool(args.baseline_method):
        raise SystemExit("--baseline-matrix and --baseline-method must be provided together.")
    if args.threshold_policy in {"source_inner_selected", "both"} and not args.source_inner_classifier_tuning:
        raise SystemExit("--threshold-policy source_inner_selected/both requires --source-inner-classifier-tuning.")
    assert_midogpp_frozen_config_file(Path(args.config))
    heldout_centers = csv_values(args.heldout_centers)
    try:
        preflight = preflight_midogpp_source_summary_inputs(
            summary_manifest=Path(args.summary_manifest),
            experiment_seeds=(int(args.experiment_seed),),
            heldout_centers=heldout_centers,
            test_cache_root=Path(args.test_cache_root) if args.test_cache_root else None,
            test_cache_path=Path(args.test_cache_path) if args.test_cache_path else None,
        )
    except ProtocolError as exc:
        _write_report(
            out_dir / "reports" / "source_summary_preflight_report.json",
            _failed_report(
                schema_version="midogpp_source_summary_preflight_report_v1",
                error_message=str(exc),
                summary_manifest=Path(args.summary_manifest),
                heldout_centers=heldout_centers,
            ),
        )
        raise
    _write_report(out_dir / "reports" / "source_summary_preflight_report.json", preflight.to_report())
    print(json.dumps(preflight.to_report(), indent=2, sort_keys=True))
    if args.baseline_matrix:
        try:
            baseline_preflight = preflight_midogpp_external_baselines(
                baseline_matrix_paths=tuple(Path(path) for path in args.baseline_matrix),
                baseline_methods=tuple(args.baseline_method),
                experiment_seed=int(args.experiment_seed),
                replicate_seed=int(args.replicate_seed),
                heldout_centers=heldout_centers,
            )
        except ProtocolError as exc:
            _write_report(
                out_dir / "reports" / "baseline_preflight_report.json",
                _failed_report(
                    schema_version="midogpp_baseline_preflight_report_v1",
                    error_message=str(exc),
                    baseline_matrix_paths=tuple(Path(path) for path in args.baseline_matrix),
                    baseline_methods=tuple(args.baseline_method),
                    heldout_centers=heldout_centers,
                ),
            )
            raise
        _write_report(out_dir / "reports" / "baseline_preflight_report.json", baseline_preflight.to_report())
        print(json.dumps(baseline_preflight.to_report(), indent=2, sort_keys=True))
    run_hashes = build_midogpp_phase1_run_hashes(
        config_path=Path(args.config),
        summary_manifest=Path(args.summary_manifest),
        preflight=preflight,
        heldout_centers=heldout_centers,
        experiment_seed=int(args.experiment_seed),
        replicate_seed=int(args.replicate_seed),
        synthetic_per_class_total=int(args.synthetic_per_class_total),
        generation_seed=int(args.generation_seed),
        latent_sample_seed=int(args.latent_sample_seed),
        classifier_seed=int(args.classifier_seed),
        out_dir=out_dir,
        classifier_config_payload=_classifier_config_payload(args),
    )
    _write_report(out_dir / "reports" / "run_hashes_report.json", run_hashes.to_report())
    print(json.dumps(run_hashes.to_report(), indent=2, sort_keys=True))
    config_hash = _resolve_hash_arg("config_hash", args.config_hash, run_hashes.config_hash)
    protocol_hash = _resolve_hash_arg("protocol_hash", args.protocol_hash, run_hashes.protocol_hash)
    feature_frame_hash = _resolve_hash_arg("feature_frame_hash", args.feature_frame_hash, run_hashes.feature_frame_hash)
    if args.preflight_only:
        return
    backend = SourceSummaryMidogppBackend(
        summary_manifest=Path(args.summary_manifest),
        test_cache_root=Path(args.test_cache_root) if args.test_cache_root else None,
        test_cache_path=Path(args.test_cache_path) if args.test_cache_path else None,
        baseline_matrix_paths=tuple(Path(path) for path in args.baseline_matrix),
    )
    contexts = [
        MidogppRunContext(
            heldout_center=center,
            experiment_seed=int(args.experiment_seed),
            replicate_seed=int(args.replicate_seed),
            support_size=0,
            support_seed="none",
            support_set_id="none",
            eval_set_id=f"midogpp_center_{center}_eval_all_no_support",
            generation_seed=int(args.generation_seed),
            latent_sample_seed=int(args.latent_sample_seed),
            classifier_seed=int(args.classifier_seed),
            synthetic_per_class_total=int(args.synthetic_per_class_total),
            config_hash=config_hash,
            protocol_hash=protocol_hash,
            feature_frame_hash=feature_frame_hash,
        )
        for center in heldout_centers
    ]
    candidates_by_heldout = {
        context.heldout_center: backend.candidates_for_context(context)
        for context in contexts
    }
    classifier_specs_by_context = {}
    threshold_decisions_by_context = {}
    if args.source_inner_classifier_tuning:
        candidate_specs = _classifier_specs_from_args(args)
        tuning_rows = []
        for context in contexts:
            selection = select_midogpp_source_inner_classifier_spec(
                backend=backend,
                outer_context=context,
                candidate_specs=candidate_specs,
            )
            classifier_specs_by_context[
                (int(context.experiment_seed), str(context.heldout_center), int(context.classifier_seed))
            ] = selection.selected_spec
            threshold_key = (int(context.experiment_seed), str(context.heldout_center), int(context.classifier_seed))
            threshold_decisions = []
            if args.threshold_policy in {"fixed_0_5", "both"}:
                threshold_decisions.append(
                    fixed_threshold_spec(
                        threshold_policy_group_id=selection.threshold_selection.decision.threshold_policy_group_id
                    )
                )
            if args.threshold_policy in {"source_inner_selected", "both"}:
                threshold_decisions.append(selection.threshold_selection.decision)
            threshold_decisions_by_context[threshold_key] = tuple(threshold_decisions)
            tuning_rows.extend(selection.to_artifact_rows(candidate_specs=candidate_specs))
        _write_csv(out_dir / "tables" / "source_inner_classifier_tuning.csv", tuning_rows)
    outputs = run_midogpp_phase1_scoring(
        backend=backend,
        contexts=contexts,
        candidates_by_heldout=candidates_by_heldout,
        artifacts_root=out_dir,
        baseline_methods=tuple(args.baseline_method),
        classifier_specs_by_context=classifier_specs_by_context,
        threshold_decisions_by_context=threshold_decisions_by_context,
    )
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


def _csv(raw: str) -> tuple[str, ...]:
    return csv_values(raw)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write empty MIDOG++ classifier tuning CSV: {path}")
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _failed_report(*, schema_version: str, error_message: str, **fields: object) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "status": "FAIL",
        "error_message": error_message,
        **{key: _jsonable(value) for key, value in fields.items()},
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _resolve_hash_arg(name: str, provided: object | None, generated: str) -> str:
    if provided in {None, ""}:
        return generated
    if str(provided) != str(generated):
        raise ProtocolError(f"Provided {name}={provided!r} does not match generated frozen value {generated!r}.")
    return str(generated)


def _classifier_config_payload(args: argparse.Namespace) -> dict[str, object]:
    if not args.source_inner_classifier_tuning:
        return {
            "family": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": None,
            "scaler_fit": "synthetic_train_only",
            "hyperparameter_tuning": "forbidden",
            "classifier_seed": int(args.classifier_seed),
            "threshold_policy": str(args.threshold_policy),
            "threshold_value": 0.5,
        }
    specs = _classifier_specs_from_args(args)
    return {
        "family": "sklearn_logistic_regression",
        "hyperparameter_tuning": "source_inner_lodo",
        "selection_metric": "bacc",
        "grid_hash": classifier_grid_hash(specs),
        "grid": [spec.to_payload() for spec in specs],
        "classifier_seed": int(args.classifier_seed),
        "outer_target_labels_used": False,
        "threshold_policy": str(args.threshold_policy),
        "threshold_rule": "fixed_0_5" if args.threshold_policy == "fixed_0_5" else "source_inner_lodo",
    }


def _classifier_specs_from_args(args: argparse.Namespace) -> tuple[ClassifierSpec, ...]:
    return classifier_specs_from_args(args)


if __name__ == "__main__":
    try:
        main()
    except ProtocolError as exc:
        raise SystemExit(str(exc)) from exc
