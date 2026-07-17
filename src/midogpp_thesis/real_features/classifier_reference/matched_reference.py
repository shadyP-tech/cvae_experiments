"""Eligible-only predict-policy tuned real-feature reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .artifacts import prepare_artifact_dirs, stable_hash, write_csv_rows, write_json
from .classifier_grid import build_classifier_specs
from .classifiers import ClassifierSpec, classifier_grid_hash, fit_logistic_classifier
from .downstream import balanced_accuracy, macro_f1
from .real_feature_frame import (
    RealFeatureFrame,
    load_midogpp_real_feature_frame,
)
from .protocol import ProtocolError
from .source_inner_classifier_tuning import (
    SourceInnerClassifierFold,
    select_classifier_spec_source_inner_lodo,
)
from .schemas.matched_reference import (
    MATCHED_REFERENCE_METHOD,
    MATCHED_REFERENCE_PREDICTION_COLUMNS,
    MATCHED_REFERENCE_PREDICTION_SCHEMA_VERSION,
    MATCHED_REFERENCE_RESULT_COLUMNS,
    MATCHED_REFERENCE_RESULT_SCHEMA_VERSION,
    MATCHED_REFERENCE_SCHEMA_VERSION,
    assert_matched_reference_artifacts,
    matched_reference_bundle_hash,
)
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS


CANONICAL_GRID_HASH = "5abd0897d02bdcaa"
CANONICAL_GRID_SIZE = 10


@dataclass(frozen=True)
class MatchedReferenceConfig:
    name: str
    artifact_root: Path
    manifest_path: Path
    feature_cache_path: Path
    heldout_centers: tuple[str, ...]
    experiment_seed: int = 42
    classifier_seed: int = 23
    expected_feature_dim: int = 2560
    classifier_specs: tuple[ClassifierSpec, ...] = ()
    allow_partial_test_coverage: bool = False


@dataclass(frozen=True)
class PredictOnlySelection:
    outer_target_center: str
    inner_pseudo_target_center: str | None
    selected_spec: ClassifierSpec
    grid_hash: str
    center_scores: Mapping[str, float]
    candidate_rows: tuple[Mapping[str, object], ...]


def canonical_matched_reference_specs(*, classifier_seed: int = 23) -> tuple[ClassifierSpec, ...]:
    specs = build_classifier_specs(
        c_grid="0.01,0.1,1.0,10.0,100.0",
        penalties="l2",
        solvers="lbfgs",
        class_weights="none,balanced",
        max_iters="5000",
        classifier_seed=int(classifier_seed),
    )
    grid_hash = classifier_grid_hash(specs)
    if int(classifier_seed) == 23 and grid_hash != CANONICAL_GRID_HASH:
        raise ProtocolError(
            f"Canonical Stage-10 grid hash drift: expected={CANONICAL_GRID_HASH} actual={grid_hash}"
        )
    return specs


def select_outer_predict_spec(
    frame: RealFeatureFrame,
    *,
    outer_target_center: str,
    candidate_specs: Sequence[ClassifierSpec],
) -> PredictOnlySelection:
    outer = _eligible_present(frame, outer_target_center)
    validation_centers = tuple(center for center in frame.eligible_centers if center != outer)
    return _select_predict_spec(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=None,
        validation_centers=validation_centers,
        excluded_centers=(outer,),
        candidate_specs=candidate_specs,
    )


def select_nested_predict_spec(
    frame: RealFeatureFrame,
    *,
    outer_target_center: str,
    inner_pseudo_target_center: str,
    candidate_specs: Sequence[ClassifierSpec],
) -> PredictOnlySelection:
    outer = _eligible_present(frame, outer_target_center)
    inner = _eligible_present(frame, inner_pseudo_target_center)
    if outer == inner:
        raise ProtocolError("Nested real reference requires different outer and inner centers.")
    validation_centers = tuple(center for center in frame.eligible_centers if center not in {outer, inner})
    if not validation_centers:
        raise ProtocolError("Nested real reference requires at least one deeper validation center.")
    return _select_predict_spec(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        validation_centers=validation_centers,
        excluded_centers=(outer, inner),
        candidate_specs=candidate_specs,
    )


def select_nested_predict_spec_source_only(
    frame: RealFeatureFrame,
    *,
    outer_target_center: str,
    inner_pseudo_target_center: str,
    candidate_specs: Sequence[ClassifierSpec],
) -> PredictOnlySelection:
    """Select on a frame from which the outer target rows were removed."""

    outer = _eligible_center(outer_target_center)
    if outer in frame.eligible_centers:
        raise ProtocolError("Source-only nested selection received outer target rows.")
    inner = _eligible_present(frame, inner_pseudo_target_center)
    validation_centers = tuple(center for center in frame.eligible_centers if center != inner)
    if not validation_centers:
        raise ProtocolError("Nested real reference requires at least one deeper validation center.")
    return _select_predict_spec(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        validation_centers=validation_centers,
        excluded_centers=(outer, inner),
        candidate_specs=candidate_specs,
    )


def run_matched_reference(config: MatchedReferenceConfig, *, artifact_root: Path | None = None) -> Path:
    root = prepare_artifact_dirs(artifact_root or config.artifact_root)
    frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.feature_cache_path,
        expected_feature_dim=config.expected_feature_dim,
    )
    specs = config.classifier_specs or canonical_matched_reference_specs(classifier_seed=config.classifier_seed)
    if len(specs) != CANONICAL_GRID_SIZE or classifier_grid_hash(specs) != CANONICAL_GRID_HASH:
        raise ProtocolError("Matched reference requires the exact canonical 10-spec classifier grid.")
    heldouts = tuple(_eligible_present(frame, center) for center in config.heldout_centers)
    coverage_mode = "partial_test" if config.allow_partial_test_coverage else "complete"
    if not config.allow_partial_test_coverage:
        if frame.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS or heldouts != MIDOGPP_ELIGIBLE_CENTERS:
            raise ProtocolError("Production matched reference requires exact nine-center eligible coverage.")
    protocol_payload = {
        "schema_version": MATCHED_REFERENCE_SCHEMA_VERSION,
        "experiment_name": config.name,
        "experiment_seed": config.experiment_seed,
        "classifier_seed": config.classifier_seed,
        "classifier_grid_hash": classifier_grid_hash(specs),
        "classifier_grid": [spec.to_payload() for spec in specs],
        "heldout_centers": list(heldouts),
        "eligible_centers": list(frame.eligible_centers),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "coverage_mode": coverage_mode,
        "manifest_hash": frame.manifest_hash,
        "feature_cache_hash": frame.feature_cache_hash,
        "method": MATCHED_REFERENCE_METHOD,
        "threshold_policy": "predict",
        "claim_scope": "real_feature_transfer_only",
        "claim_role": "real_feature_reference",
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "generated_embeddings_used": False,
        "cvae_checkpoint_used": False,
        "source_summary_manifest_used": False,
        "is_router": False,
        "probabilities_calibrated": False,
        "support_labels_used": False,
        "oracle_eligible": False,
    }
    protocol_payload["protocol_hash"] = stable_hash(protocol_payload)
    protocol_hash = str(protocol_payload["protocol_hash"])
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    for heldout in heldouts:
        selection = select_outer_predict_spec(frame, outer_target_center=heldout, candidate_specs=specs)
        tuning_rows.extend(selection.candidate_rows)
        train_centers = tuple(center for center in frame.eligible_centers if center != heldout)
        train_idx = _indices(frame, train_centers)
        eval_idx = _indices(frame, (heldout,))
        x_train, y_train = _arrays(frame, train_idx)
        x_eval, y_eval = _arrays(frame, eval_idx)
        fitted = fit_logistic_classifier(x_train, y_train, x_eval, spec=selection.selected_spec)
        if not fitted.converged:
            raise ProtocolError(f"Matched-reference final classifier did not converge for center {heldout}.")
        predictions = tuple(int(value) for value in fitted.predictions.tolist())
        probabilities = tuple(float(row[1]) for row in fitted.probabilities.tolist())
        fit_ids = [frame.rows[index].sample_id for index in train_idx]
        eval_ids = [frame.rows[index].sample_id for index in eval_idx]
        fit_row_hash = _row_hash(fit_ids)
        eval_row_hash = _row_hash(eval_ids)
        mean_inner = sum(selection.center_scores.values()) / float(len(selection.center_scores))
        result_rows.append(
            {
                "schema_version": MATCHED_REFERENCE_RESULT_SCHEMA_VERSION,
                "method": MATCHED_REFERENCE_METHOD,
                "protocol_hash": protocol_hash,
                "experiment_seed": config.experiment_seed,
                "classifier_seed": config.classifier_seed,
                "heldout_center": heldout,
                "train_centers": json.dumps(list(train_centers)),
                "n_train": len(train_idx),
                "n_eval": len(eval_idx),
                "fit_row_hash": fit_row_hash,
                "eval_row_hash": eval_row_hash,
                "classifier_grid_hash": selection.grid_hash,
                "selected_classifier_config_hash": selection.selected_spec.config_hash,
                "selected_classifier_spec": json.dumps(selection.selected_spec.to_payload(), sort_keys=True),
                "selection_metric": "bacc",
                "selection_source": "source_inner_lodo_predict",
                "source_inner_center_bacc_vector": json.dumps(dict(selection.center_scores), sort_keys=True),
                "source_inner_mean_bacc": mean_inner,
                "heldout_bacc": balanced_accuracy(y_eval, predictions),
                "heldout_macro_f1": macro_f1(y_eval, predictions),
                "converged": str(fitted.converged).lower(),
                "n_iter": json.dumps(list(fitted.n_iter)),
                "status": "ok",
                "feature_cache_hash": frame.feature_cache_hash,
                "manifest_hash": frame.manifest_hash,
                "target_eval_labels_used_for_scoring_only": "true",
                "selection_used_target_labels": "false",
                "fit_used_target_center": "false",
                "generated_embeddings_used": "false",
                "cvae_checkpoint_used": "false",
                "source_summary_manifest_used": "false",
                "is_router": "false",
                "claim_scope": "real_feature_transfer_only",
                "claim_role": "real_feature_reference",
                "row_role": "heldout_result",
                "leakage_status": "PASS",
                "support_labels_used": "false",
                "oracle_eligible": "false",
                "probabilities_calibrated": "false",
            }
        )
        for local_index, row_index in enumerate(eval_idx):
            row = frame.rows[row_index]
            prediction_rows.append(
                {
                    "schema_version": MATCHED_REFERENCE_PREDICTION_SCHEMA_VERSION,
                    "method": MATCHED_REFERENCE_METHOD,
                    "protocol_hash": protocol_hash,
                    "heldout_center": heldout,
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "center": row.center,
                    "y_true": row.label,
                    "y_pred": predictions[local_index],
                    "prob_pos": probabilities[local_index],
                    "selected_classifier_config_hash": selection.selected_spec.config_hash,
                    "eval_row_hash": eval_row_hash,
                    "claim_role": "real_feature_reference",
                    "row_role": "heldout_prediction",
                    "leakage_status": "PASS",
                    "support_labels_used": "false",
                    "oracle_eligible": "false",
                    "target_eval_labels_used_for_scoring_only": "true",
                }
            )
        leakage_rows.append(
            {
                "outer_target_center": heldout,
                "fit_centers": list(train_centers),
                "target_center_excluded_from_fit": heldout not in train_centers,
                "quarantined_center_excluded": not set(train_centers).intersection(MIDOGPP_EXCLUDED_CENTERS),
            }
        )
    protocol_payload.pop("protocol_hash", None)
    protocol_payload["reference_bundle_hash"] = matched_reference_bundle_hash(
        tuning_rows,
        result_rows,
        prediction_rows,
    )
    protocol_payload["protocol_hash"] = stable_hash(protocol_payload)
    protocol_hash = str(protocol_payload["protocol_hash"])
    for row in (*result_rows, *prediction_rows):
        row["protocol_hash"] = protocol_hash
    write_csv_rows(root / "tables/source_inner_classifier_tuning.csv", tuning_rows)
    write_csv_rows(root / "tables/classifier_tuned_source_results.csv", result_rows, MATCHED_REFERENCE_RESULT_COLUMNS)
    write_csv_rows(root / "tables/classifier_tuned_predictions.csv", prediction_rows, MATCHED_REFERENCE_PREDICTION_COLUMNS)
    write_json(root / "manifests/protocol_manifest.json", protocol_payload)
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            **protocol_payload,
            "status": "PASS",
            "leakage_status": "PASS",
            "target_labels_used_for_scoring_only": True,
            "target_metric_used_for_selection": False,
            "overlap_rows": leakage_rows,
        },
    )
    assert_matched_reference_artifacts(root)
    return root


def load_matched_reference_config(path: str | Path) -> MatchedReferenceConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Matched-reference configs require PyYAML.") from exc
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("Matched-reference config must be a mapping.")
    experiment = _mapping(payload.get("experiment"), "experiment")
    inputs = _mapping(payload.get("inputs"), "inputs")
    run = _mapping(payload.get("run", {}), "run")
    grid = _mapping(payload.get("classifier_grid"), "classifier_grid")
    heldouts_raw = run.get("heldout_centers", "all")
    heldouts = (
        MIDOGPP_ELIGIBLE_CENTERS
        if str(heldouts_raw).lower() == "all"
        else tuple(str(value) for value in heldouts_raw)
    )
    base = config_path.parent
    specs = build_classifier_specs(
        c_grid=_csv(grid.get("c_grid")),
        penalties=_csv(grid.get("penalty")),
        solvers=_csv(grid.get("solver")),
        class_weights=_csv(grid.get("class_weight")),
        max_iters=_csv(grid.get("max_iter")),
        classifier_seed=int(run.get("classifier_seed", 23)),
    )
    if int(grid.get("expected_candidate_count", -1)) != len(specs):
        raise ProtocolError("Configured matched-reference candidate count does not match the grid.")
    if str(grid.get("expected_grid_hash", "")) != classifier_grid_hash(specs):
        raise ProtocolError("Configured matched-reference grid hash does not match its payload.")
    if str(grid.get("threshold_policy", "")) != "predict":
        raise ProtocolError("Matched-reference threshold_policy must remain predict.")
    config = MatchedReferenceConfig(
        name=str(experiment.get("name", "eligible_tuned_real_reference_v2")),
        artifact_root=_resolve_path(base, str(experiment["artifact_root"])),
        manifest_path=_resolve_path(base, str(inputs["manifest_path"])),
        feature_cache_path=_resolve_path(base, str(inputs["feature_cache_path"])),
        heldout_centers=tuple(heldouts),
        experiment_seed=int(run.get("experiment_seed", 42)),
        classifier_seed=int(run.get("classifier_seed", 23)),
        expected_feature_dim=int(run.get("expected_feature_dim", 2560)),
        classifier_specs=specs,
    )
    if config.heldout_centers != MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError("Production matched-reference config must declare all eligible centers.")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_matched_reference_config(args.config)
    output = run_matched_reference(
        config,
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
    )
    print(output)
    return 0


def _select_predict_spec(
    frame: RealFeatureFrame,
    *,
    outer_target_center: str,
    inner_pseudo_target_center: str | None,
    validation_centers: Sequence[str],
    excluded_centers: Sequence[str],
    candidate_specs: Sequence[ClassifierSpec],
) -> PredictOnlySelection:
    if not candidate_specs:
        raise ProtocolError("Predict-only selection requires classifier candidates.")
    folds = []
    for validation_center in validation_centers:
        train_centers = tuple(
            center
            for center in frame.eligible_centers
            if center not in set(excluded_centers).union({validation_center})
        )
        if set(train_centers).intersection(excluded_centers):
            raise ProtocolError("Nested classifier selection leaked an excluded center.")
        x_train, y_train = _arrays(frame, _indices(frame, train_centers))
        x_validation, y_validation = _arrays(frame, _indices(frame, (validation_center,)))
        folds.append(
            SourceInnerClassifierFold(
                pseudo_target_center=validation_center,
                train_centers=train_centers,
                train_embeddings=x_train,
                train_labels=y_train,
                validation_embeddings=x_validation,
                validation_labels=y_validation,
            )
        )
    selection = select_classifier_spec_source_inner_lodo(
        outer_target_center=outer_target_center,
        folds=tuple(folds),
        candidate_specs=candidate_specs,
        classifier_seed=int(candidate_specs[0].random_state),
        selection_metric="bacc",
        reject_non_converged=True,
    )
    rows = tuple(
        {
            "schema_version": "midogpp_eligible_predict_spec_selection_v2",
            "outer_target_center": outer_target_center,
            "inner_pseudo_target_center": inner_pseudo_target_center or "",
            "deeper_validation_centers": json.dumps(list(validation_centers)),
            "excluded_centers": json.dumps(list(excluded_centers)),
            "classifier_grid_hash": selection.grid_hash,
            "classifier_spec": json.dumps(row.classifier_spec.to_payload(), sort_keys=True),
            "classifier_config_hash": row.classifier_spec.config_hash,
            "center_bacc_vector": json.dumps(dict(row.source_inner_lodo_center_bacc_vector), sort_keys=True),
            "aggregate_bacc": row.aggregate_score,
            "convergence_by_center": json.dumps(dict(row.convergence_by_center), sort_keys=True),
            "selection_source": "nested_source_inner_predict",
            "selection_used_target_labels": "false",
            "fit_used_outer_target_center": "false",
            "fit_used_inner_pseudo_target_center": "false",
            "selected": str(row.selected_by_source_inner_lodo).lower(),
        }
        for row in selection.rows
    )
    selected_row = next(row for row in selection.rows if row.selected_by_source_inner_lodo)
    return PredictOnlySelection(
        outer_target_center=outer_target_center,
        inner_pseudo_target_center=inner_pseudo_target_center,
        selected_spec=selection.selected_spec,
        grid_hash=selection.grid_hash,
        center_scores=dict(selected_row.source_inner_lodo_center_bacc_vector),
        candidate_rows=rows,
    )


def _arrays(frame: RealFeatureFrame, indices: Sequence[int]) -> tuple[object, tuple[int, ...]]:
    import numpy as np

    embeddings = frame.embeddings.detach().cpu().numpy() if hasattr(frame.embeddings, "detach") else frame.embeddings
    labels = tuple(int(frame.rows[index].label) for index in indices)
    if sorted(set(labels)) != [0, 1]:
        raise ProtocolError("Matched-reference fold must contain both classes.")
    return np.asarray(embeddings, dtype=float)[list(indices)], labels


def _row_hash(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in sample_ids).encode("utf-8")
    ).hexdigest()


def _indices(frame: RealFeatureFrame, centers: Sequence[str]) -> tuple[int, ...]:
    center_set = {str(center) for center in centers}
    if center_set.intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError("Quarantined center cannot enter matched-reference fitting.")
    return tuple(index for index, row in enumerate(frame.rows) if row.center in center_set)


def _eligible_present(frame: RealFeatureFrame, center: str) -> str:
    value = _eligible_center(center)
    if value not in frame.eligible_centers:
        raise ProtocolError(f"Unknown, quarantined, or absent MIDOG++ center: {value!r}")
    return value


def _eligible_center(center: str) -> str:
    value = str(center)
    if value not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown or quarantined MIDOG++ center: {value!r}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping.")
    return value


def _resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _csv(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join("none" if item is None else str(item) for item in value)
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
