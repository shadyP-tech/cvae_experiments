"""MIDOG++ real-feature source-only classifier reference runner."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import stable_hash
from .classifier_grid import build_classifier_specs
from .classifiers import ClassifierSpec, DEFAULT_LOCKED_CLASSIFIER_SPEC, classifier_grid_hash, fit_logistic_classifier
from .downstream import balanced_accuracy, macro_f1
from .protocol import ProtocolError
from .real_feature_frame import (
    RealFeatureFrame,
    RealFeatureRow,
    load_midogpp_real_feature_frame,
)
from .schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS, MIDOGPP_EXCLUDED_CENTERS
from .schemas.midogpp_real_feature_classifier import (
    REAL_FEATURE_PREDICTION_COLUMNS,
    REAL_FEATURE_PREDICTIONS_SCHEMA_VERSION,
    REAL_FEATURE_REFERENCE_SCHEMA_VERSION,
    REAL_FEATURE_RESULT_COLUMNS,
    REAL_FEATURE_RESULTS_SCHEMA_VERSION,
    assert_midogpp_real_feature_artifacts,
)
from .source_inner_classifier_tuning import (
    ScoreFn,
    SourceInnerClassifierFold,
    assert_source_inner_classifier_artifacts,
    select_classifier_spec_source_inner_lodo,
)
from .thresholding import (
    ThresholdDecisionSpec,
    apply_threshold,
    artifact_fields_for_decision,
    fixed_threshold_spec,
)


ScoreFnFactory = Callable[[str], ScoreFn | None]


@dataclass(frozen=True)
class RealFeatureRunPaths:
    source_inner_tuning: Path
    results: Path
    predictions: Path
    protocol_manifest: Path
    leakage_report: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "source_inner_tuning": self.source_inner_tuning,
            "results": self.results,
            "predictions": self.predictions,
            "protocol_manifest": self.protocol_manifest,
            "leakage_report": self.leakage_report,
        }


def run_midogpp_real_feature_source_inner_classifier_tuning(
    *,
    manifest_path: Path,
    feature_cache_path: Path,
    out_dir: Path,
    candidate_specs: Sequence[ClassifierSpec],
    heldout_centers: Sequence[str] = MIDOGPP_ELIGIBLE_CENTERS,
    experiment_seed: int = 42,
    classifier_seed: int = 23,
    expected_feature_dim: int = 2560,
    selection_metric: str = "bacc",
    score_fn_factory: ScoreFnFactory | None = None,
) -> RealFeatureRunPaths:
    """Run source-inner classifier tuning directly on real Virchow2 features."""

    frame = load_midogpp_real_feature_frame(
        manifest_path=manifest_path,
        feature_cache_path=feature_cache_path,
        expected_feature_dim=int(expected_feature_dim),
    )
    heldouts = _validate_heldout_centers(heldout_centers, frame=frame)
    grid_hash = classifier_grid_hash(candidate_specs)
    tuning_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    leakage_folds: list[dict[str, object]] = []
    protocol_hash_payload = {
        "schema_version": REAL_FEATURE_REFERENCE_SCHEMA_VERSION,
        "experiment_seed": int(experiment_seed),
        "classifier_seed": int(classifier_seed),
        "classifier_grid_hash": grid_hash,
        "classifier_grid": [spec.to_payload() for spec in candidate_specs],
        "feature_cache_hash": frame.feature_cache_hash,
        "manifest_hash": frame.manifest_hash,
        "heldout_centers": list(heldouts),
        "eligible_centers": list(MIDOGPP_ELIGIBLE_CENTERS),
        "excluded_centers": list(MIDOGPP_EXCLUDED_CENTERS),
        "selection_metric": selection_metric,
        "threshold_variants": ["fixed_0_5", "source_inner_selected"],
    }
    protocol_hash = stable_hash(protocol_hash_payload)
    for heldout in heldouts:
        folds, fold_audit = build_source_inner_folds(frame, outer_target_center=heldout)
        selection = select_classifier_spec_source_inner_lodo(
            outer_target_center=heldout,
            folds=folds,
            candidate_specs=candidate_specs,
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            selection_metric=selection_metric,
            score_fn=score_fn_factory(heldout) if score_fn_factory else None,
            reject_non_converged=True,
        )
        selection_artifact_rows = [
            _augment_tuning_row(
                row,
                frame=frame,
                protocol_hash=protocol_hash,
                fold_audit=fold_audit,
            )
            for row in selection.to_artifact_rows()
        ]
        tuning_rows.extend(selection_artifact_rows)
        tuned_eval = fit_and_score_heldout(
            frame,
            heldout_center=heldout,
            spec=selection.selected_spec,
            method="source_inner_tuned_fixed_0_5",
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            grid_hash=grid_hash,
            source_inner_rows=selection_artifact_rows,
            threshold_decision=fixed_threshold_spec(
                threshold_policy_group_id=selection.threshold_selection.decision.threshold_policy_group_id
            ),
        )
        result_rows.append(tuned_eval["result"])
        prediction_rows.extend(tuned_eval["predictions"])
        tuned_threshold_eval = fit_and_score_heldout(
            frame,
            heldout_center=heldout,
            spec=selection.selected_spec,
            method="source_inner_tuned_source_inner_threshold",
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            grid_hash=grid_hash,
            source_inner_rows=selection_artifact_rows,
            threshold_decision=selection.threshold_selection.decision,
        )
        result_rows.append(tuned_threshold_eval["result"])
        prediction_rows.extend(tuned_threshold_eval["predictions"])
        default_spec = _default_spec_for_seed(classifier_seed)
        default_threshold_selection = select_classifier_spec_source_inner_lodo(
            outer_target_center=heldout,
            folds=folds,
            candidate_specs=(default_spec,),
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            selection_metric=selection_metric,
            score_fn=score_fn_factory(heldout) if score_fn_factory else None,
            reject_non_converged=True,
        )
        default_eval = fit_and_score_heldout(
            frame,
            heldout_center=heldout,
            spec=default_spec,
            method="default_untuned_fixed_0_5",
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            grid_hash=grid_hash,
            source_inner_rows=(),
            threshold_decision=fixed_threshold_spec(
                threshold_policy_group_id=default_threshold_selection.threshold_selection.decision.threshold_policy_group_id
            ),
        )
        result_rows.append(default_eval["result"])
        prediction_rows.extend(default_eval["predictions"])
        default_threshold_eval = fit_and_score_heldout(
            frame,
            heldout_center=heldout,
            spec=default_spec,
            method="default_untuned_source_inner_threshold",
            experiment_seed=int(experiment_seed),
            classifier_seed=int(classifier_seed),
            grid_hash=grid_hash,
            source_inner_rows=(),
            threshold_decision=default_threshold_selection.threshold_selection.decision,
        )
        result_rows.append(default_threshold_eval["result"])
        prediction_rows.extend(default_threshold_eval["predictions"])
        leakage_folds.extend(_outer_and_inner_leakage_rows(frame, heldout_center=heldout, fold_audit=fold_audit))
    paths = _write_real_feature_artifacts(
        out_dir=Path(out_dir),
        frame=frame,
        tuning_rows=tuning_rows,
        result_rows=result_rows,
        prediction_rows=prediction_rows,
        leakage_folds=leakage_folds,
        protocol_hash=protocol_hash,
        protocol_hash_payload=protocol_hash_payload,
    )
    assert_midogpp_real_feature_artifacts(Path(out_dir))
    return paths


def build_source_inner_folds(
    frame: RealFeatureFrame,
    *,
    outer_target_center: str,
) -> tuple[tuple[SourceInnerClassifierFold, ...], dict[str, dict[str, object]]]:
    """Build source-inner LODO folds for one outer heldout center."""

    heldout = str(outer_target_center)
    if heldout not in MIDOGPP_ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ outer heldout center: {heldout!r}")
    if heldout not in frame.eligible_centers:
        raise ProtocolError(f"Heldout center {heldout!r} is absent from the real-feature frame.")
    folds: list[SourceInnerClassifierFold] = []
    audit: dict[str, dict[str, object]] = {}
    for pseudo_target in frame.eligible_centers:
        if pseudo_target == heldout:
            continue
        train_centers = tuple(center for center in frame.eligible_centers if center not in {heldout, pseudo_target})
        train_idx = _indices_for_centers(frame, train_centers)
        val_idx = _indices_for_centers(frame, (pseudo_target,))
        train_labels = _labels_for_indices(frame, train_idx)
        val_labels = _labels_for_indices(frame, val_idx)
        _assert_binary_class_support(train_labels, f"source-inner train H={heldout} P={pseudo_target}")
        _assert_binary_class_support(val_labels, f"source-inner validation H={heldout} P={pseudo_target}")
        folds.append(
            SourceInnerClassifierFold(
                pseudo_target_center=pseudo_target,
                train_centers=train_centers,
                train_embeddings=_embeddings_for_indices(frame, train_idx),
                train_labels=train_labels,
                validation_embeddings=_embeddings_for_indices(frame, val_idx),
                validation_labels=val_labels,
            )
        )
        audit[pseudo_target] = {
            "train_centers": list(train_centers),
            "validation_center": pseudo_target,
            "train_class_counts": _class_counts(train_labels),
            "validation_class_counts": _class_counts(val_labels),
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
        }
    return tuple(folds), audit


def fit_and_score_heldout(
    frame: RealFeatureFrame,
    *,
    heldout_center: str,
    spec: ClassifierSpec,
    method: str,
    experiment_seed: int,
    classifier_seed: int,
    grid_hash: str,
    source_inner_rows: Sequence[Mapping[str, object]],
    threshold_decision: ThresholdDecisionSpec,
) -> dict[str, object]:
    """Fit a frozen classifier on source centers and score the heldout center."""

    heldout = str(heldout_center)
    train_centers = tuple(center for center in frame.eligible_centers if center != heldout)
    train_idx = _indices_for_centers(frame, train_centers)
    eval_idx = _indices_for_centers(frame, (heldout,))
    train_labels = _labels_for_indices(frame, train_idx)
    eval_labels = _labels_for_indices(frame, eval_idx)
    _assert_binary_class_support(train_labels, f"final train H={heldout}")
    _assert_binary_class_support(eval_labels, f"final eval H={heldout}")
    fitted = fit_logistic_classifier(
        _embeddings_for_indices(frame, train_idx),
        train_labels,
        _embeddings_for_indices(frame, eval_idx),
        spec=spec,
    )
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("MIDOG++ real-feature scoring requires numpy.") from exc
    probabilities = np.asarray(fitted.probabilities, dtype=float)
    predictions = apply_threshold(probabilities[:, 1].tolist(), threshold_decision.threshold_value)
    classes = tuple(int(value) for value in fitted.classes)
    if classes != (0, 1):
        raise ProtocolError("MIDOG++ real-feature classifier expects classes (0, 1).")
    selected_vectors = _source_inner_summary(source_inner_rows)
    result_row = _result_row(
        method=method,
        experiment_seed=experiment_seed,
        classifier_seed=classifier_seed,
        heldout_center=heldout,
        train_centers=train_centers,
        n_train=len(train_idx),
        n_eval=len(eval_idx),
        grid_hash=grid_hash,
        spec=spec,
        source_inner_summary=selected_vectors,
        threshold_decision=threshold_decision,
        heldout_bacc=balanced_accuracy(eval_labels, predictions),
        heldout_macro_f1=macro_f1(eval_labels, predictions),
        converged=fitted.converged,
        n_iter=fitted.n_iter,
        feature_cache_hash=frame.feature_cache_hash,
        manifest_hash=frame.manifest_hash,
    )
    prediction_rows = []
    for local_idx, row_index in enumerate(eval_idx):
        row = frame.rows[row_index]
        prediction_rows.append(
            _prediction_row(
                method=method,
                experiment_seed=experiment_seed,
                classifier_seed=classifier_seed,
                heldout_center=heldout,
                sample=row,
                y_pred=int(predictions[local_idx]),
                prob_pos=float(probabilities[local_idx, 1]),
                spec=spec,
                threshold_decision=threshold_decision,
                feature_cache_hash=frame.feature_cache_hash,
                manifest_hash=frame.manifest_hash,
            )
        )
    return {"result": result_row, "predictions": prediction_rows}


def validate_real_feature_outputs(root: Path) -> None:
    assert_midogpp_real_feature_artifacts(Path(root))


def _write_real_feature_artifacts(
    *,
    out_dir: Path,
    frame: RealFeatureFrame,
    tuning_rows: list[dict[str, object]],
    result_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    leakage_folds: list[dict[str, object]],
    protocol_hash: str,
    protocol_hash_payload: Mapping[str, object],
) -> RealFeatureRunPaths:
    tables_dir = out_dir / "tables"
    manifests_dir = out_dir / "manifests"
    reports_dir = out_dir / "reports"
    paths = RealFeatureRunPaths(
        source_inner_tuning=tables_dir / "source_inner_classifier_tuning.csv",
        results=tables_dir / "classifier_tuned_source_results.csv",
        predictions=tables_dir / "classifier_tuned_predictions.csv",
        protocol_manifest=manifests_dir / "protocol_manifest.json",
        leakage_report=reports_dir / "leakage_provenance_report.json",
    )
    assert_source_inner_classifier_artifacts(tuning_rows)
    _write_csv(paths.source_inner_tuning, tuning_rows)
    _write_csv(paths.results, result_rows, columns=REAL_FEATURE_RESULT_COLUMNS)
    _write_csv(paths.predictions, prediction_rows, columns=REAL_FEATURE_PREDICTION_COLUMNS)
    protocol_manifest = {
        **dict(protocol_hash_payload),
        "protocol_hash": protocol_hash,
        "artifact_identity": "real_feature_source_only_classifier_reference",
        "method": "real_feature_source_only_classifier_reference",
        "feature_cache_path": str(frame.feature_cache_path),
        "manifest_path": str(frame.manifest_path),
        "expected_feature_dim": int(frame.expected_feature_dim),
        "is_router": False,
        "claim_scope": "real_feature_transfer_only",
        "generated_embeddings_used": False,
        "cvae_checkpoint_used": False,
        "source_summary_manifest_used": False,
        "target_eval_labels_used_for_scoring_only": True,
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "selections_from_target_metrics": "forbidden",
        "probabilities_calibrated": False,
        "threshold_selection": "fixed_0_5_and_source_inner_lodo",
        "external_baseline_comparison": "not_imported_requires_matching_cache_manifest_protocol_hashes",
    }
    _write_json(paths.protocol_manifest, protocol_manifest)
    leakage_report = {
        "schema_version": REAL_FEATURE_REFERENCE_SCHEMA_VERSION,
        "status": "PASS",
        "fold_unit": "heldout_center",
        "target_center_excluded_from_tuning": True,
        "target_center_excluded_from_final_fit": True,
        "target_labels_used_for_scoring_only": True,
        "target_metric_used_for_selection": False,
        "selection_used_target_labels": False,
        "fit_used_target_center": False,
        "generated_embeddings_used": False,
        "cvae_checkpoint_used": False,
        "source_summary_manifest_used": False,
        "is_router": False,
        "claim_scope": "real_feature_transfer_only",
        "probabilities_calibrated": False,
        "threshold_selection_source": "fixed_0_5_and_source_inner_lodo",
        "overlap_rows": leakage_folds,
    }
    _write_json(paths.leakage_report, leakage_report)
    return paths


def _augment_tuning_row(
    row: Mapping[str, object],
    *,
    frame: RealFeatureFrame,
    protocol_hash: str,
    fold_audit: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        **dict(row),
        "real_feature_schema_version": REAL_FEATURE_REFERENCE_SCHEMA_VERSION,
        "protocol_hash": protocol_hash,
        "feature_cache_path": str(frame.feature_cache_path),
        "feature_cache_hash": frame.feature_cache_hash,
        "manifest_path": str(frame.manifest_path),
        "manifest_hash": frame.manifest_hash,
        "source_inner_fold_audit": json.dumps(fold_audit, sort_keys=True),
        "source_inner_aggregation": "unweighted_mean_over_pseudo_target_centers",
        "target_eval_labels_used_for_scoring_only": "false",
        "generated_embeddings_used": "false",
        "cvae_checkpoint_used": "false",
        "source_summary_manifest_used": "false",
        "is_router": "false",
        "claim_scope": "real_feature_transfer_only",
        "probabilities_calibrated": "false",
    }


def _result_row(
    *,
    method: str,
    experiment_seed: int,
    classifier_seed: int,
    heldout_center: str,
    train_centers: Sequence[str],
    n_train: int,
    n_eval: int,
    grid_hash: str,
    spec: ClassifierSpec,
    source_inner_summary: Mapping[str, object],
    threshold_decision: ThresholdDecisionSpec,
    heldout_bacc: float,
    heldout_macro_f1: float,
    converged: bool,
    n_iter: Sequence[int],
    feature_cache_hash: str,
    manifest_hash: str,
) -> dict[str, object]:
    threshold_fields = _json_threshold_fields(artifact_fields_for_decision(threshold_decision))
    return {
        "schema_version": REAL_FEATURE_RESULTS_SCHEMA_VERSION,
        "method": method,
        "experiment_seed": int(experiment_seed),
        "classifier_seed": int(classifier_seed),
        "heldout_center": heldout_center,
        "train_centers": json.dumps(list(train_centers)),
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "classifier_grid_hash": grid_hash,
        "selected_classifier_config_hash": spec.config_hash,
        "selected_classifier_spec": json.dumps(spec.to_payload(), sort_keys=True),
        **threshold_fields,
        "selection_source": "source_inner_lodo" if method.startswith("source_inner_tuned") else "default_locked_classifier",
        "source_inner_mean_bacc": source_inner_summary.get("mean_bacc", ""),
        "source_inner_min_bacc": source_inner_summary.get("min_bacc", ""),
        "source_inner_std_bacc": source_inner_summary.get("std_bacc", ""),
        "source_inner_n_centers": source_inner_summary.get("n_centers", 0),
        "heldout_bacc": float(heldout_bacc),
        "heldout_macro_f1": float(heldout_macro_f1),
        "converged": str(bool(converged)).lower(),
        "n_iter": json.dumps(list(int(value) for value in n_iter)),
        "status": "ok" if converged else "non_converged_final_fit",
        "error_message": "",
        "feature_cache_hash": feature_cache_hash,
        "manifest_hash": manifest_hash,
        "target_eval_labels_used_for_scoring_only": "true",
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "generated_embeddings_used": "false",
        "cvae_checkpoint_used": "false",
        "source_summary_manifest_used": "false",
        "is_router": "false",
        "claim_scope": "real_feature_transfer_only",
        "probabilities_calibrated": "false",
    }


def _prediction_row(
    *,
    method: str,
    experiment_seed: int,
    classifier_seed: int,
    heldout_center: str,
    sample: RealFeatureRow,
    y_pred: int,
    prob_pos: float,
    spec: ClassifierSpec,
    threshold_decision: ThresholdDecisionSpec,
    feature_cache_hash: str,
    manifest_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": REAL_FEATURE_PREDICTIONS_SCHEMA_VERSION,
        "method": method,
        "experiment_seed": int(experiment_seed),
        "classifier_seed": int(classifier_seed),
        "heldout_center": heldout_center,
        "sample_id": sample.sample_id,
        "case_id": sample.case_id,
        "center": sample.center,
        "y_true": int(sample.label),
        "y_pred": int(y_pred),
        "prob_pos": float(prob_pos),
        "threshold_policy": threshold_decision.threshold_policy,
        "threshold_value": float(threshold_decision.threshold_value),
        "threshold_policy_group_id": threshold_decision.threshold_policy_group_id,
        "threshold_selection_source": threshold_decision.threshold_selection_source,
        "threshold_decision_config_hash": threshold_decision.decision_config_hash,
        "selected_classifier_config_hash": spec.config_hash,
        "feature_cache_hash": feature_cache_hash,
        "manifest_hash": manifest_hash,
        "target_eval_labels_used_for_scoring_only": "true",
        "selection_used_target_labels": "false",
        "fit_used_target_center": "false",
        "probabilities_calibrated": "false",
    }


def _source_inner_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = [row for row in rows if str(row.get("selected_by_source_inner_lodo")).lower() == "true"]
    if not selected:
        return {"mean_bacc": "", "min_bacc": "", "std_bacc": "", "n_centers": 0}
    vector = json.loads(str(selected[0]["source_inner_lodo_center_bacc_vector"]))
    values = [float(value) for value in vector.values()]
    if not values:
        return {"mean_bacc": "", "min_bacc": "", "std_bacc": "", "n_centers": 0}
    mean_value = sum(values) / float(len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / float(len(values))
    return {
        "mean_bacc": mean_value,
        "min_bacc": min(values),
        "std_bacc": math.sqrt(variance),
        "n_centers": len(values),
    }


def _json_threshold_fields(fields: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(fields)
    for key in (
        "selected_source_inner_score_vector",
        "threshold_grid",
        "threshold_objective_by_threshold",
        "threshold_valid_pseudo_target_centers",
    ):
        if key in normalized and not isinstance(normalized[key], str):
            normalized[key] = json.dumps(normalized[key], sort_keys=True)
    return normalized


def _outer_and_inner_leakage_rows(
    frame: RealFeatureFrame,
    *,
    heldout_center: str,
    fold_audit: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    final_train_idx = _indices_for_centers(frame, tuple(center for center in frame.eligible_centers if center != heldout_center))
    final_eval_idx = _indices_for_centers(frame, (heldout_center,))
    rows.append(
        _overlap_row(
            frame,
            fold_role="outer_final_fit",
            heldout_center=heldout_center,
            pseudo_target_center="",
            train_idx=final_train_idx,
            eval_idx=final_eval_idx,
        )
    )
    for pseudo_target, audit in fold_audit.items():
        train_centers = tuple(str(center) for center in audit["train_centers"])  # type: ignore[index]
        train_idx = _indices_for_centers(frame, train_centers)
        eval_idx = _indices_for_centers(frame, (str(pseudo_target),))
        rows.append(
            _overlap_row(
                frame,
                fold_role="source_inner_lodo",
                heldout_center=heldout_center,
                pseudo_target_center=str(pseudo_target),
                train_idx=train_idx,
                eval_idx=eval_idx,
            )
        )
    return rows


def _overlap_row(
    frame: RealFeatureFrame,
    *,
    fold_role: str,
    heldout_center: str,
    pseudo_target_center: str,
    train_idx: Sequence[int],
    eval_idx: Sequence[int],
) -> dict[str, object]:
    train_samples = {frame.rows[idx].sample_id for idx in train_idx}
    eval_samples = {frame.rows[idx].sample_id for idx in eval_idx}
    train_cases = {frame.rows[idx].case_id for idx in train_idx}
    eval_cases = {frame.rows[idx].case_id for idx in eval_idx}
    return {
        "fold_role": fold_role,
        "heldout_center": heldout_center,
        "pseudo_target_center": pseudo_target_center,
        "sample_overlap_count": len(train_samples.intersection(eval_samples)),
        "case_overlap_count": len(train_cases.intersection(eval_cases)),
        "target_center_excluded_from_fit": heldout_center not in {frame.rows[idx].center for idx in train_idx},
        "pseudo_target_excluded_from_inner_fit": (
            True if not pseudo_target_center else pseudo_target_center not in {frame.rows[idx].center for idx in train_idx}
        ),
    }


def _validate_heldout_centers(heldout_centers: Sequence[str], *, frame: RealFeatureFrame) -> tuple[str, ...]:
    heldouts = tuple(str(center) for center in heldout_centers)
    if not heldouts:
        raise ProtocolError("At least one heldout center is required.")
    unknown = sorted(set(heldouts).difference(MIDOGPP_ELIGIBLE_CENTERS))
    if unknown:
        raise ProtocolError(f"Unknown or quarantined MIDOG++ heldout centers: {unknown}")
    missing = sorted(set(heldouts).difference(frame.eligible_centers))
    if missing:
        raise ProtocolError(f"Heldout centers are absent from real-feature frame: {missing}")
    return heldouts


def _indices_for_centers(frame: RealFeatureFrame, centers: Sequence[str]) -> tuple[int, ...]:
    center_set = set(str(center) for center in centers)
    if center_set.intersection(MIDOGPP_EXCLUDED_CENTERS):
        raise ProtocolError(f"Quarantined MIDOG++ centers cannot be used: {sorted(center_set)}")
    return tuple(idx for idx, row in enumerate(frame.rows) if row.center in center_set)


def _embeddings_for_indices(frame: RealFeatureFrame, indices: Sequence[int]) -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("MIDOG++ real-feature classifier requires numpy.") from exc
    embeddings = frame.embeddings
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype=float)[list(indices)]


def _labels_for_indices(frame: RealFeatureFrame, indices: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(frame.rows[idx].label) for idx in indices)


def _assert_binary_class_support(labels: Sequence[int], context: str) -> None:
    if sorted({int(label) for label in labels}) != [0, 1]:
        raise ProtocolError(f"{context} must contain both binary classes 0/1.")


def _class_counts(labels: Sequence[int]) -> dict[str, int]:
    return {str(label): sum(1 for value in labels if int(value) == label) for label in (0, 1)}


def _default_spec_for_seed(classifier_seed: int) -> ClassifierSpec:
    return ClassifierSpec(
        C=DEFAULT_LOCKED_CLASSIFIER_SPEC.C,
        penalty=DEFAULT_LOCKED_CLASSIFIER_SPEC.penalty,
        solver=DEFAULT_LOCKED_CLASSIFIER_SPEC.solver,
        max_iter=DEFAULT_LOCKED_CLASSIFIER_SPEC.max_iter,
        class_weight=DEFAULT_LOCKED_CLASSIFIER_SPEC.class_weight,
        random_state=int(classifier_seed),
        l1_ratio=DEFAULT_LOCKED_CLASSIFIER_SPEC.l1_ratio,
        threshold_policy=DEFAULT_LOCKED_CLASSIFIER_SPEC.threshold_policy,
        scaler_fit=DEFAULT_LOCKED_CLASSIFIER_SPEC.scaler_fit,
        family=DEFAULT_LOCKED_CLASSIFIER_SPEC.family,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write empty CSV artifact: {path}")
    fieldnames = tuple(columns or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def specs_from_cli_values(
    *,
    c_grid: str,
    penalties: str,
    solvers: str,
    class_weights: str,
    max_iters: str,
    classifier_seed: int,
    l1_ratios: str = "",
) -> tuple[ClassifierSpec, ...]:
    """Compatibility wrapper for tests and thin CLIs."""

    return build_classifier_specs(
        c_grid=c_grid,
        penalties=penalties,
        solvers=solvers,
        class_weights=class_weights,
        max_iters=max_iters,
        classifier_seed=classifier_seed,
        l1_ratios=l1_ratios,
    )
