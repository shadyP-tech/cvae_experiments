"""Schemas for MIDOG++ real-feature recipe-reference artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from .midogpp import MIDOGPP_EXCLUDED_CENTERS

RECIPE_REFERENCE_SCHEMA_VERSION = "midogpp_real_feature_source_only_recipe_reference_v1"
SOURCE_INNER_RECIPE_TUNING_SCHEMA_VERSION = "midogpp_real_feature_source_inner_recipe_tuning_v1"
RECIPE_RESULTS_SCHEMA_VERSION = "midogpp_real_feature_recipe_results_v1"
RECIPE_PREDICTIONS_SCHEMA_VERSION = "midogpp_real_feature_recipe_predictions_v1"

SOURCE_INNER_RECIPE_TUNING_COLUMNS = (
    "schema_version",
    "experiment_seed",
    "classifier_seed",
    "outer_target_center",
    "selector_centers",
    "inner_lodo_centers",
    "recipe_grid_hash",
    "recipe_spec",
    "selected_recipe_config_hash",
    "recipe_family",
    "preprocessing_id",
    "source_inner_lodo_center_bacc_vector",
    "source_inner_lodo_center_macro_f1_vector",
    "selection_metric",
    "selection_source",
    "selected_by_source_inner_lodo",
    "aggregate_selection_score",
    "min_selection_score",
    "se_selection_score",
    "within_one_se_best",
    "tie_breaker",
    "convergence_by_center",
    "n_iter_by_center",
    "status_by_center",
    "row_role",
    "protocol_hash",
    "feature_cache_path",
    "feature_cache_hash",
    "manifest_path",
    "manifest_hash",
    "source_inner_fold_audit",
    "fit_scope_policy",
    "preprocessing_fit_scope",
    "model_fit_scope",
    "decision_rule",
    "selection_used_target_labels",
    "fit_used_target_center",
    "target_eval_labels_used_for_scoring",
    "generated_embeddings_used",
    "cvae_checkpoint_used",
    "source_summary_manifest_used",
    "is_router",
    "claim_scope",
    "probabilities_calibrated",
)

RECIPE_RESULT_COLUMNS = (
    "schema_version",
    "method",
    "row_role",
    "experiment_seed",
    "classifier_seed",
    "heldout_center",
    "train_centers",
    "n_train",
    "n_eval",
    "recipe_grid_hash",
    "selected_recipe_config_hash",
    "selected_recipe_spec",
    "recipe_family",
    "preprocessing_id",
    "selection_source",
    "source_inner_mean_bacc",
    "source_inner_min_bacc",
    "source_inner_se_bacc",
    "source_inner_n_centers",
    "heldout_bacc",
    "heldout_macro_f1",
    "converged",
    "n_iter",
    "status",
    "error_message",
    "preprocessing_fit_scope",
    "model_fit_scope",
    "decision_rule",
    "feature_cache_hash",
    "manifest_hash",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "fit_used_target_center",
    "generated_embeddings_used",
    "cvae_checkpoint_used",
    "source_summary_manifest_used",
    "is_router",
    "claim_scope",
    "probabilities_calibrated",
)

RECIPE_PREDICTION_COLUMNS = (
    "schema_version",
    "method",
    "row_role",
    "experiment_seed",
    "classifier_seed",
    "heldout_center",
    "sample_id",
    "case_id",
    "center",
    "y_true",
    "y_pred",
    "score_pos",
    "decision_score",
    "score_kind",
    "selected_recipe_config_hash",
    "recipe_family",
    "preprocessing_id",
    "feature_cache_hash",
    "manifest_hash",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "fit_used_target_center",
    "probabilities_calibrated",
)

RECIPE_REQUIRED_OUTPUTS = (
    "tables/source_inner_recipe_tuning.csv",
    "tables/recipe_tuned_source_results.csv",
    "tables/recipe_tuned_predictions.csv",
    "manifests/protocol_manifest.json",
    "reports/leakage_provenance_report.json",
)


def assert_midogpp_real_feature_recipe_artifacts(root: Path) -> None:
    root = Path(root)
    missing = [rel for rel in RECIPE_REQUIRED_OUTPUTS if not (root / rel).exists()]
    if missing:
        raise ProtocolError(f"MIDOG++ real-feature recipe artifacts missing outputs: {missing}")
    protocol = _read_json(root / "manifests" / "protocol_manifest.json")
    leakage = _read_json(root / "reports" / "leakage_provenance_report.json")
    if protocol.get("schema_version") != RECIPE_REFERENCE_SCHEMA_VERSION:
        raise ProtocolError("Unexpected MIDOG++ real-feature recipe protocol schema_version.")
    if leakage.get("status") != "PASS":
        raise ProtocolError("MIDOG++ real-feature recipe leakage report must have status=PASS.")
    _assert_protocol_boundary_flags(protocol)
    _assert_protocol_boundary_flags(leakage)
    _assert_no_forbidden_protocol_keys(protocol)
    tuning_fields, tuning_rows = _read_csv(root / "tables" / "source_inner_recipe_tuning.csv")
    result_fields, result_rows = _read_csv(root / "tables" / "recipe_tuned_source_results.csv")
    prediction_fields, prediction_rows = _read_csv(root / "tables" / "recipe_tuned_predictions.csv")
    _assert_columns(tuning_rows, SOURCE_INNER_RECIPE_TUNING_COLUMNS, "source_inner_recipe_tuning.csv")
    _assert_columns(result_rows, RECIPE_RESULT_COLUMNS, "recipe_tuned_source_results.csv")
    _assert_columns(prediction_rows, RECIPE_PREDICTION_COLUMNS, "recipe_tuned_predictions.csv")
    for fields in (tuning_fields, result_fields, prediction_fields):
        _assert_no_threshold_or_calibration_columns(fields)
    for row in (*tuning_rows, *result_rows, *prediction_rows):
        _assert_protocol_boundary_flags(row)
        if str(row.get("heldout_center", "")) in MIDOGPP_EXCLUDED_CENTERS:
            raise ProtocolError("Excluded MIDOG++ center appears in recipe artifact rows.")
    _assert_recipe_payloads(tuning_rows, "recipe_spec")
    _assert_recipe_payloads(result_rows, "selected_recipe_spec")
    _assert_prediction_scores(prediction_rows)
    _assert_locked_logistic_baseline(protocol, result_rows)
    _assert_fit_scope_report(leakage)


def _assert_protocol_boundary_flags(row: Mapping[str, object]) -> None:
    expected_false = (
        "selection_used_target_labels",
        "fit_used_target_center",
        "generated_embeddings_used",
        "cvae_checkpoint_used",
        "source_summary_manifest_used",
        "is_router",
        "probabilities_calibrated",
    )
    for field in expected_false:
        if field in row and str(row[field]).lower() != "false":
            raise ProtocolError(f"{field} must be false in MIDOG++ real-feature recipe artifacts.")
    if "claim_scope" in row and str(row["claim_scope"]) != "real_feature_transfer_only":
        raise ProtocolError("claim_scope must be real_feature_transfer_only.")


def _assert_no_forbidden_protocol_keys(protocol: Mapping[str, object]) -> None:
    forbidden = [key for key in protocol if key.startswith("threshold") or key.startswith("calibration")]
    if forbidden:
        raise ProtocolError(f"Recipe protocol must not contain threshold/calibration keys: {forbidden}")
    if "4" in [str(value) for value in protocol.get("heldout_centers", [])]:
        raise ProtocolError("Excluded MIDOG++ center 4 appears in protocol heldout centers.")


def _assert_no_threshold_or_calibration_columns(fields: Sequence[str]) -> None:
    forbidden = [
        field
        for field in fields
        if field.startswith("threshold") or field.startswith("calibration")
    ]
    if forbidden:
        raise ProtocolError(f"Recipe artifacts must not contain threshold/calibration columns: {forbidden}")


def _assert_recipe_payloads(rows: Sequence[Mapping[str, object]], field: str) -> None:
    for row in rows:
        payload = _json_object(row[field], field)
        preprocessing = payload.get("preprocessing")
        model = payload.get("model")
        if not isinstance(preprocessing, dict) or not isinstance(model, dict):
            raise ProtocolError("Recipe payload must include preprocessing and model objects.")
        if not preprocessing.get("preprocessing_id"):
            raise ProtocolError("Recipe payload missing preprocessing_id.")
        if not model.get("family"):
            raise ProtocolError("Recipe payload missing model family.")
        family = str(model["family"])
        if family in {"logistic", "linear_svm"} and model.get("C") in {None, ""}:
            raise ProtocolError(f"{family} recipe missing C.")
        if family == "nystroem_svm" and (
            model.get("C") in {None, ""}
            or model.get("nystroem_components") in {None, ""}
            or model.get("gamma") in {None, ""}
        ):
            raise ProtocolError("nystroem_svm recipe missing family hyperparameters.")
        if family == "mlp" and (not model.get("hidden_layer_sizes") or model.get("alpha") in {None, ""}):
            raise ProtocolError("MLP recipe missing family hyperparameters.")
        if model.get("decision_rule") != "predict":
            raise ProtocolError("Recipe decision_rule must be predict.")


def _assert_prediction_scores(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        family = str(row["recipe_family"])
        score_pos = str(row.get("score_pos", ""))
        if family in {"linear_svm", "nystroem_svm"}:
            if score_pos.strip():
                raise ProtocolError("SVM recipe prediction rows must not contain probabilities.")
            if str(row.get("score_kind")) != "decision_function":
                raise ProtocolError("SVM recipe prediction rows must store decision_function diagnostics.")


def _assert_locked_logistic_baseline(protocol: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> None:
    heldouts = [str(center) for center in protocol.get("heldout_centers", [])]
    baseline_rows = [row for row in rows if str(row.get("method")) == "locked_logistic_baseline_predict"]
    baseline_heldouts = {str(row["heldout_center"]) for row in baseline_rows}
    missing = sorted(set(heldouts).difference(baseline_heldouts))
    if missing:
        raise ProtocolError(f"Missing same-cache locked logistic baseline for heldouts: {missing}")
    feature_hashes = {str(row["feature_cache_hash"]) for row in rows}
    manifest_hashes = {str(row["manifest_hash"]) for row in rows}
    if len(feature_hashes) != 1 or len(manifest_hashes) != 1:
        raise ProtocolError("Recipe rows must use one same-cache/same-manifest frame.")


def _assert_fit_scope_report(leakage: Mapping[str, object]) -> None:
    rows = leakage.get("fit_scope_rows")
    if not isinstance(rows, list) or not rows:
        raise ProtocolError("Recipe leakage report missing fit_scope_rows.")
    for row in rows:
        if not isinstance(row, dict):
            raise ProtocolError("Malformed fit_scope_rows entry.")
        if str(row.get("target_center_excluded_from_fit")).lower() != "true":
            raise ProtocolError("Fit-scope report must exclude outer target center.")
        if row.get("fold_role") == "source_inner_lodo" and str(row.get("pseudo_target_excluded_from_inner_fit")).lower() != "true":
            raise ProtocolError("Fit-scope report must exclude inner pseudo-target center.")


def _assert_columns(rows: Sequence[Mapping[str, object]], required: Sequence[str], label: str) -> None:
    if not rows:
        raise ProtocolError(f"{label} is empty.")
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise ProtocolError(f"{label} missing columns: {missing}")


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty CSV: {path}")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON artifact is not an object: {path}")
    return payload


def _json_object(raw: object, field: str) -> dict[str, object]:
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON field {field}: {raw!r}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected {field} to be a JSON object.")
    return payload
