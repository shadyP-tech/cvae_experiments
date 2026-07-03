"""Schemas for MIDOG++ real-feature classifier-reference artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from .classifier_tuning import SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS

REAL_FEATURE_REFERENCE_SCHEMA_VERSION = "midogpp_real_feature_source_only_classifier_reference_v1"
REAL_FEATURE_TUNING_SCHEMA_VERSION = "midogpp_real_feature_source_inner_classifier_tuning_v1"
REAL_FEATURE_RESULTS_SCHEMA_VERSION = "midogpp_real_feature_classifier_results_v1"
REAL_FEATURE_PREDICTIONS_SCHEMA_VERSION = "midogpp_real_feature_classifier_predictions_v1"

REAL_FEATURE_RESULT_COLUMNS = (
    "schema_version",
    "method",
    "experiment_seed",
    "classifier_seed",
    "heldout_center",
    "train_centers",
    "n_train",
    "n_eval",
    "classifier_grid_hash",
    "selected_classifier_config_hash",
    "selected_classifier_spec",
    "selection_source",
    "source_inner_mean_bacc",
    "source_inner_min_bacc",
    "source_inner_std_bacc",
    "source_inner_n_centers",
    "heldout_bacc",
    "heldout_macro_f1",
    "converged",
    "n_iter",
    "status",
    "error_message",
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

REAL_FEATURE_PREDICTION_COLUMNS = (
    "schema_version",
    "method",
    "experiment_seed",
    "classifier_seed",
    "heldout_center",
    "sample_id",
    "case_id",
    "center",
    "y_true",
    "y_pred",
    "prob_pos",
    "threshold_policy",
    "selected_classifier_config_hash",
    "feature_cache_hash",
    "manifest_hash",
    "target_eval_labels_used_for_scoring_only",
    "selection_used_target_labels",
    "fit_used_target_center",
    "probabilities_calibrated",
)

REAL_FEATURE_REQUIRED_OUTPUTS = (
    "tables/source_inner_classifier_tuning.csv",
    "tables/classifier_tuned_source_results.csv",
    "tables/classifier_tuned_predictions.csv",
    "manifests/protocol_manifest.json",
    "reports/leakage_provenance_report.json",
)


def assert_midogpp_real_feature_artifacts(root: Path) -> None:
    """Validate the required real-feature classifier-reference artifact surface."""

    root = Path(root)
    missing = [rel for rel in REAL_FEATURE_REQUIRED_OUTPUTS if not (root / rel).exists()]
    if missing:
        raise ProtocolError(f"MIDOG++ real-feature artifacts missing outputs: {missing}")
    protocol = _read_json(root / "manifests" / "protocol_manifest.json")
    leakage = _read_json(root / "reports" / "leakage_provenance_report.json")
    if protocol.get("schema_version") != REAL_FEATURE_REFERENCE_SCHEMA_VERSION:
        raise ProtocolError("Unexpected MIDOG++ real-feature protocol schema_version.")
    if leakage.get("status") != "PASS":
        raise ProtocolError("MIDOG++ real-feature leakage report must have status=PASS.")
    _assert_protocol_boundary_flags(protocol)
    _assert_protocol_boundary_flags(leakage)
    tuning_rows = _read_csv(root / "tables" / "source_inner_classifier_tuning.csv")
    result_rows = _read_csv(root / "tables" / "classifier_tuned_source_results.csv")
    prediction_rows = _read_csv(root / "tables" / "classifier_tuned_predictions.csv")
    _assert_columns(tuning_rows, SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS, "source_inner_classifier_tuning.csv")
    _assert_columns(result_rows, REAL_FEATURE_RESULT_COLUMNS, "classifier_tuned_source_results.csv")
    _assert_columns(prediction_rows, REAL_FEATURE_PREDICTION_COLUMNS, "classifier_tuned_predictions.csv")
    for row in (*tuning_rows, *result_rows, *prediction_rows):
        _assert_protocol_boundary_flags(row)
    forbidden_values = ("breakhis", "camelyon17", "exported_source_summary_manifest.csv", "SourceSummaryMidogppBackend")
    for path in (root / rel for rel in REAL_FEATURE_REQUIRED_OUTPUTS):
        text = path.read_text(encoding="utf-8").lower()
        found = [value for value in forbidden_values if value.lower() in text]
        if found:
            raise ProtocolError(f"Forbidden backend/dataset reference in {path}: {found}")


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
            raise ProtocolError(f"{field} must be false in MIDOG++ real-feature artifacts.")
    if "claim_scope" in row and str(row["claim_scope"]) != "real_feature_transfer_only":
        raise ProtocolError("claim_scope must be real_feature_transfer_only.")


def _assert_columns(rows: Sequence[Mapping[str, object]], required: Sequence[str], label: str) -> None:
    if not rows:
        raise ProtocolError(f"{label} is empty.")
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise ProtocolError(f"{label} missing columns: {missing}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ProtocolError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON artifact is not an object: {path}")
    return payload
