"""Shared constants and strict readers for prediction-only validation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import EXPECTED_SOURCE_ROWS, EXPECTED_TEST_ROWS
from .hashing import canonical_hash


EXPECTED_SOURCE_CASE_COUNT = 216
EXPECTED_TEST_CASE_COUNT = 218
EXPECTED_MODEL_BANK_COUNT = 54
EXPECTED_MODEL_COUNT = 432
EXPECTED_SOURCE_FEATURE_ROWS = 82_944
EXPECTED_SOURCE_RESPONSE_ROWS = 27_648
EXPECTED_TEST_FEATURE_ROWS = 11_772
EXPECTED_TEST_CONTRAST_ROWS = 10_464
EXPECTED_TEST_SELECTION_ROWS = 1_308

SOURCE_FEATURE_FIELDS = (
    "outer_target_id",
    "geometry_id",
    "family",
    "query_id",
    "case_id",
    "action_id",
    "source_id",
    "sample_count",
    "disagreement_count",
    "prediction_seal_hash",
    "feature_origin_action_id",
    "feature_hash",
    *(f"feature_{index:02d}" for index in range(15)),
)
TEST_FEATURE_FIELDS = tuple(
    value for value in SOURCE_FEATURE_FIELDS if value != "query_id"
)
SOURCE_RESPONSE_FIELDS = (
    "outer_target_id",
    "geometry_id",
    "query_id",
    "case_id",
    "action_id",
    "source_id",
    "source_exact_bacc_gain_vs_control",
    "source_exact_regret_from_case_best",
    "disagreement_count",
    "positive_class_count",
    "negative_class_count",
    "response_hash",
    "response_surface_hash",
)
MODEL_TABLE_FIELDS = (
    "ordinal",
    "outer_target_id",
    "geometry_id",
    "family",
    "candidate_action_id",
    "candidate_source_id",
    "observation_count",
    "iteration_count",
    "model_hash",
    "model_bank_hash",
)
CONTRAST_FIELDS = (
    "geometry_id",
    "family",
    "target_query_id",
    "case_id",
    "candidate_action_id",
    "candidate_source_id",
    "predicted_preference_margin_vs_control",
    "standard_error_vs_control",
    "predicted_preference_margin_vs_baseline",
    "standard_error_vs_baseline",
    "model_hash",
    "score_semantics",
)
SELECTION_FIELDS = (
    "geometry_id",
    "family",
    "target_query_id",
    "case_id",
    "raw_action_id",
    "safe_action_id",
    "baseline_action_id",
    "control_action_id",
    "simultaneous_z_value",
    "safe_margin",
    "fallback_reason",
    "claim_role",
    "may_authorize_routing",
    "may_authorize_promotion",
)
SUMMARY_FIELDS = (
    "geometry_id",
    "target_query_id",
    "family",
    "raw_action_id",
    "safe_action_id",
    "fallback_reason",
    "case_count",
    "test_labels_used",
    "test_metric_computed",
)

_FORBIDDEN_JSON_KEYS = frozenset(
    {
        "label",
        "labels",
        "raw_label",
        "raw_labels",
        "raw_source_labels",
        "raw_test_labels",
        "y_true",
        "bacc",
        "accuracy",
        "regret",
        "utility",
        "oracle",
        "nelbo",
        "downstream_metric",
    }
)
_FORBIDDEN_COLUMN_TOKENS = (
    "label",
    "y_true",
    "bacc",
    "accuracy",
    "regret",
    "utility",
    "oracle",
    "nelbo",
    "downstream",
)
_ALLOWED_SOURCE_RESPONSE_METRICS = frozenset(
    {
        "source_exact_bacc_gain_vs_control",
        "source_exact_regret_from_case_best",
    }
)
_ALLOWED_FAIL_CLOSED_COLUMNS = frozenset(
    {"test_labels_used", "test_metric_computed"}
)


def reject_forbidden_persisted_fields(root: Path) -> None:
    """Reject target outcomes/metrics while allowing two source aggregates."""

    for directory in ("manifests", "reports", "provenance"):
        for path in sorted((root / directory).glob("*.json")):
            if contains_forbidden_json_key(read_object(path)):
                raise ProtocolError(
                    f"Prediction-only JSON persisted a forbidden outcome key: {path}."
                )
    for path in sorted((root / "tables").glob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
        except (OSError, StopIteration, csv.Error) as exc:
            raise ProtocolError(f"Cannot inspect prediction-only table: {path}.") from exc
        if len(header) != len(set(header)):
            raise ProtocolError(f"Prediction-only table header is duplicated: {path}.")
        metric_columns = {
            name
            for name in header
            if any(token in name.lower() for token in _FORBIDDEN_COLUMN_TOKENS)
        }
        permitted = _ALLOWED_FAIL_CLOSED_COLUMNS.union(
            _ALLOWED_SOURCE_RESPONSE_METRICS
            if path.name == "source_regret_responses.csv"
            else ()
        )
        if metric_columns.difference(permitted):
            raise ProtocolError(
                f"Prediction-only table persisted a forbidden target/outcome column: {path}."
            )
        if (
            path.name == "source_regret_responses.csv"
            and not _ALLOWED_SOURCE_RESPONSE_METRICS.issubset(header)
        ):
            raise ProtocolError("Prediction-only source response allowlist is incomplete.")


def contains_forbidden_json_key(value: object) -> bool:
    if isinstance(value, Mapping):
        if _FORBIDDEN_JSON_KEYS.intersection(str(key).lower() for key in value):
            return True
        return any(contains_forbidden_json_key(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_json_key(item) for item in value)
    return False


def read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read prediction-only JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Prediction-only JSON must be an object: {path}.")
    return value


def read_csv(path: Path, *, fields: Sequence[str]) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise ProtocolError(f"Prediction-only CSV schema drifted: {path}.")
            rows = tuple(dict(row) for row in reader)
    except (OSError, csv.Error) as exc:
        raise ProtocolError(f"Cannot read prediction-only CSV: {path}.") from exc
    if any(
        set(row) != set(fields) or any(value is None for value in row.values())
        for row in rows
    ):
        raise ProtocolError(f"Prediction-only CSV row width drifted: {path}.")
    return rows


def integer(value: object) -> int:
    text = str(value)
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ProtocolError("Prediction-only integer field is malformed.") from exc
    if str(parsed) != text:
        raise ProtocolError("Prediction-only integer field is noncanonical.")
    return parsed


def finite_float(value: object) -> float:
    try:
        parsed = float(str(value))
    except ValueError as exc:
        raise ProtocolError("Prediction-only numeric field is malformed.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError("Prediction-only numeric field must be finite.")
    return parsed


def csv_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def expected_disjointness_report() -> Mapping[str, object]:
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_prediction_only_train_test_disjointness_v1",
        "status": "PASS",
        "source_row_count": EXPECTED_SOURCE_ROWS,
        "test_row_count": EXPECTED_TEST_ROWS,
        "source_case_count": EXPECTED_SOURCE_CASE_COUNT,
        "test_case_count": EXPECTED_TEST_CASE_COUNT,
        "case_overlap_count": 0,
        "opaque_row_identity_overlap_count": 0,
        "source_split": "train",
        "test_split": "test",
    }
    return {**unhashed, "audit_hash": canonical_hash(unhashed)}


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "contains_forbidden_json_key",
    "csv_text",
    "expected_disjointness_report",
    "finite_float",
    "integer",
    "is_sha256",
    "read_csv",
    "read_object",
    "reject_forbidden_persisted_fields",
)
