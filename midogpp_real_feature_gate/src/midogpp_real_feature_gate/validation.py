"""Validation helpers for row-role and artifact protocol checks."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from .contracts import QUARANTINE_CENTERS, REQUIRED_MATRIX_COLUMNS, RowRole, SCHEMA_VERSION


class ValidationError(ValueError):
    """Raised when an artifact row violates the frozen gate contract."""


def missing_required_columns(columns: set[str]) -> tuple[str, ...]:
    return tuple(column for column in REQUIRED_MATRIX_COLUMNS if column not in columns)


def validate_row_role_flags(row: Mapping[str, object]) -> None:
    """Validate non-negotiable source-only versus diagnostic row semantics."""
    role = str(row.get("row_role", ""))
    adoption_eligible = _as_bool(row.get("adoption_eligible"))
    diagnostic_only = _as_bool(row.get("diagnostic_only"))
    fit_used_target_center = _as_bool(row.get("fit_used_target_center"))
    selection_used_target_labels = _as_bool(row.get("selection_used_target_labels"))
    scoring_only = _as_bool(row.get("target_eval_labels_used_for_scoring_only"))

    if role == RowRole.SOURCE_ONLY_TRANSFER:
        if str(row.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
            raise ValidationError("row schema_version does not match frozen gate schema")
        if str(row.get("heldout_center", "")) in QUARANTINE_CENTERS:
            raise ValidationError("quarantine centers cannot be source_only_transfer rows")
        if not adoption_eligible or diagnostic_only:
            raise ValidationError("source_only_transfer rows must be adoption-eligible transfer baselines.")
        if fit_used_target_center:
            raise ValidationError("source_only_transfer rows must not fit on the held-out target center.")
        if selection_used_target_labels:
            raise ValidationError("source_only_transfer rows must not use target labels for selection.")
        if not scoring_only:
            raise ValidationError("source_only_transfer rows must mark target labels as scoring-only.")
        return

    if role in {RowRole.POOLED_DIAGNOSTIC_CEILING, RowRole.SOURCE_ORACLE_DIAGNOSTIC}:
        if adoption_eligible or not diagnostic_only:
            raise ValidationError(f"{role} rows must be diagnostic-only and non-adoption-eligible.")
        return

    raise ValidationError(f"unknown row_role: {role!r}")


def validate_matrix_csv(path: Path) -> None:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"empty matrix CSV: {path}")
        missing = missing_required_columns(set(reader.fieldnames))
        if missing:
            raise ValidationError(f"matrix CSV missing required columns: {missing}")
        for row in reader:
            validate_row_role_flags(row)


def validate_artifact_bundle(root: Path) -> None:
    required = (
        "tables/matrix.csv",
        "tables/predictions.csv",
        "tables/confusion_summary.csv",
        "tables/stratified_breakdown.csv",
        "tables/source_only_ranking_gap.csv",
        "tables/source_vs_pooled_delta.csv",
        "tables/worst_domain_summary.csv",
        "manifests/protocol_manifest.json",
        "reports/leakage_provenance_report.json",
        "reports/decision_report.md",
    )
    missing = [item for item in required if not (Path(root) / item).exists()]
    if missing:
        raise ValidationError(f"artifact bundle missing required outputs: {missing}")
    validate_matrix_csv(Path(root) / "tables" / "matrix.csv")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
