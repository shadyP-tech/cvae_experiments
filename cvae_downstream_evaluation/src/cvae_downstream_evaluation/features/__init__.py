"""Allowed pre-evaluation feature table validation."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..schemas import (
    DIAGNOSTIC_ONLY,
    FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS,
    REQUIRED_LINEAGE_COLUMNS,
    SELECTION_ELIGIBLE,
)

ALLOWED_FEATURE_SOURCES = (
    "target_support_only",
    "source_inner_only",
    "metadata_only",
    "source_manifest_only",
)


def assert_allowed_feature_table(rows: Sequence[Mapping[str, object]]) -> None:
    """Fail closed if deployable features contain oracle or target-eval columns."""

    if not rows:
        raise ProtocolError("Allowed pre-evaluation feature table is empty.")
    for idx, row in enumerate(rows):
        missing = [key for key in REQUIRED_LINEAGE_COLUMNS if key not in row]
        if missing:
            raise ProtocolError(f"Feature row {idx} missing lineage columns: {missing}")
        eligibility = str(row.get("eligibility", ""))
        if eligibility not in {SELECTION_ELIGIBLE, DIAGNOSTIC_ONLY}:
            raise ProtocolError(f"Feature row {idx} has unknown eligibility: {eligibility!r}")
        predictive_columns = set(row).difference(REQUIRED_LINEAGE_COLUMNS)
        forbidden = sorted(predictive_columns.intersection(FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS))
        if forbidden:
            raise ProtocolError(f"Feature row {idx} contains forbidden deployable columns: {forbidden}")
        source = str(row.get("feature_source", ""))
        if source and source not in ALLOWED_FEATURE_SOURCES:
            raise ProtocolError(f"Feature row {idx} has non-whitelisted feature_source={source!r}")


def assert_no_target_identity_features(columns: Sequence[str]) -> None:
    """Reject direct target identity aliases as predictive feature columns."""

    forbidden = sorted(set(str(column) for column in columns).intersection(FORBIDDEN_DEPLOYABLE_FEATURE_COLUMNS))
    if forbidden:
        raise ProtocolError(f"Deployable feature columns include target/oracle identity fields: {forbidden}")


def deployable_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return model-input columns after lineage fields are removed and validated."""

    predictive = tuple(
        str(column)
        for column in columns
        if str(column) not in set(REQUIRED_LINEAGE_COLUMNS).union({"feature_source"})
    )
    assert_no_target_identity_features(predictive)
    return predictive
