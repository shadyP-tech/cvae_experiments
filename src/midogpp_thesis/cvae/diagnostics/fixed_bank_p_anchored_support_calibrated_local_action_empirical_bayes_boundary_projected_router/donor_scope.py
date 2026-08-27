"""H/J/d and delete-center scope validation for SCALE-BP donor fits."""

from __future__ import annotations

from collections.abc import Sequence

from .donor_contracts import DonorDeleteCenterFold, DonorObservation
from .protocol import ProtocolError
from .replay_scope import DonorScope


def validate_scope_rows(
    rows: tuple[DonorObservation, ...], scope: DonorScope
) -> None:
    centers = {row.query_center for row in rows}
    cases = {row.case_id for row in rows}
    excluded = set(scope.source_excluded_centers)
    if (
        centers != set(scope.donor_training_centers)
        or cases != set(scope.donor_training_case_ids)
        or scope.held_case_id in cases
        or any(
            row.scope_hash != scope.scope_hash
            or row.query_center not in scope.donor_training_centers
            or not set(row.source_centers) <= set(scope.donor_training_centers)
            or set(row.source_centers) & excluded
            for row in rows
        )
    ):
        raise ProtocolError("SCALE-BP donor H/J/d scope lineage drifted.")


def validate_delete_center_folds(
    folds: Sequence[DonorDeleteCenterFold],
    base_rows: tuple[DonorObservation, ...],
    scope: DonorScope,
) -> tuple[DonorDeleteCenterFold, ...]:
    rows = tuple(folds)
    centers = tuple(scope.donor_training_centers)
    if (
        tuple(fold.deleted_center for fold in rows) != centers
        or len({fold.fold_hash for fold in rows}) != len(rows)
    ):
        raise ProtocolError("SCALE-BP donor delete-center inventory drifted.")
    base_names = base_rows[0].descriptor.feature_names
    for fold in rows:
        training = fold.training_observations
        expected_centers = set(centers) - {fold.deleted_center}
        if (
            {row.query_center for row in training} != expected_centers
            or any(row.scope_hash != scope.scope_hash for row in training)
            or any(row.descriptor.feature_names != base_names for row in training)
            or any(
                set(row.source_centers)
                & ({fold.deleted_center} | set(scope.source_excluded_centers))
                for row in training
            )
            or scope.held_case_id in {row.case_id for row in training}
        ):
            raise ProtocolError(
                "SCALE-BP donor delete-center candidate/source exclusion drifted."
            )
    return rows


__all__ = ("validate_delete_center_folds", "validate_scope_rows")
