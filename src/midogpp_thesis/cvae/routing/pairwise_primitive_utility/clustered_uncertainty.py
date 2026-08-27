"""Action/pair-specific center/case-OOF one-sided residual calibration."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CalibratedBound,
    OOFResidualObservation,
    SourceScopeReceipt,
    UncertaintyCalibration,
    UncertaintyComponent,
    canonical_sha256,
)


ONE_SIDED_ALPHA = 0.20
_SIDE_BY_METRIC = {
    "bacc": "lower",
    "brier": "upper",
    "log": "upper",
    "pairwise": "lower",
}


def _center_cluster_offset(
    rows: Sequence[OOFResidualObservation],
    *,
    side: str,
) -> tuple[float, int, int]:
    """Calibrate on center-level maxima after case-level deduplication."""

    keys = tuple((row.center_id, row.case_id) for row in rows)
    if len(set(keys)) != len(keys):
        raise ProtocolError("Residual calibration has duplicate center/case rows for one component.")
    errors_by_center: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        raw = (
            row.predicted - row.observed
            if side == "lower"
            else row.observed - row.predicted
        )
        errors_by_center[row.center_id].append(max(float(raw), 0.0))
    center_errors = np.asarray(
        [max(values) for _, values in sorted(errors_by_center.items())],
        dtype=np.float64,
    )
    if len(center_errors) < 4:
        raise ProtocolError("One-sided residual calibration requires at least four OOF centers.")
    # Finite-sample conformal order statistic over independent held centers.
    quantile_rank = int(
        math.ceil((len(center_errors) + 1) * (1.0 - ONE_SIDED_ALPHA))
    )
    if quantile_rank > len(center_errors):
        raise ProtocolError("Requested one-sided residual quantile is infeasible for held centers.")
    offset = float(np.sort(center_errors)[quantile_rank - 1])
    return offset, len(center_errors), len(rows)


def calibrate_clustered_uncertainty(
    observations: Sequence[OOFResidualObservation],
    *,
    calibration_scopes: Sequence[SourceScopeReceipt],
) -> UncertaintyCalibration:
    """Fit strict source-OOF offsets separately for every action/pair metric.

    Each residual must originate from the center L named by its exact scope
    receipt.  No global or cross-action fallback component is constructed;
    missing action/pair evidence therefore fails closed at lookup time.
    """

    rows = tuple(observations)
    scopes = tuple(calibration_scopes)
    if not rows or len(scopes) < 4:
        raise ProtocolError("Clustered uncertainty requires residual rows and four L scopes.")
    scope_by_hash = {scope.receipt_hash: scope for scope in scopes}
    if len(scope_by_hash) != len(scopes):
        raise ProtocolError("Residual calibration scope receipts are duplicated.")
    calibration_centers = tuple(sorted({scope.calibration_center for scope in scopes}))
    outer_targets = {scope.outer_target_center for scope in scopes}
    if len(outer_targets) != 1:
        raise ProtocolError("Residual calibration scopes mixed outer targets H.")
    outer_target = next(iter(outer_targets))
    if len(calibration_centers) < 4:
        raise ProtocolError("Residual calibration must rotate across at least four L centers.")
    for row in rows:
        scope = scope_by_hash.get(row.source_scope_receipt_hash)
        if (
            scope is None
            or row.center_id != scope.calibration_center
            or (row.center_id, row.case_id)
            == (scope.heldout_case_center, scope.heldout_case_id)
        ):
            raise ProtocolError("Residual row did not come from its receipt's held L center.")

    grouped: dict[tuple[str, str, str, str], list[OOFResidualObservation]] = defaultdict(list)
    for row in rows:
        side = _SIDE_BY_METRIC[row.metric]
        grouped[(row.action_id, row.comparator_id, row.metric, side)].append(row)
    components: list[UncertaintyComponent] = []
    for key, group in sorted(grouped.items()):
        action_id, comparator_id, metric, side = key
        centers = {row.center_id for row in group}
        if centers != set(calibration_centers):
            raise ProtocolError(
                "Every action/pair uncertainty component must cover every predeclared L center."
            )
        offset, center_count, case_count = _center_cluster_offset(group, side=side)
        components.append(
            UncertaintyComponent(
                action_id=action_id,
                comparator_id=comparator_id,
                metric=metric,
                side=side,
                offset=offset,
                alpha=ONE_SIDED_ALPHA,
                center_count=center_count,
                case_count=case_count,
                scope_receipt_hashes=tuple(sorted({row.source_scope_receipt_hash for row in group})),
            )
        )
    combined_scope_hash = canonical_sha256(
        {
            "schema": "rotating_L_calibration_scope_v1",
            "scope_receipt_hashes": tuple(sorted(scope_by_hash)),
            "calibration_centers": calibration_centers,
            "outer_target_H": outer_target,
        }
    )
    return UncertaintyCalibration(
        components=tuple(components),
        outer_target_center=outer_target,
        calibration_scope_receipt_hashes=tuple(sorted(scope_by_hash)),
    )


def apply_calibrated_bound(
    calibration: UncertaintyCalibration,
    *,
    action_id: object,
    comparator_id: object,
    metric: object,
    mean: float,
) -> CalibratedBound:
    """Apply the exact action/pair component, with no pooling fallback."""

    metric_name = str(metric)
    if metric_name not in _SIDE_BY_METRIC or not math.isfinite(float(mean)):
        raise ProtocolError("Calibrated-bound request is invalid.")
    side = _SIDE_BY_METRIC[metric_name]
    component = calibration.component(action_id, comparator_id, metric_name, side)
    bound = float(mean) - component.offset if side == "lower" else float(mean) + component.offset
    return CalibratedBound(
        mean=float(mean),
        bound=bound,
        side=side,
        component_hash=component.component_hash,
    )


__all__ = (
    "ONE_SIDED_ALPHA",
    "apply_calibrated_bound",
    "calibrate_clustered_uncertainty",
)
