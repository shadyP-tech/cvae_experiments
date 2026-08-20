"""Cell-wise donor residuals and partially pooled robust scales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ALTERNATIVE_METHOD_IDS,
    DIRECTION_IDS,
    RESIDUAL_POOLING_PSEUDOCOUNT,
    RESIDUAL_SCALE_FLOOR,
    ROBUST_MAD_SCALE,
    UTILITY_FEATURE_NAMES,
    UTILITY_RESPONSE_IDS,
)
from .utility_contracts import (
    DonorUtilityRow,
    FeatureReference,
    PosteriorUtilityPrediction,
    ResidualScale,
)


@dataclass(frozen=True, order=True)
class ResidualObservation:
    donor_center: str
    case_id: str
    alternative: str
    direction: str
    descriptor_hash: str
    crossing_count: int
    feature_values: tuple[float, ...]
    bacc_error: float
    brier_error: float
    log_loss_error: float

    @property
    def cell(self) -> tuple[str, str]:
        return self.alternative, self.direction

    def error(self, response_id: str) -> float:
        if response_id == "bacc_contribution_delta":
            return self.bacc_error
        if response_id == "brier_contribution_delta":
            return self.brier_error
        if response_id == "log_loss_contribution_delta":
            return self.log_loss_error
        raise ProtocolError("PSSCUR requested an unknown residual endpoint.")


def posterior_point(
    prediction: PosteriorUtilityPrediction, response_id: str
) -> float:
    """Return the fold-median posterior utility for one endpoint."""

    if response_id == "bacc_contribution_delta":
        values = prediction.fold_bacc_deltas
    elif response_id == "brier_contribution_delta":
        values = prediction.fold_brier_deltas
    elif response_id == "log_loss_contribution_delta":
        values = prediction.fold_log_loss_deltas
    else:
        raise ProtocolError("PSSCUR requested an unknown posterior endpoint.")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def build_residual_observations(
    predictions: Sequence[PosteriorUtilityPrediction],
    donor_rows: Sequence[DonorUtilityRow],
    *,
    allowed_donors: Sequence[str],
) -> tuple[ResidualObservation, ...]:
    """Align donor predictions and outcomes without treating rows as IID units."""

    allowed = tuple(str(value) for value in allowed_donors)
    utility_by_hash = {
        row.descriptor_hash: row
        for row in predictions
        if row.target_center in set(allowed)
    }
    outcome_by_hash = {
        row.descriptor_hash: row
        for row in donor_rows
        if row.donor_center in set(allowed)
    }
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or not utility_by_hash
        or set(utility_by_hash) != set(outcome_by_hash)
    ):
        raise ProtocolError("PSSCUR donor residual alignment drifted.")
    observations: list[ResidualObservation] = []
    for descriptor_hash in sorted(utility_by_hash):
        prediction = utility_by_hash[descriptor_hash]
        outcome = outcome_by_hash[descriptor_hash]
        if (
            prediction.key
            != (
                outcome.donor_center,
                outcome.case_id,
                outcome.alternative,
                outcome.direction,
            )
            or prediction.crossing_count != outcome.crossing_count
        ):
            raise ProtocolError("PSSCUR donor residual binding drifted.")
        observations.append(
            ResidualObservation(
                outcome.donor_center,
                outcome.case_id,
                outcome.alternative,
                outcome.direction,
                descriptor_hash,
                outcome.crossing_count,
                outcome.feature_values,
                posterior_point(prediction, "bacc_contribution_delta")
                - outcome.bacc_contribution_delta,
                outcome.brier_contribution_delta
                - posterior_point(prediction, "brier_contribution_delta"),
                outcome.log_loss_contribution_delta
                - posterior_point(prediction, "log_loss_contribution_delta"),
            )
        )
    expected_cells = {
        (alternative, direction)
        for alternative in ALTERNATIVE_METHOD_IDS
        for direction in DIRECTION_IDS
    }
    for donor in allowed:
        cases: dict[str, set[tuple[str, str]]] = {}
        for row in observations:
            if row.donor_center == donor:
                cases.setdefault(row.case_id, set()).add(row.cell)
        if not cases or any(cells != expected_cells for cells in cases.values()):
            raise ProtocolError("PSSCUR donor residual rectangle drifted.")
    return tuple(sorted(observations))


def fit_residual_scales(
    observations: Sequence[ResidualObservation],
) -> tuple[ResidualScale, ...]:
    """Fit deterministic cell scales shrunk toward direction/endpoint pools."""

    rows = tuple(observations)
    if not rows:
        raise ProtocolError("PSSCUR cannot fit empty residual scales.")
    output: list[ResidualScale] = []
    for alternative in ALTERNATIVE_METHOD_IDS:
        for direction in DIRECTION_IDS:
            for response_id in UTILITY_RESPONSE_IDS:
                cell = tuple(
                    row
                    for row in rows
                    if row.alternative == alternative
                    and row.direction == direction
                    and row.crossing_count > 0
                )
                crossing_pool = tuple(
                    row
                    for row in rows
                    if row.direction == direction and row.crossing_count > 0
                )
                structural_cell = tuple(
                    row
                    for row in rows
                    if row.alternative == alternative
                    and row.direction == direction
                )
                structural_pool = tuple(
                    row for row in rows if row.direction == direction
                )
                pool = crossing_pool or structural_pool
                if not pool or not structural_cell:
                    raise ProtocolError("PSSCUR residual direction pool is empty.")
                effective_cell = cell or structural_cell
                cell_scale = _zero_centered_robust_scale(
                    tuple(row.error(response_id) for row in effective_cell)
                )
                pooled_scale = _zero_centered_robust_scale(
                    tuple(row.error(response_id) for row in pool)
                )
                weight = len(cell) / (
                    len(cell) + RESIDUAL_POOLING_PSEUDOCOUNT
                )
                shrunk = float(
                    np.sqrt(
                        weight * cell_scale * cell_scale
                        + (1.0 - weight) * pooled_scale * pooled_scale
                    )
                )
                output.append(
                    ResidualScale(
                        alternative,
                        direction,
                        response_id,
                        len(effective_cell),
                        cell_scale,
                        pooled_scale,
                        max(shrunk, RESIDUAL_SCALE_FLOOR),
                    )
                )
    return tuple(sorted(output, key=lambda row: row.key))


def fit_feature_references(
    observations: Sequence[ResidualObservation],
) -> tuple[FeatureReference, ...]:
    """Fit robust, label-free descriptor references for held-case shift."""

    rows = tuple(observations)
    output: list[FeatureReference] = []
    for alternative in ALTERNATIVE_METHOD_IDS:
        for direction in DIRECTION_IDS:
            cell = tuple(
                row
                for row in rows
                if row.alternative == alternative and row.direction == direction
            )
            if not cell:
                raise ProtocolError("PSSCUR feature reference cell is empty.")
            matrix = np.asarray(
                [row.feature_values for row in cell], dtype=np.float64
            )
            if matrix.shape != (len(cell), len(UTILITY_FEATURE_NAMES)):
                raise ProtocolError("PSSCUR feature reference matrix drifted.")
            location = np.median(matrix, axis=0)
            scale = np.maximum(
                ROBUST_MAD_SCALE * np.median(np.abs(matrix - location), axis=0),
                RESIDUAL_SCALE_FLOOR,
            )
            output.append(
                FeatureReference(
                    alternative,
                    direction,
                    len(cell),
                    tuple(float(value) for value in location),
                    tuple(float(value) for value in scale),
                )
            )
    return tuple(sorted(output, key=lambda row: row.key))


def scale_index(
    scales: Sequence[ResidualScale],
) -> Mapping[tuple[str, str, str], ResidualScale]:
    result = {row.key: row for row in scales}
    if len(result) != 18:
        raise ProtocolError("PSSCUR residual scale index drifted.")
    return result


def _zero_centered_robust_scale(values: Sequence[float]) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ProtocolError("PSSCUR residual scale input drifted.")
    return max(
        float(ROBUST_MAD_SCALE * np.median(np.abs(array))),
        RESIDUAL_SCALE_FLOOR,
    )


__all__ = (
    "ResidualObservation",
    "build_residual_observations",
    "fit_feature_references",
    "fit_residual_scales",
    "posterior_point",
    "scale_index",
)
