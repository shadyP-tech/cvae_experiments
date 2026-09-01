"""Source-only donor residual bounds for HARP v4.

These are deliberately named *donor-calibrated* bounds.  The source centers
are few and exchangeability is not asserted, so this module does not describe
the bounds as formal conformal guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import Comparison, EffectVector


CALIBRATION_METHOD = (
    "source_lodo_donor_case_balanced_joint_harm_envelope_"
    "worst_donor_finite_sample_not_formal_conformal_v2"
)
FINITE_SAMPLE_RULE = (
    "max_donor_order_statistic_k_min_n_ceil_q_times_n_plus_1_v1"
)


@dataclass(frozen=True)
class DonorResidualCalibration:
    comparison: Comparison
    quantile_level: float
    endpoint_scales: tuple[float, float, float]
    joint_harm_quantile: float
    calibration_row_count: int
    calibration_case_block_count: int
    donor_ids: tuple[str, ...]
    donor_case_counts: tuple[int, ...]
    donor_joint_harm_quantiles: tuple[float, ...]
    calibration_method: str = CALIBRATION_METHOD
    finite_sample_rule: str = FINITE_SAMPLE_RULE

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparison", Comparison(self.comparison))
        donors = tuple(sorted({str(value) for value in self.donor_ids}))
        scales = tuple(float(value) for value in self.endpoint_scales)
        counts = tuple(int(value) for value in self.donor_case_counts)
        donor_quantiles = tuple(float(value) for value in self.donor_joint_harm_quantiles)
        if (
            self.calibration_method != CALIBRATION_METHOD
            or self.finite_sample_rule != FINITE_SAMPLE_RULE
            or donors != self.donor_ids
            or not donors
            or type(self.calibration_row_count) is not int
            or type(self.calibration_case_block_count) is not int
            or self.calibration_row_count < self.calibration_case_block_count
            or self.calibration_case_block_count < len(donors)
            or not 0.5 <= float(self.quantile_level) < 1.0
            or len(scales) != 3
            or any(not math.isfinite(value) or value <= 0 for value in scales)
            or len(counts) != len(donors)
            or any(value < 1 for value in counts)
            or sum(counts) != self.calibration_case_block_count
            or len(donor_quantiles) != len(donors)
            or any(not math.isfinite(value) or value < 0 for value in donor_quantiles)
            or not math.isfinite(float(self.joint_harm_quantile))
            or float(self.joint_harm_quantile) < 0
            or float(self.joint_harm_quantile) != max(donor_quantiles)
        ):
            raise ProtocolError("HARP v4 donor residual calibration is malformed.")
        object.__setattr__(self, "quantile_level", float(self.quantile_level))
        object.__setattr__(self, "endpoint_scales", scales)
        object.__setattr__(self, "joint_harm_quantile", float(self.joint_harm_quantile))
        object.__setattr__(self, "donor_case_counts", counts)
        object.__setattr__(self, "donor_joint_harm_quantiles", donor_quantiles)

    @property
    def endpoint_allowances(self) -> tuple[float, float, float]:
        return tuple(self.joint_harm_quantile * value for value in self.endpoint_scales)


@dataclass(frozen=True)
class ConservativeBounds:
    prediction_center: EffectVector
    case_equal_bacc_contribution_gain_lower: float
    brier_upper: float
    log_loss_upper: float
    calibration_method: str = CALIBRATION_METHOD

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_center, EffectVector) or self.calibration_method != CALIBRATION_METHOD:
            raise ProtocolError("HARP v4 conservative bounds are malformed.")
        for name in (
            "case_equal_bacc_contribution_gain_lower",
            "brier_upper",
            "log_loss_upper",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ProtocolError("HARP v4 conservative bounds must be finite.")
            object.__setattr__(self, name, value)


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, level: float
) -> float:
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not 0.0 <= float(level) <= 1.0
    ):
        raise ProtocolError("HARP v4 weighted calibration quantile is malformed.")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = float(level) * float(np.sum(ordered_weights, dtype=np.float64))
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    index = min(
        len(ordered_values) - 1,
        int(np.searchsorted(cumulative, threshold, side="left")),
    )
    return float(ordered_values[index])


def _finite_sample_upper_quantile(values: Sequence[float], level: float) -> float:
    array = np.sort(np.asarray(tuple(values), dtype=np.float64))
    if (
        array.ndim != 1
        or not len(array)
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or not 0.5 <= float(level) < 1.0
    ):
        raise ProtocolError("HARP v4 finite-sample calibration tail is malformed.")
    rank = min(len(array), int(math.ceil(float(level) * (len(array) + 1))))
    return float(array[rank - 1])


def calibrate_donor_residuals(
    comparison: Comparison,
    predictions: Sequence[EffectVector],
    observed: Sequence[EffectVector],
    donor_ids: Sequence[str],
    case_block_ids: Sequence[str],
    *,
    quantile_level: float,
) -> DonorResidualCalibration:
    predicted_rows = tuple(predictions)
    observed_rows = tuple(observed)
    donors = tuple(str(value) for value in donor_ids)
    blocks = tuple(str(value) for value in case_block_ids)
    if (
        not predicted_rows
        or len(predicted_rows) != len(observed_rows)
        or len(predicted_rows) != len(donors)
        or len(predicted_rows) != len(blocks)
        or any(not donor for donor in donors)
        or any(not block for block in blocks)
        or any(not isinstance(value, EffectVector) for value in (*predicted_rows, *observed_rows))
    ):
        raise ProtocolError("Donor residual calibration rows are invalid or misaligned.")
    predicted_matrix = np.asarray([row.as_tuple() for row in predicted_rows], dtype=np.float64)
    observed_matrix = np.asarray([row.as_tuple() for row in observed_rows], dtype=np.float64)
    # Lower gain needs prediction-observation; upper losses need
    # observation-prediction.  Candidate-action rows are first collapsed to an
    # elementwise maximum within each held-out-donor/case block.  This prevents
    # either a large center or a case with more candidate actions from
    # manufacturing calibration replication.
    one_sided = np.column_stack(
        (
            np.maximum(0.0, predicted_matrix[:, 0] - observed_matrix[:, 0]),
            np.maximum(0.0, observed_matrix[:, 1] - predicted_matrix[:, 1]),
            np.maximum(0.0, observed_matrix[:, 2] - predicted_matrix[:, 2]),
        )
    )
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for donor, block, residual in zip(donors, blocks, one_sided, strict=True):
        grouped.setdefault((donor, block), []).append(residual)
    block_keys = tuple(sorted(grouped))
    block_donors = tuple(key[0] for key in block_keys)
    block_matrix = np.asarray(
        [np.max(np.stack(grouped[key]), axis=0) for key in block_keys],
        dtype=np.float64,
    )
    source_donors = tuple(sorted(set(block_donors)))
    donor_counts = tuple(block_donors.count(donor) for donor in source_donors)
    weights = np.asarray(
        [
            1.0 / (len(source_donors) * donor_counts[source_donors.index(donor)])
            for donor in block_donors
        ],
        dtype=np.float64,
    )
    # Robust scale estimates receive equal total mass per donor and equal mass
    # per case inside a donor.  IQR and an absolute q75 provide deterministic
    # fallbacks when the median absolute deviation degenerates.
    floor = np.sqrt(np.finfo(np.float64).eps)
    scales_list: list[float] = []
    for endpoint in range(block_matrix.shape[1]):
        values = block_matrix[:, endpoint]
        median = _weighted_quantile(values, weights, 0.5)
        mad = 1.4826 * _weighted_quantile(np.abs(values - median), weights, 0.5)
        iqr = (
            _weighted_quantile(values, weights, 0.75)
            - _weighted_quantile(values, weights, 0.25)
        ) / 1.349
        absolute_q75 = _weighted_quantile(np.abs(values), weights, 0.75)
        scale = next(
            (value for value in (mad, iqr, absolute_q75) if value > floor),
            floor,
        )
        scales_list.append(float(scale))
    scales = np.asarray(scales_list, dtype=np.float64)
    joint_scores = np.max(block_matrix / scales, axis=1)
    donor_quantiles = tuple(
        _finite_sample_upper_quantile(
            joint_scores[
                np.asarray([value == donor for value in block_donors], dtype=bool)
            ],
            quantile_level,
        )
        for donor in source_donors
    )
    return DonorResidualCalibration(
        comparison=Comparison(comparison),
        quantile_level=quantile_level,
        endpoint_scales=tuple(float(value) for value in scales),
        joint_harm_quantile=max(donor_quantiles),
        calibration_row_count=len(predicted_rows),
        calibration_case_block_count=len(block_keys),
        donor_ids=source_donors,
        donor_case_counts=donor_counts,
        donor_joint_harm_quantiles=donor_quantiles,
    )


def conservative_bounds(
    predictions: Sequence[EffectVector],
    calibration: DonorResidualCalibration,
    *,
    compatibility_shrinkage: float,
) -> ConservativeBounds:
    rows = tuple(predictions)
    rho = float(compatibility_shrinkage)
    if (
        not rows
        or any(not isinstance(row, EffectVector) for row in rows)
        or not isinstance(calibration, DonorResidualCalibration)
        or not math.isfinite(rho)
        or not 0.0 <= rho <= 1.0
    ):
        raise ProtocolError("HARP v4 bound inputs are invalid.")
    matrix = np.asarray([row.as_tuple() for row in rows], dtype=np.float64)
    gain, brier, log_loss = (float(value) for value in np.median(matrix, axis=0))
    # Geometry may only reduce favorable evidence: positive gain and negative
    # loss deltas shrink toward the neutral value zero; harmful evidence stays.
    gain = rho * gain if gain > 0 else gain
    brier = rho * brier if brier < 0 else brier
    log_loss = rho * log_loss if log_loss < 0 else log_loss
    center = EffectVector(gain, brier, log_loss)
    gain_allowance, brier_allowance, log_loss_allowance = calibration.endpoint_allowances
    return ConservativeBounds(
        prediction_center=center,
        case_equal_bacc_contribution_gain_lower=gain - gain_allowance,
        brier_upper=brier + brier_allowance,
        log_loss_upper=log_loss + log_loss_allowance,
    )


__all__ = (
    "CALIBRATION_METHOD",
    "FINITE_SAMPLE_RULE",
    "ConservativeBounds",
    "DonorResidualCalibration",
    "calibrate_donor_residuals",
    "conservative_bounds",
)
