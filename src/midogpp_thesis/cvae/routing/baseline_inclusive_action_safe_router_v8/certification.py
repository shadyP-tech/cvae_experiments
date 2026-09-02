"""Center-group source-OOF residual calibration for action safety."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import ActionCertificate, ActionEstimate, SourceActionOutcome, finite
from .hashing import canonical_hash


UNAVAILABLE_CALIBRATION_HASH = canonical_hash(
    {"schema_version": "baseline_inclusive_unavailable_calibration_v8"}
)


@dataclass(frozen=True, slots=True)
class ResidualObservation:
    query_center_id: str
    estimate: ActionEstimate
    outcome: SourceActionOutcome

    def __post_init__(self) -> None:
        if (
            self.query_center_id != self.outcome.action.query_center_id
            or self.estimate.action_id != self.outcome.action.action_id
            or self.estimate.action_hash != self.outcome.action.action_hash
            or self.estimate.action_group
            != f"{self.outcome.action.action_kind}:{self.outcome.action.direction.value}"
        ):
            raise ProtocolError("HARP v8 residual row is not action/outcome aligned.")


@dataclass(frozen=True, slots=True)
class ResidualCalibrationCell:
    action_group: str
    calibration_center_ids: tuple[str, ...]
    residual_quantile: float
    gain_shortfall_radius: float
    harm_excess_radius: float
    brier_excess_radius: float
    log_excess_radius: float
    harm_brier_risk: float
    harm_log_loss_risk: float
    row_count: int
    available: bool
    cell_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(sorted(self.calibration_center_ids))
        values = tuple(
            finite(value, name="residual calibration value")
            for value in (
                self.residual_quantile,
                self.gain_shortfall_radius,
                self.harm_excess_radius,
                self.brier_excess_radius,
                self.log_excess_radius,
                self.harm_brier_risk,
                self.harm_log_loss_risk,
            )
        )
        if (
            len(set(centers)) != len(centers)
            or not 0.0 < values[0] <= 1.0
            or any(value < 0.0 for value in values[1:])
            or self.row_count < 0
            or (self.available and (not centers or self.row_count < 1))
        ):
            raise ProtocolError("HARP v8 residual calibration cell is malformed.")
        object.__setattr__(self, "calibration_center_ids", centers)
        for name, value in zip(
            (
                "residual_quantile",
                "gain_shortfall_radius",
                "harm_excess_radius",
                "brier_excess_radius",
                "log_excess_radius",
                "harm_brier_risk",
                "harm_log_loss_risk",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "cell_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_center_group_residual_cell_v8",
                    "action_group": self.action_group,
                    "calibration_center_ids": centers,
                    "residual_quantile": values[0],
                    "gain_shortfall_radius": values[1],
                    "harm_excess_radius": values[2],
                    "brier_excess_radius": values[3],
                    "log_excess_radius": values[4],
                    "harm_brier_risk": values[5],
                    "harm_log_loss_risk": values[6],
                    "row_count": self.row_count,
                    "available": bool(self.available),
                    "center_group_max_envelope": True,
                    "evaluation_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResidualCalibration:
    outer_target_id: str
    cells: tuple[ResidualCalibrationCell, ...]
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cells = tuple(sorted(self.cells, key=lambda row: row.action_group))
        if len({row.action_group for row in cells}) != len(cells):
            raise ProtocolError("HARP v8 residual calibration contains duplicate cells.")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "baseline_inclusive_residual_calibration_v8",
                    "outer_target_id": self.outer_target_id,
                    "cell_hashes": tuple(row.cell_hash for row in cells),
                    "fit_surface": "STRICT_SOURCE_CENTER_OOF_ONLY",
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def for_group(self, action_group: str) -> ResidualCalibrationCell | None:
        return next((row for row in self.cells if row.action_group == action_group), None)


def _higher_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    ordinal = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[ordinal]


def _center_equal_mean(by_center: dict[str, list[float]]) -> float:
    means = [sum(values) / len(values) for _, values in sorted(by_center.items()) if values]
    return sum(means) / len(means) if means else 0.0


def calibrate_center_group_residuals(
    rows: Sequence[ResidualObservation],
    *,
    outer_target_id: str,
    residual_quantile: float,
    min_calibration_centers: int,
    min_calibration_rows_per_group: int,
) -> ResidualCalibration:
    """Build one-sided envelopes without a pooled fallback.

    A residual quantile is computed separately inside each source center and
    the worst center envelope is retained.  This prevents a large center from
    washing out a smaller harmful center.  Missing/sparse action groups remain
    explicitly unavailable and therefore cannot enter a target safe set.
    """

    if (
        not 0.0 < float(residual_quantile) <= 1.0
        or int(min_calibration_centers) < 1
        or int(min_calibration_rows_per_group) < 1
    ):
        raise ProtocolError("HARP v8 residual calibration configuration is malformed.")
    grouped: dict[str, list[ResidualObservation]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, ResidualObservation):
            raise ProtocolError("HARP v8 residual calibration requires typed OOF rows.")
        if (
            row.outcome.action.outer_target_id != outer_target_id
            or row.query_center_id == outer_target_id
        ):
            raise ProtocolError("HARP v8 residual calibration crossed the outer target.")
        grouped[row.estimate.action_group].append(row)
    cells: list[ResidualCalibrationCell] = []
    epsilon = 1e-12
    for group, members in sorted(grouped.items()):
        residuals: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        risk_brier: dict[str, list[float]] = defaultdict(list)
        risk_log: dict[str, list[float]] = defaultdict(list)
        for row in members:
            center = row.query_center_id
            observed_harm = float(row.outcome.bacc_gain < 0.0)
            predicted_harm = min(max(row.estimate.predicted_harm_probability, epsilon), 1.0 - epsilon)
            residuals[center]["gain"].append(
                max(row.estimate.predicted_bacc_gain - row.outcome.bacc_gain, 0.0)
            )
            residuals[center]["harm"].append(max(observed_harm - predicted_harm, 0.0))
            residuals[center]["brier"].append(
                max(row.outcome.brier_delta - row.estimate.predicted_brier_delta, 0.0)
            )
            residuals[center]["log"].append(
                max(row.outcome.log_delta - row.estimate.predicted_log_delta, 0.0)
            )
            risk_brier[center].append((predicted_harm - observed_harm) ** 2)
            risk_log[center].append(
                -(
                    observed_harm * math.log(predicted_harm)
                    + (1.0 - observed_harm) * math.log(1.0 - predicted_harm)
                )
            )
        centers = tuple(sorted(residuals))
        available = bool(
            len(centers) >= int(min_calibration_centers)
            and len(members) >= int(min_calibration_rows_per_group)
        )

        def radius(endpoint: str) -> float:
            return max(
                (_higher_quantile(residuals[center][endpoint], residual_quantile) for center in centers),
                default=0.0,
            )

        cells.append(
            ResidualCalibrationCell(
                action_group=group,
                calibration_center_ids=centers,
                residual_quantile=float(residual_quantile),
                gain_shortfall_radius=radius("gain"),
                harm_excess_radius=radius("harm"),
                brier_excess_radius=radius("brier"),
                log_excess_radius=radius("log"),
                harm_brier_risk=_center_equal_mean(risk_brier),
                harm_log_loss_risk=_center_equal_mean(risk_log),
                row_count=len(members),
                available=available,
            )
        )
    return ResidualCalibration(outer_target_id=outer_target_id, cells=tuple(cells))


def certify_action(
    estimate: ActionEstimate,
    cell: ResidualCalibrationCell | None,
    *,
    max_harm_probability: float,
    max_brier_delta: float,
    max_log_delta: float,
    max_harm_brier_risk: float,
    max_harm_log_loss_risk: float,
) -> ActionCertificate:
    """Apply the frozen source-only endpoint-safety contract to one estimate.

    ``gain_lcb`` remains a persisted diagnostic.  It is deliberately not an
    action-admission gate: v8 exploits source-OOF relative ordering inside the
    harm/proper-loss-safe action set and requires positive utility from the
    nested, whole-policy replay instead.
    """

    if not isinstance(estimate, ActionEstimate):
        raise ProtocolError("HARP v8 certification requires a typed estimate.")
    if cell is None:
        conservative = 1e300
        return ActionCertificate(
            estimate=estimate,
            gain_lcb=estimate.predicted_bacc_gain,
            harm_probability_ucb=1.0,
            brier_delta_ucb=conservative,
            log_delta_ucb=conservative,
            harm_brier_risk=conservative,
            harm_log_loss_risk=conservative,
            calibration_cell_hash=UNAVAILABLE_CALIBRATION_HASH,
            safe=False,
            failed_gates=("RESIDUAL_CALIBRATION_UNAVAILABLE",),
        )
    gain_lcb = estimate.predicted_bacc_gain - cell.gain_shortfall_radius
    harm_ucb = min(1.0, estimate.predicted_harm_probability + cell.harm_excess_radius)
    brier_ucb = estimate.predicted_brier_delta + cell.brier_excess_radius
    log_ucb = estimate.predicted_log_delta + cell.log_excess_radius
    failed: list[str] = []
    if not estimate.model_available:
        failed.append("ACTION_HEAD_UNAVAILABLE")
    if not cell.available:
        failed.append("RESIDUAL_CALIBRATION_UNAVAILABLE")
    if harm_ucb > max_harm_probability:
        failed.append("HARM_PROBABILITY_UCB_ABOVE_CEILING")
    if brier_ucb > max_brier_delta:
        failed.append("BRIER_DELTA_UCB_ABOVE_CEILING")
    if log_ucb > max_log_delta:
        failed.append("LOG_DELTA_UCB_ABOVE_CEILING")
    if cell.harm_brier_risk > max_harm_brier_risk:
        failed.append("HARM_BRIER_RISK_ABOVE_CEILING")
    if cell.harm_log_loss_risk > max_harm_log_loss_risk:
        failed.append("HARM_LOG_LOSS_RISK_ABOVE_CEILING")
    return ActionCertificate(
        estimate=estimate,
        gain_lcb=gain_lcb,
        harm_probability_ucb=harm_ucb,
        brier_delta_ucb=brier_ucb,
        log_delta_ucb=log_ucb,
        harm_brier_risk=cell.harm_brier_risk,
        harm_log_loss_risk=cell.harm_log_loss_risk,
        calibration_cell_hash=cell.cell_hash,
        safe=not failed,
        failed_gates=tuple(failed),
    )


__all__ = (
    "ResidualCalibration",
    "ResidualCalibrationCell",
    "ResidualObservation",
    "UNAVAILABLE_CALIBRATION_HASH",
    "calibrate_center_group_residuals",
    "certify_action",
)
