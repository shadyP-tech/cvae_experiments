"""Menu-wise one-sided risk certificates for HARP v16 actions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import CasePrediction, EndpointPrediction, RouterFitConfig
from .crossfit import SupportOOFRecord
from .hashing import canonical_hash, require_sha256


def _corrected_upper_quantile(values: Sequence[float], *, alpha: float) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ProtocolError("HARP v16 certificate residuals are malformed.")
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - float(alpha)))) - 1
    return max(0.0, ordered[min(max(rank, 0), len(ordered) - 1)])


@dataclass(frozen=True, slots=True)
class MenuRiskCalibration:
    outer_target_id: str
    support_case_ids: tuple[str, ...]
    active_calibration_case_ids: tuple[str, ...]
    gain_lower_offset: float
    harm_upper_offset: float
    brier_upper_offset: float
    log_loss_upper_offset: float
    alpha: float
    support_crossfit_hash: str
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(sorted(str(value) for value in self.support_case_ids))
        active_cases = tuple(
            sorted(str(value) for value in self.active_calibration_case_ids)
        )
        offsets = tuple(
            float(value)
            for value in (
                self.gain_lower_offset,
                self.harm_upper_offset,
                self.brier_upper_offset,
                self.log_loss_upper_offset,
            )
        )
        if (
            not self.outer_target_id
            or not cases
            or len(cases) != len(set(cases))
            or len(active_cases) != len(set(active_cases))
            or not set(active_cases).issubset(cases)
            or any(not math.isfinite(value) or value < 0.0 for value in offsets)
            or not 0.0 < float(self.alpha) < 1.0
        ):
            raise ProtocolError("HARP v16 menu-risk calibration is malformed.")
        support_hash = require_sha256(
            self.support_crossfit_hash, name="certificate support crossfit hash"
        )
        object.__setattr__(self, "support_case_ids", cases)
        object.__setattr__(self, "active_calibration_case_ids", active_cases)
        object.__setattr__(self, "gain_lower_offset", offsets[0])
        object.__setattr__(self, "harm_upper_offset", offsets[1])
        object.__setattr__(self, "brier_upper_offset", offsets[2])
        object.__setattr__(self, "log_loss_upper_offset", offsets[3])
        object.__setattr__(self, "support_crossfit_hash", support_hash)
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_menu_risk_calibration_v16",
                    "outer_target_id": self.outer_target_id,
                    "support_case_ids": cases,
                    "active_calibration_case_ids": active_cases,
                    "gain_lower_offset": offsets[0],
                    "harm_upper_offset": offsets[1],
                    "brier_upper_offset": offsets[2],
                    "log_loss_upper_offset": offsets[3],
                    "alpha": float(self.alpha),
                    "support_crossfit_hash": support_hash,
                    "case_level_max_residual_calibration": True,
                    "exact_b_controls_excluded_from_residuals": True,
                    "simultaneous_within_menu": True,
                    "formal_target_exchangeability_claimed": False,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_support_menu_risk_calibration_v16",
            "outer_target_id": self.outer_target_id,
            "support_case_ids": list(self.support_case_ids),
            "active_calibration_case_ids": list(self.active_calibration_case_ids),
            "gain_lower_offset": self.gain_lower_offset,
            "harm_upper_offset": self.harm_upper_offset,
            "brier_upper_offset": self.brier_upper_offset,
            "log_loss_upper_offset": self.log_loss_upper_offset,
            "alpha": self.alpha,
            "support_crossfit_hash": self.support_crossfit_hash,
            "calibration_hash": self.calibration_hash,
            "case_level_max_residual_calibration": True,
            "exact_b_controls_excluded_from_residuals": True,
            "formal_target_exchangeability_claimed": False,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class ActionRiskCertificate:
    prediction: EndpointPrediction
    gain_lcb: float
    harm_ucb: float
    brier_delta_ucb: float
    log_loss_delta_ucb: float
    gain_passed: bool
    harm_passed: bool
    brier_passed: bool
    log_loss_passed: bool
    calibration_hash: str
    certificate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        bounds = tuple(
            float(value)
            for value in (
                self.gain_lcb,
                self.harm_ucb,
                self.brier_delta_ucb,
                self.log_loss_delta_ucb,
            )
        )
        if (
            not isinstance(self.prediction, EndpointPrediction)
            or any(not math.isfinite(value) for value in bounds)
        ):
            raise ProtocolError("HARP v16 action certificate is malformed.")
        calibration_hash = require_sha256(
            self.calibration_hash, name="certificate calibration hash"
        )
        object.__setattr__(self, "calibration_hash", calibration_hash)
        object.__setattr__(
            self,
            "certificate_hash",
            canonical_hash(
                {
                    "schema_version": "hierarchical_support_action_risk_certificate_v16",
                    "prediction_hash": self.prediction.prediction_hash,
                    "gain_lcb": bounds[0],
                    "harm_ucb": bounds[1],
                    "brier_delta_ucb": bounds[2],
                    "log_loss_delta_ucb": bounds[3],
                    "gain_passed": bool(self.gain_passed),
                    "harm_passed": bool(self.harm_passed),
                    "brier_passed": bool(self.brier_passed),
                    "log_loss_passed": bool(self.log_loss_passed),
                    "calibration_hash": calibration_hash,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def passed(self) -> bool:
        return (
            self.gain_passed
            and self.harm_passed
            and self.brier_passed
            and self.log_loss_passed
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "action_id": self.prediction.action.action_id,
            "action_hash": self.prediction.action.action_hash,
            "prediction_hash": self.prediction.prediction_hash,
            "gain_lcb": self.gain_lcb,
            "harm_ucb": self.harm_ucb,
            "brier_delta_ucb": self.brier_delta_ucb,
            "log_loss_delta_ucb": self.log_loss_delta_ucb,
            "gain_passed": self.gain_passed,
            "harm_passed": self.harm_passed,
            "brier_passed": self.brier_passed,
            "log_loss_passed": self.log_loss_passed,
            "passed": self.passed,
            "calibration_hash": self.calibration_hash,
            "certificate_hash": self.certificate_hash,
            "evaluation_labels_consumed": False,
        }


def fit_menu_risk_calibration(
    records: Sequence[SupportOOFRecord],
    *,
    alpha: float,
    support_crossfit_hash: str,
    support_case_ids: Sequence[str] | None = None,
    outer_target_id: str | None = None,
) -> MenuRiskCalibration:
    rows = tuple(records)
    outer_ids = {row.prediction.action.outer_target_id for row in rows}
    by_case: dict[str, list[SupportOOFRecord]] = {}
    for row in rows:
        by_case.setdefault(row.prediction.action.case_id, []).append(row)
    cases = tuple(
        sorted(
            str(value)
            for value in (
                tuple(by_case) if support_case_ids is None else tuple(support_case_ids)
            )
        )
    )
    if not cases or len(cases) != len(set(cases)) or not set(by_case).issubset(cases):
        raise ProtocolError("HARP v16 calibration support inventory is malformed.")
    if outer_target_id is None:
        if len(outer_ids) != 1:
            raise ProtocolError("HARP v16 calibration lacks one outer target H.")
        outer = next(iter(outer_ids))
    else:
        outer = str(outer_target_id)
        if outer_ids and outer_ids != {outer}:
            raise ProtocolError("HARP v16 calibration crossed outer targets.")
    gain_errors: list[float] = []
    harm_errors: list[float] = []
    brier_errors: list[float] = []
    log_errors: list[float] = []
    for case_id in sorted(by_case):
        case_rows = by_case[case_id]
        gain_errors.append(
            max(row.prediction.predicted_gain - row.outcome.bacc_gain for row in case_rows)
        )
        harm_errors.append(
            max(
                float(row.outcome.harmed) - row.prediction.predicted_harm_probability
                for row in case_rows
            )
        )
        brier_errors.append(
            max(
                row.outcome.brier_delta - row.prediction.predicted_brier_delta
                for row in case_rows
            )
        )
        log_errors.append(
            max(
                row.outcome.log_loss_delta - row.prediction.predicted_log_loss_delta
                for row in case_rows
            )
        )
    return MenuRiskCalibration(
        outer_target_id=outer,
        support_case_ids=cases,
        active_calibration_case_ids=tuple(sorted(by_case)),
        gain_lower_offset=(
            0.0 if not gain_errors else _corrected_upper_quantile(gain_errors, alpha=alpha)
        ),
        harm_upper_offset=(
            0.0 if not harm_errors else _corrected_upper_quantile(harm_errors, alpha=alpha)
        ),
        brier_upper_offset=(
            0.0 if not brier_errors else _corrected_upper_quantile(brier_errors, alpha=alpha)
        ),
        log_loss_upper_offset=(
            0.0 if not log_errors else _corrected_upper_quantile(log_errors, alpha=alpha)
        ),
        alpha=float(alpha),
        support_crossfit_hash=support_crossfit_hash,
    )


def certify_case_prediction(
    prediction: CasePrediction,
    calibration: MenuRiskCalibration,
    *,
    config: RouterFitConfig,
) -> tuple[ActionRiskCertificate, ...]:
    rows = prediction.action_predictions
    if rows and any(
        row.action.outer_target_id != calibration.outer_target_id for row in rows
    ):
        raise ProtocolError("HARP v16 calibration crossed an outer-target boundary.")
    certificates: list[ActionRiskCertificate] = []
    for row in rows:
        gain_lcb = row.predicted_gain - calibration.gain_lower_offset
        harm_ucb = min(1.0, row.predicted_harm_probability + calibration.harm_upper_offset)
        brier_ucb = row.predicted_brier_delta + calibration.brier_upper_offset
        log_ucb = row.predicted_log_loss_delta + calibration.log_loss_upper_offset
        certificates.append(
            ActionRiskCertificate(
                prediction=row,
                gain_lcb=gain_lcb,
                harm_ucb=harm_ucb,
                brier_delta_ucb=brier_ucb,
                log_loss_delta_ucb=log_ucb,
                gain_passed=gain_lcb > config.minimum_gain_lcb,
                harm_passed=harm_ucb <= config.maximum_harm_ucb,
                brier_passed=brier_ucb <= config.maximum_brier_delta_ucb,
                log_loss_passed=log_ucb <= config.maximum_log_loss_delta_ucb,
                calibration_hash=calibration.calibration_hash,
            )
        )
    return tuple(sorted(certificates, key=lambda row: row.prediction.action.action_id))


__all__ = (
    "ActionRiskCertificate",
    "MenuRiskCalibration",
    "certify_case_prediction",
    "fit_menu_risk_calibration",
)
