"""Selection-aware descriptive uncertainty envelopes for SCALE-BP."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .empirical_bayes import EmpiricalBayesEstimate
from .hashing import canonical_hash
from .influence.contracts import MetricStandardError, require_sha256
from .local_residual.contracts import LocalCrossfitResult
from .protocol import ProtocolError


SELECTION_RESIDUAL_QUANTILE = 0.90
STANDARD_ERROR_MULTIPLIER = 1.0


@dataclass(frozen=True, slots=True)
class SelectionAwareRadius:
    crossfit_hash: str
    quantile: float
    member_count: int
    member_maxima: tuple[tuple[str, float, float, float], ...]
    radius: MetricStandardError
    radius_hash: str = field(init=False)

    def __post_init__(self) -> None:
        crossfit_hash = require_sha256(self.crossfit_hash, "selection-radius crossfit hash")
        quantile = float(self.quantile)
        member_count = int(self.member_count)
        maxima = tuple(
            (str(member_id), float(bacc), float(brier), float(log))
            for member_id, bacc, brier, log in self.member_maxima
        )
        if (
            not math.isclose(
                quantile, SELECTION_RESIDUAL_QUANTILE, rel_tol=0.0, abs_tol=0.0
            )
            or member_count < 4
            or len(maxima) != member_count
            or maxima != tuple(sorted(maxima, key=lambda row: row[0]))
            or len({row[0] for row in maxima}) != len(maxima)
            or any(
                not math.isfinite(value) or value < 0.0
                for row in maxima
                for value in row[1:]
            )
        ):
            raise ProtocolError("SCALE-BP selection-aware radius drifted.")
        payload = {
            "schema_version": "scale_bp_selection_aware_radius_v1",
            "crossfit_hash": crossfit_hash,
            "quantile": quantile,
            "member_count": member_count,
            "member_maxima": maxima,
            "radius": self.radius.to_payload(),
            "max_over_actions_before_quantile": True,
            "confidence_or_conformal_claimed": False,
        }
        object.__setattr__(self, "crossfit_hash", crossfit_hash)
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "member_count", member_count)
        object.__setattr__(self, "member_maxima", maxima)
        object.__setattr__(self, "radius_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_selection_aware_radius_v1",
            "crossfit_hash": self.crossfit_hash,
            "quantile": self.quantile,
            "member_count": self.member_count,
            "member_maxima": self.member_maxima,
            "radius": self.radius.to_payload(),
            "max_over_actions_before_quantile": True,
            "confidence_or_conformal_claimed": False,
            "radius_hash": self.radius_hash,
        }


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    estimate: EmpiricalBayesEstimate
    radius: SelectionAwareRadius
    standard_error_multiplier: float
    bacc_lower: float
    brier_upper: float
    log_upper: float
    envelope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        multiplier = float(self.standard_error_multiplier)
        bacc_lower = float(self.bacc_lower)
        brier_upper = float(self.brier_upper)
        log_upper = float(self.log_upper)
        point = self.estimate.posterior_metrics
        se = self.estimate.posterior_standard_error
        residual = self.radius.radius
        expected = (
            point.bacc_gain - multiplier * se.bacc - residual.bacc,
            point.brier_loss_delta + multiplier * se.brier + residual.brier,
            point.log_loss_delta + multiplier * se.log + residual.log,
        )
        if (
            not math.isclose(
                multiplier, STANDARD_ERROR_MULTIPLIER, rel_tol=0.0, abs_tol=0.0
            )
            or not all(math.isfinite(value) for value in (bacc_lower, brier_upper, log_upper))
            or any(
                not math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12)
                for actual, target in zip(
                    (bacc_lower, brier_upper, log_upper), expected, strict=True
                )
            )
        ):
            raise ProtocolError("SCALE-BP action uncertainty envelope drifted.")
        payload = {
            "schema_version": "scale_bp_action_envelope_v1",
            "action_id": self.estimate.action_id,
            "estimate_hash": self.estimate.estimate_hash,
            "radius_hash": self.radius.radius_hash,
            "standard_error_multiplier": multiplier,
            "bacc_lower": bacc_lower,
            "brier_upper": brier_upper,
            "log_upper": log_upper,
            "confidence_or_conformal_claimed": False,
        }
        object.__setattr__(self, "standard_error_multiplier", multiplier)
        object.__setattr__(self, "bacc_lower", bacc_lower)
        object.__setattr__(self, "brier_upper", brier_upper)
        object.__setattr__(self, "log_upper", log_upper)
        object.__setattr__(self, "envelope_hash", canonical_hash(payload))

    @property
    def action_id(self) -> str:
        return self.estimate.action_id

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_action_envelope_v1",
            "action_id": self.action_id,
            "estimate_hash": self.estimate.estimate_hash,
            "radius_hash": self.radius.radius_hash,
            "standard_error_multiplier": self.standard_error_multiplier,
            "bacc_lower": self.bacc_lower,
            "brier_upper": self.brier_upper,
            "log_upper": self.log_upper,
            "confidence_or_conformal_claimed": False,
            "envelope_hash": self.envelope_hash,
        }


def _higher_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ProtocolError("SCALE-BP residual-radius population is empty.")
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def fit_selection_aware_radius(
    crossfit: LocalCrossfitResult,
    *,
    quantile: float = SELECTION_RESIDUAL_QUANTILE,
) -> SelectionAwareRadius:
    """Take a case-wise maximum over actions before the fixed upper quantile."""

    if float(quantile) != SELECTION_RESIDUAL_QUANTILE:
        raise ProtocolError("SCALE-BP selection-radius quantile is not frozen.")
    by_member: dict[str, list[tuple[float, float, float]]] = {}
    for prediction in crossfit.predictions:
        error = prediction.residual_error.as_tuple()
        by_member.setdefault(prediction.member_id, []).append(
            tuple(abs(value) for value in error)
        )
    maxima = []
    for member_id, rows in sorted(by_member.items()):
        maxima.append(
            (
                member_id,
                max(row[0] for row in rows),
                max(row[1] for row in rows),
                max(row[2] for row in rows),
            )
        )
    if len(maxima) < 4:
        raise ProtocolError("SCALE-BP selection-radius support is insufficient.")
    radius = MetricStandardError(
        _higher_quantile([row[1] for row in maxima], quantile),
        _higher_quantile([row[2] for row in maxima], quantile),
        _higher_quantile([row[3] for row in maxima], quantile),
    )
    return SelectionAwareRadius(
        crossfit_hash=crossfit.crossfit_hash,
        quantile=quantile,
        member_count=len(maxima),
        member_maxima=tuple(maxima),
        radius=radius,
    )


def build_action_envelope(
    estimate: EmpiricalBayesEstimate,
    radius: SelectionAwareRadius,
    *,
    standard_error_multiplier: float = STANDARD_ERROR_MULTIPLIER,
) -> ActionEnvelope:
    if float(standard_error_multiplier) != STANDARD_ERROR_MULTIPLIER:
        raise ProtocolError("SCALE-BP standard-error multiplier is not frozen.")
    point = estimate.posterior_metrics
    se = estimate.posterior_standard_error
    residual = radius.radius
    return ActionEnvelope(
        estimate=estimate,
        radius=radius,
        standard_error_multiplier=standard_error_multiplier,
        bacc_lower=point.bacc_gain - standard_error_multiplier * se.bacc - residual.bacc,
        brier_upper=(
            point.brier_loss_delta + standard_error_multiplier * se.brier + residual.brier
        ),
        log_upper=(point.log_loss_delta + standard_error_multiplier * se.log + residual.log),
    )


__all__ = (
    "ActionEnvelope",
    "SELECTION_RESIDUAL_QUANTILE",
    "STANDARD_ERROR_MULTIPLIER",
    "SelectionAwareRadius",
    "build_action_envelope",
    "fit_selection_aware_radius",
)
