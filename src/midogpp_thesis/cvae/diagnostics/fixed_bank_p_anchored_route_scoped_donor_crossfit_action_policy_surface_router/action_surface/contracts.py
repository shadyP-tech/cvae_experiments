"""Immutable, pickle-safe contracts for the P-DCAPS action surface.

The contracts deliberately contain only primitives, tuples, and the small
immutable contracts owned by the P-DCAPS package.  In particular, fitted
estimators, mappings, closures, labels, and mutable array views never cross a
worker boundary or enter a persisted payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import BankViability, FavorableUtility, RouteKey
from ..identity import (
    ACTION_FAMILIES,
    ACTION_STRATA,
    DIRECTIONS,
    METRICS,
    RIDGE_ALPHA,
    TIE_TOLERANCE,
    canonical_hash,
    require_sha256,
)


def _finite_tuple(values: object, *, role: str) -> tuple[float, ...]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ProtocolError(f"P-DCAPS {role} must be a finite one-dimensional vector.")
    return tuple(float(value) for value in array)


def _optional_center(value: object | None, *, role: str) -> str | None:
    if value is None:
        return None
    center = str(value)
    if center not in CENTERS:
        raise ProtocolError(f"P-DCAPS {role} is not a canonical center.")
    return center


@dataclass(frozen=True, order=True)
class ActionKey:
    """Identity of one already-sealed action on a target or pseudo route."""

    route_key: RouteKey
    family: str
    direction: str
    action_id: str
    probability_hash: str
    action_surface_seal_hash: str
    action_key_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        family = str(self.family)
        direction = str(self.direction)
        action_id = str(self.action_id)
        if family not in ACTION_FAMILIES or direction not in DIRECTIONS or not action_id:
            raise ProtocolError("P-DCAPS action identity drifted.")
        require_sha256(self.probability_hash, "action probability hash")
        require_sha256(self.action_surface_seal_hash, "action-surface seal hash")
        payload = {
            "schema_version": "pdcaps_action_key_v1",
            "route_key": self.route_key.to_payload(),
            "family": family,
            "direction": direction,
            "action_id": action_id,
            "probability_hash": self.probability_hash,
            "action_surface_seal_hash": self.action_surface_seal_hash,
        }
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_key_hash", canonical_hash(payload))

    @property
    def stratum(self) -> tuple[str, str]:
        return self.family, self.direction

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_key_v1",
            "route_key": self.route_key.to_payload(),
            "family": self.family,
            "direction": self.direction,
            "action_id": self.action_id,
            "probability_hash": self.probability_hash,
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "action_key_hash": self.action_key_hash,
        }


@dataclass(frozen=True)
class ActionPrediction:
    """Label-free prediction and descriptor inputs for one sealed action."""

    key: ActionKey
    predicted_utility: FavorableUtility
    crossing_fraction: float
    bank_viability: BankViability
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        crossing = float(self.crossing_fraction)
        if not math.isfinite(crossing) or crossing < 0.0 or crossing > 1.0:
            raise ProtocolError("P-DCAPS crossing fraction is outside [0, 1].")
        payload = {
            "schema_version": "pdcaps_action_prediction_v1",
            "action_key_hash": self.key.action_key_hash,
            "predicted_utility": self.predicted_utility.to_payload(),
            "crossing_fraction": crossing,
            "bank_viability": self.bank_viability.to_payload(),
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "crossing_fraction", crossing)
        object.__setattr__(self, "prediction_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_prediction_v1",
            "key": self.key.to_payload(),
            "predicted_utility": self.predicted_utility.to_payload(),
            "crossing_fraction": self.crossing_fraction,
            "bank_viability": self.bank_viability.to_payload(),
            "raw_labels_persisted": False,
            "prediction_hash": self.prediction_hash,
        }


@dataclass(frozen=True)
class ActionResponse:
    """Realized favorable utility with fixed denominator provenance.

    Only aggregate counts and hashes survive construction.  The labels and the
    two probability arrays used to compute the response are intentionally not
    fields of this DTO.
    """

    key: ActionKey
    prediction_hash: str
    realized_utility: FavorableUtility
    label_count: int
    positive_denominator: int
    negative_denominator: int
    row_denominator: int
    baseline_probability_hash: str
    evaluation_row_hash: str
    response_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.key.route_key.surface_role != "pseudo":
            raise ProtocolError("P-DCAPS action responses are donor-pseudo evidence only.")
        require_sha256(self.prediction_hash, "action prediction hash")
        require_sha256(self.baseline_probability_hash, "baseline probability hash")
        require_sha256(self.evaluation_row_hash, "evaluation row-order hash")
        label_count = int(self.label_count)
        positive = int(self.positive_denominator)
        negative = int(self.negative_denominator)
        total = int(self.row_denominator)
        if (
            label_count <= 0
            or positive <= 0
            or negative <= 0
            or total != positive + negative
            or label_count > total
        ):
            raise ProtocolError("P-DCAPS action-response denominators drifted.")
        payload = {
            "schema_version": "pdcaps_action_response_v1",
            "action_key_hash": self.key.action_key_hash,
            "prediction_hash": self.prediction_hash,
            "realized_utility": self.realized_utility.to_payload(),
            "label_count": label_count,
            "positive_denominator": positive,
            "negative_denominator": negative,
            "row_denominator": total,
            "baseline_probability_hash": self.baseline_probability_hash,
            "evaluation_row_hash": self.evaluation_row_hash,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "label_count", label_count)
        object.__setattr__(self, "positive_denominator", positive)
        object.__setattr__(self, "negative_denominator", negative)
        object.__setattr__(self, "row_denominator", total)
        object.__setattr__(self, "response_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_response_v1",
            "key": self.key.to_payload(),
            "prediction_hash": self.prediction_hash,
            "realized_utility": self.realized_utility.to_payload(),
            "label_count": self.label_count,
            "positive_denominator": self.positive_denominator,
            "negative_denominator": self.negative_denominator,
            "row_denominator": self.row_denominator,
            "baseline_probability_hash": self.baseline_probability_hash,
            "evaluation_row_hash": self.evaluation_row_hash,
            "raw_labels_persisted": False,
            "response_hash": self.response_hash,
        }


@dataclass(frozen=True)
class ActionCalibrationModel:
    """One serialized metric-specific weighted-ridge model."""

    metric: str
    excluded_outer_center: str
    excluded_scored_center: str | None
    training_centers: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    ridge_alpha: float
    training_row_count: int
    training_response_hash: str
    weight_audit_hash: str
    solver: str
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        metric = str(self.metric)
        outer = str(self.excluded_outer_center)
        scored = _optional_center(self.excluded_scored_center, role="model scored exclusion")
        centers = tuple(str(center) for center in self.training_centers)
        names = tuple(str(name) for name in self.feature_names)
        mean = _finite_tuple(self.feature_mean, role="model feature mean")
        scale = _finite_tuple(self.feature_scale, role="model feature scale")
        coefficients = _finite_tuple(self.coefficients, role="model coefficients")
        intercept = float(self.intercept)
        alpha = float(self.ridge_alpha)
        if (
            metric not in METRICS
            or outer not in CENTERS
            or scored == outer
            or not centers
            or len(centers) != len(set(centers))
            or tuple(sorted(centers, key=CENTERS.index)) != centers
            or outer in centers
            or (scored is not None and scored in centers)
            or any(center not in CENTERS for center in centers)
            or not names
            or len(names) != len(set(names))
            or not (len(names) == len(mean) == len(scale) == len(coefficients))
            or any(value <= 0.0 for value in scale)
            or not math.isfinite(intercept)
            or not math.isclose(alpha, RIDGE_ALPHA, abs_tol=0.0, rel_tol=0.0)
            or int(self.training_row_count) <= 0
            or self.solver not in {"solve", "pinv"}
        ):
            raise ProtocolError("P-DCAPS action calibration model drifted.")
        require_sha256(self.training_response_hash, "model training-response hash")
        require_sha256(self.weight_audit_hash, "model weight-audit hash")
        payload = {
            "schema_version": "pdcaps_action_calibration_model_v1",
            "metric": metric,
            "excluded_outer_center": outer,
            "excluded_scored_center": scored,
            "training_centers": centers,
            "feature_names": names,
            "feature_mean": mean,
            "feature_scale": scale,
            "intercept": intercept,
            "coefficients": coefficients,
            "ridge_alpha": alpha,
            "training_row_count": int(self.training_row_count),
            "training_response_hash": self.training_response_hash,
            "weight_audit_hash": self.weight_audit_hash,
            "solver": self.solver,
            "estimator_persisted": False,
        }
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "excluded_outer_center", outer)
        object.__setattr__(self, "excluded_scored_center", scored)
        object.__setattr__(self, "training_centers", centers)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "ridge_alpha", alpha)
        object.__setattr__(self, "training_row_count", int(self.training_row_count))
        object.__setattr__(self, "model_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_calibration_model_v1",
            "metric": self.metric,
            "excluded_outer_center": self.excluded_outer_center,
            "excluded_scored_center": self.excluded_scored_center,
            "training_centers": list(self.training_centers),
            "all_excluded_centers": list(self.all_excluded_centers),
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "ridge_alpha": self.ridge_alpha,
            "training_row_count": self.training_row_count,
            "training_response_hash": self.training_response_hash,
            "weight_audit_hash": self.weight_audit_hash,
            "solver": self.solver,
            "estimator_persisted": False,
            "model_hash": self.model_hash,
        }

    @property
    def all_excluded_centers(self) -> tuple[str, ...]:
        """Expose every exclusion, including nested reliability exclusions."""

        return tuple(center for center in CENTERS if center not in self.training_centers)


@dataclass(frozen=True)
class ActionStratumReliability:
    """Fully OOF reliability decision for one family/direction stratum."""

    excluded_outer_center: str
    excluded_scored_center: str | None
    family: str
    direction: str
    represented_centers: tuple[str, ...]
    center_metric_means: tuple[tuple[str, float, float, float], ...]
    equal_center_utility: FavorableUtility
    bacc_spearman: float | None
    bacc_spearman_defined: bool
    positive_bacc_center_count: int
    minimum_center_count: int
    bank_viable: bool
    oof_row_count: int
    evidence_hash: str
    reason_codes: tuple[str, ...] = field(init=False)
    passed: bool = field(init=False)
    reliability_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.excluded_outer_center)
        scored = _optional_center(self.excluded_scored_center, role="reliability scored exclusion")
        family = str(self.family)
        direction = str(self.direction)
        centers = tuple(str(center) for center in self.represented_centers)
        means = tuple(
            (str(center), float(bacc), float(brier), float(log_gain))
            for center, bacc, brier, log_gain in self.center_metric_means
        )
        minimum = int(self.minimum_center_count)
        positive = int(self.positive_bacc_center_count)
        rho = None if self.bacc_spearman is None else float(self.bacc_spearman)
        if (
            outer not in CENTERS
            or scored == outer
            or family not in ACTION_FAMILIES
            or direction not in DIRECTIONS
            or centers != tuple(center for center in CENTERS if center in set(centers))
            or len(centers) != len(set(centers))
            or outer in centers
            or (scored is not None and scored in centers)
            or tuple(row[0] for row in means) != centers
            or not all(math.isfinite(value) for row in means for value in row[1:])
            or minimum <= 0
            or positive != sum(1 for _center, value, _brier, _log in means if value > 0.0)
            or positive < 0
            or positive > len(centers)
            or int(self.oof_row_count) < 0
            or bool(self.bacc_spearman_defined) != (rho is not None)
            or (rho is not None and (not math.isfinite(rho) or rho < -1.0 or rho > 1.0))
        ):
            raise ProtocolError("P-DCAPS action-stratum reliability evidence drifted.")
        require_sha256(self.evidence_hash, "reliability evidence hash")
        reasons: list[str] = []
        if len(centers) < minimum:
            reasons.append("INSUFFICIENT_REPRESENTED_CENTERS")
        if rho is None:
            reasons.append("BACC_SPEARMAN_UNDEFINED")
        elif rho <= 0.0:
            reasons.append("BACC_SPEARMAN_NOT_POSITIVE")
        if self.equal_center_utility.bacc_gain <= 0.0:
            reasons.append("EQUAL_CENTER_BACC_NOT_POSITIVE")
        if positive <= len(centers) / 2.0:
            reasons.append("BACC_CENTER_MAJORITY_NOT_POSITIVE")
        if self.equal_center_utility.brier_gain < 0.0:
            reasons.append("EQUAL_CENTER_BRIER_NEGATIVE")
        if self.equal_center_utility.log_gain < 0.0:
            reasons.append("EQUAL_CENTER_LOG_NEGATIVE")
        if not bool(self.bank_viable):
            reasons.append("BANK_NOT_VIABLE")
        payload = {
            "schema_version": "pdcaps_action_stratum_reliability_v1",
            "excluded_outer_center": outer,
            "excluded_scored_center": scored,
            "family": family,
            "direction": direction,
            "represented_centers": centers,
            "center_metric_means": means,
            "equal_center_utility": self.equal_center_utility.to_payload(),
            "bacc_spearman": rho,
            "bacc_spearman_defined": bool(self.bacc_spearman_defined),
            "positive_bacc_center_count": positive,
            "minimum_center_count": minimum,
            "bank_viable": bool(self.bank_viable),
            "oof_row_count": int(self.oof_row_count),
            "evidence_hash": self.evidence_hash,
            "reason_codes": tuple(reasons),
            "passed": not reasons,
            "raw_labels_persisted": False,
        }
        object.__setattr__(self, "excluded_outer_center", outer)
        object.__setattr__(self, "excluded_scored_center", scored)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "represented_centers", centers)
        object.__setattr__(self, "center_metric_means", means)
        object.__setattr__(self, "bacc_spearman", rho)
        object.__setattr__(self, "positive_bacc_center_count", positive)
        object.__setattr__(self, "minimum_center_count", minimum)
        object.__setattr__(self, "bank_viable", bool(self.bank_viable))
        object.__setattr__(self, "oof_row_count", int(self.oof_row_count))
        object.__setattr__(self, "reason_codes", tuple(reasons))
        object.__setattr__(self, "passed", not reasons)
        object.__setattr__(self, "reliability_hash", canonical_hash(payload))

    @property
    def stratum(self) -> tuple[str, str]:
        return self.family, self.direction

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_action_stratum_reliability_v1",
            "excluded_outer_center": self.excluded_outer_center,
            "excluded_scored_center": self.excluded_scored_center,
            "family": self.family,
            "direction": self.direction,
            "represented_centers": list(self.represented_centers),
            "center_metric_means": [list(row) for row in self.center_metric_means],
            "equal_center_utility": self.equal_center_utility.to_payload(),
            "bacc_spearman": self.bacc_spearman,
            "bacc_spearman_defined": self.bacc_spearman_defined,
            "positive_bacc_center_count": self.positive_bacc_center_count,
            "minimum_center_count": self.minimum_center_count,
            "bank_viable": self.bank_viable,
            "oof_row_count": self.oof_row_count,
            "evidence_hash": self.evidence_hash,
            "reason_codes": list(self.reason_codes),
            "passed": self.passed,
            "raw_labels_persisted": False,
            "reliability_hash": self.reliability_hash,
        }


@dataclass(frozen=True)
class CalibratedAction:
    prediction: ActionPrediction
    calibrated_utility: FavorableUtility
    model_hashes: tuple[tuple[str, str], ...]
    model_excluded_outer_center: str
    model_excluded_scored_center: str | None
    reliability: ActionStratumReliability
    quarantine_reasons: tuple[str, ...] = field(init=False)
    quarantined: bool = field(init=False)
    eligible: bool = field(init=False)
    calibrated_action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = tuple((str(metric), str(digest)) for metric, digest in self.model_hashes)
        outer = str(self.model_excluded_outer_center)
        scored = _optional_center(self.model_excluded_scored_center, role="calibrated scored exclusion")
        route = self.prediction.key.route_key
        expected_scored = route.excluded_scored_center
        if (
            tuple(metric for metric, _ in hashes) != METRICS
            or outer != route.excluded_outer_center
            or scored != expected_scored
            or self.reliability.excluded_outer_center != outer
            or self.reliability.excluded_scored_center != scored
            or self.reliability.stratum != self.prediction.key.stratum
        ):
            raise ProtocolError("P-DCAPS calibrated action exclusion lineage drifted.")
        for metric, digest in hashes:
            require_sha256(digest, f"{metric} action-model hash")
        reasons = list(self.reliability.reason_codes)
        viability = self.prediction.bank_viability
        if not viability.row_preserving:
            reasons.append("BANK_NOT_ROW_PRESERVING")
        if not viability.passed:
            reasons.append("ACTION_BANK_NOT_VIABLE")
        quarantined = bool(reasons)
        utility = self.calibrated_utility
        eligible = (
            not quarantined
            and utility.bacc_gain > TIE_TOLERANCE
            and utility.brier_gain >= -TIE_TOLERANCE
            and utility.log_gain >= -TIE_TOLERANCE
        )
        payload = {
            "schema_version": "pdcaps_calibrated_action_v1",
            "prediction_hash": self.prediction.prediction_hash,
            "calibrated_utility": utility.to_payload(),
            "model_hashes": hashes,
            "model_excluded_outer_center": outer,
            "model_excluded_scored_center": scored,
            "reliability_hash": self.reliability.reliability_hash,
            "quarantine_reasons": tuple(reasons),
            "quarantined": quarantined,
            "eligible": eligible,
        }
        object.__setattr__(self, "model_hashes", hashes)
        object.__setattr__(self, "model_excluded_outer_center", outer)
        object.__setattr__(self, "model_excluded_scored_center", scored)
        object.__setattr__(self, "quarantine_reasons", tuple(reasons))
        object.__setattr__(self, "quarantined", quarantined)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "calibrated_action_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_calibrated_action_v1",
            "prediction": self.prediction.to_payload(),
            "calibrated_utility": self.calibrated_utility.to_payload(),
            "model_hashes": [list(row) for row in self.model_hashes],
            "model_excluded_outer_center": self.model_excluded_outer_center,
            "model_excluded_scored_center": self.model_excluded_scored_center,
            "reliability": self.reliability.to_payload(),
            "quarantine_reasons": list(self.quarantine_reasons),
            "quarantined": self.quarantined,
            "eligible": self.eligible,
            "calibrated_action_hash": self.calibrated_action_hash,
        }


@dataclass(frozen=True)
class CalibratedActionSelection:
    route_key: RouteKey
    calibrated_action_hashes: tuple[str, ...]
    quarantined_action_hashes: tuple[str, ...]
    selected_action_key: ActionKey | None
    selected_utility: FavorableUtility
    exact_p_fallback: bool
    reason: str
    selection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action_hashes = tuple(str(value) for value in self.calibrated_action_hashes)
        quarantined = tuple(str(value) for value in self.quarantined_action_hashes)
        for digest in (*action_hashes, *quarantined):
            require_sha256(digest, "calibrated action hash")
        if (
            len(action_hashes) != len(set(action_hashes))
            or not set(quarantined).issubset(set(action_hashes))
            or bool(self.exact_p_fallback) != (self.selected_action_key is None)
            or (self.selected_action_key is not None and self.selected_action_key.route_key != self.route_key)
            or (self.exact_p_fallback and self.selected_utility != FavorableUtility.zeros())
            or not str(self.reason)
            or (
                not action_hashes
                and (
                    not self.exact_p_fallback
                    or quarantined
                    or self.reason != "EXACT_P_NO_CROSSING_ACTION"
                )
            )
        ):
            raise ProtocolError("P-DCAPS calibrated action selection drifted.")
        payload = {
            "schema_version": "pdcaps_calibrated_action_selection_v1",
            "route_key": self.route_key.to_payload(),
            "calibrated_action_hashes": action_hashes,
            "quarantined_action_hashes": quarantined,
            "selected_action_key_hash": (
                None if self.selected_action_key is None else self.selected_action_key.action_key_hash
            ),
            "selected_utility": self.selected_utility.to_payload(),
            "exact_p_fallback": bool(self.exact_p_fallback),
            "reason": str(self.reason),
        }
        object.__setattr__(self, "calibrated_action_hashes", action_hashes)
        object.__setattr__(self, "quarantined_action_hashes", quarantined)
        object.__setattr__(self, "exact_p_fallback", bool(self.exact_p_fallback))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "selection_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_calibrated_action_selection_v1",
            "route_key": self.route_key.to_payload(),
            "calibrated_action_hashes": list(self.calibrated_action_hashes),
            "quarantined_action_hashes": list(self.quarantined_action_hashes),
            "selected_action_key": (
                None if self.selected_action_key is None else self.selected_action_key.to_payload()
            ),
            "selected_utility": self.selected_utility.to_payload(),
            "exact_p_fallback": self.exact_p_fallback,
            "reason": self.reason,
            "selection_hash": self.selection_hash,
        }


__all__ = (
    "ActionCalibrationModel",
    "ActionKey",
    "ActionPrediction",
    "ActionResponse",
    "ActionStratumReliability",
    "CalibratedAction",
    "CalibratedActionSelection",
)
