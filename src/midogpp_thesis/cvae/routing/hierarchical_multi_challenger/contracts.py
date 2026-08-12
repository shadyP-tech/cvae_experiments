"""Immutable contracts for hierarchical multi-challenger case routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_hash, fitted_numeric_fingerprint


DIRECTIONS = ("0to1", "1to0")
MODEL_FAMILIES = ("G", "R", "P")


def _text(value: object, role: str) -> str:
    result = str(value)
    if not result:
        raise ProtocolError(f"{role} must be non-empty.")
    return result


def _finite(value: object, role: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{role} must be finite.")
    return result


def _sha256(value: object, role: str) -> str:
    result = str(value)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ProtocolError(f"{role} must be a lowercase SHA-256 digest.")
    return result


@dataclass(frozen=True, order=True)
class DirectionalDonorRow:
    """One aggregated strict-H/q/e binomial direction response."""

    model_target: str
    query_center: str
    candidate_source: str
    case_id: str
    action_id: str
    feature_case_id: str
    direction: str
    success_count: int
    trial_count: int
    feature_names: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        for role in (
            "model_target",
            "query_center",
            "candidate_source",
            "case_id",
            "action_id",
            "feature_case_id",
        ):
            _text(getattr(self, role), role)
        if self.direction not in DIRECTIONS:
            raise ProtocolError("Directional donor direction drifted.")
        if self.query_center == self.model_target or self.candidate_source in {
            self.model_target,
            self.query_center,
        }:
            raise ProtocolError("Directional donor violates strict H/q/e exclusion.")
        if self.trial_count < 0 or not 0 <= self.success_count <= self.trial_count:
            raise ProtocolError("Directional donor binomial counts are invalid.")
        names = tuple(_text(value, "feature_name") for value in self.feature_names)
        values = tuple(_finite(value, "feature_value") for value in self.values)
        if not names or len(names) != len(values) or len(set(names)) != len(names):
            raise ProtocolError("Directional donor feature schema drifted.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)

    @property
    def case_cluster(self) -> str:
        return f"{self.query_center}::{self.case_id}"

    def to_payload(self) -> dict[str, object]:
        return {
            "model_target": self.model_target,
            "query_center": self.query_center,
            "candidate_source": self.candidate_source,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "feature_case_id": self.feature_case_id,
            "direction": self.direction,
            "success_count": self.success_count,
            "trial_count": self.trial_count,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class DirectionalLogitModel:
    """One penalized pooled direction model with Laplace covariance."""

    model_target: str
    family: str
    direction: str
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    candidate_sources: tuple[str, ...]
    query_centers: tuple[str, ...]
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    feature_alpha: float
    source_alpha: float
    query_alpha: float
    intercept_alpha: float
    training_row_count: int
    training_trial_count: int
    training_case_clusters: tuple[str, ...]
    provenance_hash: str
    fit_fingerprint: str = ""

    def __post_init__(self) -> None:
        _text(self.model_target, "model_target")
        if self.family not in MODEL_FAMILIES or self.direction not in DIRECTIONS:
            raise ProtocolError("Directional model identity drifted.")
        names = tuple(_text(value, "feature_name") for value in self.feature_names)
        means = tuple(_finite(value, "feature_mean") for value in self.feature_mean)
        scales = tuple(_finite(value, "feature_scale") for value in self.feature_scale)
        sources = tuple(
            _text(value, "candidate_source") for value in self.candidate_sources
        )
        queries = tuple(_text(value, "query_center") for value in self.query_centers)
        coefficients = tuple(_finite(value, "coefficient") for value in self.coefficients)
        covariance = tuple(
            tuple(_finite(value, "covariance") for value in row)
            for row in self.covariance
        )
        expected_features = 0 if self.family == "G" else len(names)
        dimension = 1 + expected_features + len(sources) + len(queries)
        if (
            not names
            or len(names) != len(means)
            or len(names) != len(scales)
            or any(value <= 0.0 for value in scales)
            or not sources
            or tuple(sorted(set(sources))) != sources
            or self.model_target in sources
            or tuple(sorted(set(queries))) != queries
            or len(coefficients) != dimension
            or len(covariance) != dimension
            or any(len(row) != dimension for row in covariance)
            or self.training_row_count <= 0
            or self.training_trial_count < self.training_row_count
            or len(set(self.training_case_clusters)) < 2
        ):
            raise ProtocolError("Directional model geometry drifted.")
        for value, role in (
            (self.feature_alpha, "feature_alpha"),
            (self.source_alpha, "source_alpha"),
            (self.query_alpha, "query_alpha"),
            (self.intercept_alpha, "intercept_alpha"),
        ):
            if _finite(value, role) <= 0.0:
                raise ProtocolError(f"{role} must be positive.")
        _sha256(self.provenance_hash, "provenance_hash")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", means)
        object.__setattr__(self, "feature_scale", scales)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "query_centers", queries)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "covariance", covariance)
        expected = fitted_numeric_fingerprint(self._fit_payload())
        if self.fit_fingerprint and self.fit_fingerprint != expected:
            raise ProtocolError("Directional model fitted-numeric fingerprint drifted.")
        object.__setattr__(self, "fit_fingerprint", expected)

    @property
    def dimension(self) -> int:
        return len(self.coefficients)

    @property
    def source_effects(self) -> Mapping[str, float]:
        """Return the fitted ridge effect for each trained candidate source."""

        offset = 1 + (0 if self.family == "G" else len(self.feature_names))
        return {
            source: self.coefficients[offset + ordinal]
            for ordinal, source in enumerate(self.candidate_sources)
        }

    def _fit_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_directional_logit_fit_v2",
            "model_target": self.model_target,
            "family": self.family,
            "direction": self.direction,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "candidate_sources": list(self.candidate_sources),
            "query_centers": list(self.query_centers),
            "coefficients": list(self.coefficients),
            "covariance": [list(row) for row in self.covariance],
            "feature_alpha": self.feature_alpha,
            "source_alpha": self.source_alpha,
            "query_alpha": self.query_alpha,
            "intercept_alpha": self.intercept_alpha,
            "training_row_count": self.training_row_count,
            "training_trial_count": self.training_trial_count,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._fit_payload(),
            "training_case_clusters": list(self.training_case_clusters),
            "provenance_hash": self.provenance_hash,
            "fit_fingerprint": self.fit_fingerprint,
        }


@dataclass(frozen=True)
class DirectionalPrediction:
    probability: float
    design: tuple[float, ...]
    parameter_variance: float
    model_fingerprint: str

    def __post_init__(self) -> None:
        probability = _finite(self.probability, "probability")
        design = tuple(_finite(value, "design") for value in self.design)
        variance = _finite(self.parameter_variance, "parameter_variance")
        if not 0.0 < probability < 1.0 or not design or variance < 0.0:
            raise ProtocolError("Directional prediction drifted.")
        _sha256(self.model_fingerprint, "model_fingerprint")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "parameter_variance", variance)


@dataclass(frozen=True, order=True)
class SupportActionScore:
    action_id: str
    exact_gain: float
    shrunken_gain: float
    support_case_count: int

    def __post_init__(self) -> None:
        _text(self.action_id, "action_id")
        object.__setattr__(self, "exact_gain", _finite(self.exact_gain, "exact_gain"))
        object.__setattr__(
            self, "shrunken_gain", _finite(self.shrunken_gain, "shrunken_gain")
        )
        if self.support_case_count <= 0:
            raise ProtocolError("Support action score lacks cases.")

    def to_payload(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "exact_gain": self.exact_gain,
            "shrunken_gain": self.shrunken_gain,
            "support_case_count": self.support_case_count,
        }


@dataclass(frozen=True)
class CandidateMenu:
    action_ids: tuple[str, ...]
    anchor_action_id: str
    ranked_support_actions: tuple[SupportActionScore, ...]
    top_k: int
    menu_hash: str = ""

    def __post_init__(self) -> None:
        actions = tuple(_text(value, "action_id") for value in self.action_ids)
        if (
            self.top_k != 3
            or not actions
            or actions[0] != "B"
            or len(set(actions)) != len(actions)
            or self.anchor_action_id not in actions
            or len(actions) > self.top_k + 1
            or len(self.ranked_support_actions) < self.top_k
        ):
            raise ProtocolError("Candidate menu topology drifted.")
        unhashed = {
            "schema_version": "hierarchical_multi_challenger_menu_v1",
            "action_ids": list(actions),
            "anchor_action_id": self.anchor_action_id,
            "ranked_support_actions": [
                row.to_payload() for row in self.ranked_support_actions
            ],
            "top_k": self.top_k,
        }
        expected = canonical_hash(unhashed)
        if self.menu_hash and self.menu_hash != expected:
            raise ProtocolError("Candidate menu hash drifted.")
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "menu_hash", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "hierarchical_multi_challenger_menu_v1",
            "action_ids": list(self.action_ids),
            "anchor_action_id": self.anchor_action_id,
            "ranked_support_actions": [
                row.to_payload() for row in self.ranked_support_actions
            ],
            "top_k": self.top_k,
            "menu_hash": self.menu_hash,
        }


@dataclass(frozen=True)
class DirectionalCalibration:
    direction: str
    offset: float
    offset_variance: float
    success_count: int
    trial_count: int
    row_count: int
    case_count: int
    alpha: float
    menu_hash: str
    valid: bool
    calibration_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ProtocolError("Calibration direction drifted.")
        offset = _finite(self.offset, "offset")
        variance = _finite(self.offset_variance, "offset_variance")
        if (
            variance < 0.0
            or self.success_count < 0
            or self.trial_count < self.success_count
            or self.row_count < 0
            or self.case_count < 0
            or self.case_count > self.row_count
            or _finite(self.alpha, "alpha") <= 0.0
            or (
                not self.valid
                and any(
                    value != 0
                    for value in (
                        self.success_count,
                        self.trial_count,
                        self.row_count,
                        self.case_count,
                    )
                )
            )
        ):
            raise ProtocolError("Directional calibration drifted.")
        _sha256(self.menu_hash, "menu_hash")
        payload = {
            "schema_version": "hierarchical_direction_calibration_v1",
            "direction": self.direction,
            "offset": offset,
            "offset_variance": variance,
            "success_count": self.success_count,
            "trial_count": self.trial_count,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "alpha": self.alpha,
            "menu_hash": self.menu_hash,
            "valid": self.valid,
        }
        expected = fitted_numeric_fingerprint(payload)
        if self.calibration_fingerprint and self.calibration_fingerprint != expected:
            raise ProtocolError("Calibration fitted-numeric fingerprint drifted.")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "offset_variance", variance)
        object.__setattr__(self, "calibration_fingerprint", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "offset": self.offset,
            "offset_variance": self.offset_variance,
            "success_count": self.success_count,
            "trial_count": self.trial_count,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "alpha": self.alpha,
            "menu_hash": self.menu_hash,
            "valid": self.valid,
            "calibration_fingerprint": self.calibration_fingerprint,
        }


@dataclass(frozen=True)
class ActionScore:
    action_id: str
    expected_gain: float
    epistemic_variance: float
    calibration_variance: float
    model_gradients: Mapping[str, tuple[float, ...]]
    calibration_gradients: Mapping[str, float]

    def __post_init__(self) -> None:
        _text(self.action_id, "action_id")
        object.__setattr__(
            self, "expected_gain", _finite(self.expected_gain, "expected_gain")
        )
        for role in ("epistemic_variance", "calibration_variance"):
            value = _finite(getattr(self, role), role)
            if value < 0.0:
                raise ProtocolError(f"{role} cannot be negative.")
            object.__setattr__(self, role, value)
        gradients = {
            str(direction): tuple(_finite(value, "model_gradient") for value in row)
            for direction, row in self.model_gradients.items()
        }
        calibration = {
            str(direction): _finite(value, "calibration_gradient")
            for direction, value in self.calibration_gradients.items()
        }
        if set(gradients) != set(DIRECTIONS) or set(calibration) != set(DIRECTIONS):
            raise ProtocolError("Action-score uncertainty components drifted.")
        object.__setattr__(self, "model_gradients", gradients)
        object.__setattr__(self, "calibration_gradients", calibration)

    def to_payload(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "expected_gain": self.expected_gain,
            "epistemic_standard_error": math.sqrt(self.epistemic_variance),
            "calibration_standard_error": math.sqrt(self.calibration_variance),
        }


@dataclass(frozen=True)
class MultiChallengerDecision:
    case_id: str
    method_id: str
    anchor_action_id: str
    selected_action_id: str
    best_action_id: str
    runner_up_action_id: str
    predicted_gain: float
    action_margin: float
    epistemic_standard_error: float
    calibration_standard_error: float
    margin_standard_error: float
    margin_lcb: float
    reason: str
    menu_hash: str

    def __post_init__(self) -> None:
        for role in (
            "case_id",
            "method_id",
            "anchor_action_id",
            "selected_action_id",
            "best_action_id",
            "runner_up_action_id",
            "reason",
        ):
            _text(getattr(self, role), role)
        for role in (
            "predicted_gain",
            "action_margin",
            "epistemic_standard_error",
            "calibration_standard_error",
            "margin_standard_error",
            "margin_lcb",
        ):
            value = _finite(getattr(self, role), role)
            if "standard_error" in role and value < 0.0:
                raise ProtocolError(f"{role} cannot be negative.")
            object.__setattr__(self, role, value)
        _sha256(self.menu_hash, "menu_hash")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


__all__ = (
    "ActionScore",
    "CandidateMenu",
    "DIRECTIONS",
    "DirectionalCalibration",
    "DirectionalDonorRow",
    "DirectionalLogitModel",
    "DirectionalPrediction",
    "MODEL_FAMILIES",
    "MultiChallengerDecision",
    "SupportActionScore",
)
