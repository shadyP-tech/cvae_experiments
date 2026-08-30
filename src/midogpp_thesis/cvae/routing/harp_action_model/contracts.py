"""Typed contracts for HARP's source-only action-response model.

The contracts deliberately distinguish source-inner training observations from
label-free target actions.  Target truth is therefore not representable at the
model or policy boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import struct

from ...protocol import ProtocolError


LAMBDA_GRID = (0.25, 0.5, 0.75, 1.0)
DIRECTIONS = ("D01", "D10", "ALL_MARGINS")
OUTCOMES = ("gain", "brier", "log_loss")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a canonical nonempty string.")
    return value


def _hash(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if _SHA256.fullmatch(text) is None:
        raise ProtocolError(f"{name} must be a lowercase SHA-256 identity.")
    return text


def _lambda(value: object) -> float:
    number = float(value)
    if number not in LAMBDA_GRID:
        raise ProtocolError("HARP action lambda is outside the frozen grid.")
    return number


def _features(names: tuple[str, ...], values: tuple[float, ...]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    normalized_names = tuple(_text(value, name="feature name") for value in names)
    normalized_values = tuple(float(value) for value in values)
    if (
        not normalized_names
        or len(normalized_names) != len(normalized_values)
        or len(set(normalized_names)) != len(normalized_names)
        or any(not math.isfinite(value) for value in normalized_values)
    ):
        raise ProtocolError("HARP features must be finite, unique, and aligned.")
    return normalized_names, normalized_values


@dataclass(frozen=True, kw_only=True)
class HarpTrainingObservation:
    """One authorized source-inner response row after outer-H exclusion."""

    outer_target_id: str
    pseudo_query_id: str
    candidate_source_id: str
    case_id: str
    sample_id: str
    lambda_value: float
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    weighted_correctness_surrogate: float
    brier_delta: float
    log_loss_delta: float
    truth_class: int
    ensemble_size: int
    ensemble_receipt_hash: str
    case_aggregation_receipt_hash: str
    prediction_seal_hash: str
    response_receipt_hash: str

    def __post_init__(self) -> None:
        for name in ("outer_target_id", "pseudo_query_id", "candidate_source_id", "case_id", "sample_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.outer_target_id in (self.pseudo_query_id, self.candidate_source_id):
            raise ProtocolError("Outer H must be excluded from both query and candidate roles.")
        if self.pseudo_query_id == self.candidate_source_id:
            raise ProtocolError("A pseudo-query cannot use its own expert as a candidate.")
        if type(self.ensemble_size) is not int or self.ensemble_size != 9:
            raise ProtocolError("HARP training rows require one exact-nine ensemble.")
        object.__setattr__(self, "lambda_value", _lambda(self.lambda_value))
        if self.direction not in DIRECTIONS:
            raise ProtocolError("HARP direction must be D01, D10, or ALL_MARGINS.")
        names, values = _features(self.feature_names, self.feature_values)
        if "seed_dispersion" not in names:
            raise ProtocolError("HARP features must retain exact-nine seed dispersion.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        for name in ("weighted_correctness_surrogate", "brier_delta", "log_loss_delta"):
            if not math.isfinite(float(getattr(self, name))):
                raise ProtocolError(f"{name} must be finite.")
            object.__setattr__(self, name, float(getattr(self, name)))
        if type(self.truth_class) is not int or self.truth_class not in (0, 1):
            raise ProtocolError("Source-inner truth_class must be binary.")
        object.__setattr__(self, "ensemble_receipt_hash", _hash(self.ensemble_receipt_hash, name="ensemble_receipt_hash"))
        object.__setattr__(self, "case_aggregation_receipt_hash", _hash(self.case_aggregation_receipt_hash, name="case_aggregation_receipt_hash"))
        object.__setattr__(self, "prediction_seal_hash", _hash(self.prediction_seal_hash, name="prediction_seal_hash"))
        object.__setattr__(self, "response_receipt_hash", _hash(self.response_receipt_hash, name="response_receipt_hash"))

    @property
    def row_key(self) -> tuple[str, str, str, str, float]:
        return (self.pseudo_query_id, self.candidate_source_id, self.case_id, self.sample_id, self.lambda_value)


@dataclass(frozen=True, kw_only=True)
class HarpTargetAction:
    """A label-free target action.  No target outcome field exists."""

    outer_target_id: str
    target_query_id: str
    candidate_source_id: str
    case_id: str
    sample_id: str
    lambda_value: float
    direction: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    baseline_probability_bytes: bytes
    expert_probability: float
    ensemble_size: int
    ensemble_receipt_hash: str
    prediction_seal_hash: str
    compatibility_shrinkage: float = 1.0
    operational_fallback_probability_bytes: bytes | None = None

    def __post_init__(self) -> None:
        for name in ("outer_target_id", "target_query_id", "candidate_source_id", "case_id", "sample_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.target_query_id != self.outer_target_id:
            raise ProtocolError("A target action must be scored only for its outer H.")
        if self.candidate_source_id == self.outer_target_id:
            raise ProtocolError("The held-out target expert cannot be a HARP action.")
        if type(self.ensemble_size) is not int or self.ensemble_size != 9:
            raise ProtocolError("HARP target actions require one exact-nine ensemble.")
        object.__setattr__(self, "lambda_value", _lambda(self.lambda_value))
        if self.direction not in DIRECTIONS:
            raise ProtocolError("HARP direction must be D01, D10, or ALL_MARGINS.")
        names, values = _features(self.feature_names, self.feature_values)
        if "seed_dispersion" not in names:
            raise ProtocolError("HARP target features must retain exact-nine seed dispersion.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        raw = self.baseline_probability_bytes
        if type(raw) is not bytes or len(raw) != 8:
            raise ProtocolError(
                "Predictive-reference probability must retain exactly eight float64 bytes."
            )
        fallback_raw = self.operational_fallback_probability_bytes
        if fallback_raw is None:
            # Backward-compatible typed construction for unit/library callers;
            # Stage-70 supplies distinct exact-B bytes explicitly.
            fallback_raw = raw
        if type(fallback_raw) is not bytes or len(fallback_raw) != 8:
            raise ProtocolError(
                "Operational fallback probability must retain exactly eight float64 bytes."
            )
        baseline = struct.unpack("<d", raw)[0]
        fallback = struct.unpack("<d", fallback_raw)[0]
        expert = float(self.expert_probability)
        if (
            not math.isfinite(baseline)
            or not 0.0 <= baseline <= 1.0
            or not math.isfinite(fallback)
            or not 0.0 <= fallback <= 1.0
            or not math.isfinite(expert)
            or not 0.0 <= expert <= 1.0
        ):
            raise ProtocolError("HARP probabilities must lie in [0, 1].")
        object.__setattr__(self, "operational_fallback_probability_bytes", fallback_raw)
        object.__setattr__(self, "expert_probability", expert)
        rho = float(self.compatibility_shrinkage)
        if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ProtocolError("Compatibility may only shrink or abstain in [0, 1].")
        object.__setattr__(self, "compatibility_shrinkage", rho)
        object.__setattr__(self, "ensemble_receipt_hash", _hash(self.ensemble_receipt_hash, name="ensemble_receipt_hash"))
        object.__setattr__(self, "prediction_seal_hash", _hash(self.prediction_seal_hash, name="prediction_seal_hash"))

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.outer_target_id, self.case_id, self.sample_id)

    @property
    def action_key(self) -> tuple[str, str, str, str, float]:
        return (*self.group_key, self.candidate_source_id, self.lambda_value)

    @property
    def baseline_probability(self) -> float:
        return float(struct.unpack("<d", self.baseline_probability_bytes)[0])

    @property
    def operational_fallback_probability(self) -> float:
        raw = self.operational_fallback_probability_bytes
        assert isinstance(raw, bytes)
        return float(struct.unpack("<d", raw)[0])


@dataclass(frozen=True)
class HarpSupportCell:
    candidate_source_id: str
    lambda_value: float
    direction: str
    donor_count: int
    paired_case_count: int
    truth_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.candidate_source_id) is not str
            or not self.candidate_source_id
            or float(self.lambda_value) not in LAMBDA_GRID
            or self.direction not in DIRECTIONS
            or type(self.donor_count) is not int
            or self.donor_count < 0
            or type(self.paired_case_count) is not int
            or self.paired_case_count < 0
            or self.truth_classes not in ((), (0,), (1,), (0, 1))
        ):
            raise ProtocolError("HARP support cell is malformed.")


@dataclass(frozen=True)
class HarpActionScore:
    action: HarpTargetAction
    gain_predictions: tuple[float, ...]
    brier_predictions: tuple[float, ...]
    log_loss_predictions: tuple[float, ...]
    leverages: tuple[float, ...]
    support: HarpSupportCell
    delete_donors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.action, HarpTargetAction) or not isinstance(self.support, HarpSupportCell):
            raise ProtocolError("HARP scores require typed action and support contracts.")
        lengths = {len(self.gain_predictions), len(self.brier_predictions), len(self.log_loss_predictions), len(self.leverages), len(self.delete_donors)}
        if lengths != {next(iter(lengths))} or next(iter(lengths)) <= 0:
            raise ProtocolError("Delete-donor predictions must be nonempty and aligned.")
        values = (*self.gain_predictions, *self.brier_predictions, *self.log_loss_predictions, *self.leverages)
        if any(not math.isfinite(float(value)) for value in values) or any(float(value) < 0 for value in self.leverages):
            raise ProtocolError("HARP score predictions and leverages must be finite.")
        if tuple(sorted(set(self.delete_donors))) != self.delete_donors:
            raise ProtocolError("Delete donors must be canonical and unique.")
        if (
            self.support.candidate_source_id != self.action.candidate_source_id
            or self.support.lambda_value != self.action.lambda_value
            or self.support.direction not in (self.action.direction, "ALL_MARGINS")
        ):
            raise ProtocolError("HARP action score escaped its support cell.")


__all__ = (
    "DIRECTIONS", "LAMBDA_GRID", "OUTCOMES", "HarpActionScore",
    "HarpSupportCell", "HarpTargetAction", "HarpTrainingObservation",
)
