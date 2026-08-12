"""Immutable, serializable contracts for threshold-flip case routing.

The contracts deliberately have no field capable of carrying held-evaluation
labels.  Evaluation labels enter only the terminal metric functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError


SCHEMA_VERSION = "threshold_flip_case_router_core_v1"


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, name: str) -> str:
    result = str(value)
    if not result:
        raise ProtocolError(f"{name} must be non-empty.")
    return result


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite.")
    return result


def _sha256(value: object, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ProtocolError(f"{name} must be a lowercase SHA-256 digest.")
    return result


@dataclass(frozen=True, order=True)
class ContributionTarget:
    """Additive confusion-count change from B to one physical action."""

    case_id: str
    action_id: str
    delta_tp: int
    delta_tn: int
    n_positive: int
    n_negative: int

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.action_id, "action_id")
        if self.n_positive < 0 or self.n_negative < 0:
            raise ProtocolError("Class counts cannot be negative.")
        if abs(self.delta_tp) > self.n_positive or abs(self.delta_tn) > self.n_negative:
            raise ProtocolError("Contribution exceeds its available class count.")

    def to_payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "action_id": self.action_id,
            "delta_tp": self.delta_tp,
            "delta_tn": self.delta_tn,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ContributionTarget":
        return cls(
            case_id=str(payload["case_id"]),
            action_id=str(payload["action_id"]),
            delta_tp=int(payload["delta_tp"]),
            delta_tn=int(payload["delta_tn"]),
            n_positive=int(payload["n_positive"]),
            n_negative=int(payload["n_negative"]),
        )


@dataclass(frozen=True, order=True)
class CaseActionFeatures:
    """One label-free feature vector for a B-versus-action case comparison."""

    target_center: str
    case_id: str
    action_id: str
    candidate_source: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    flip_0to1_count: int
    flip_1to0_count: int
    feature_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("target_center", "case_id", "action_id", "candidate_source"):
            _text(getattr(self, name), name)
        names = tuple(_text(value, "feature_name") for value in self.feature_names)
        values = tuple(_finite(value, "feature_value") for value in self.values)
        if not names or len(names) != len(values) or len(set(names)) != len(names):
            raise ProtocolError("Feature names and values are not a unique aligned vector.")
        if self.flip_0to1_count < 0 or self.flip_1to0_count < 0:
            raise ProtocolError("Flip counts cannot be negative.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)
        expected = canonical_hash(self._unhashed_payload())
        if self.feature_hash:
            if _sha256(self.feature_hash, "feature_hash") != expected:
                raise ProtocolError("Case-action feature hash drifted.")
        else:
            object.__setattr__(self, "feature_hash", expected)

    @property
    def has_flips(self) -> bool:
        return self.flip_0to1_count + self.flip_1to0_count > 0

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "target_center": self.target_center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "candidate_source": self.candidate_source,
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "flip_0to1_count": self.flip_0to1_count,
            "flip_1to0_count": self.flip_1to0_count,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "feature_hash": self.feature_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CaseActionFeatures":
        return cls(
            target_center=str(payload["target_center"]),
            case_id=str(payload["case_id"]),
            action_id=str(payload["action_id"]),
            candidate_source=str(payload["candidate_source"]),
            feature_names=tuple(str(v) for v in payload["feature_names"]),  # type: ignore[index]
            values=tuple(float(v) for v in payload["values"]),  # type: ignore[index]
            flip_0to1_count=int(payload["flip_0to1_count"]),
            flip_1to0_count=int(payload["flip_1to0_count"]),
            feature_hash=str(payload["feature_hash"]),
        )


@dataclass(frozen=True, order=True)
class DonorRow:
    """A labeled donor row for a model dedicated to one held-out H.

    Construction enforces q != H and e != H,q.  ``feature_case_id`` may differ
    from ``case_id`` only for the blocked whole-case permutation control.
    """

    model_target: str
    query_center: str
    candidate_source: str
    case_id: str
    action_id: str
    feature_case_id: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    target: ContributionTarget

    def __post_init__(self) -> None:
        fields = ("model_target", "query_center", "candidate_source", "case_id", "action_id", "feature_case_id")
        for name in fields:
            _text(getattr(self, name), name)
        if self.query_center == self.model_target:
            raise ProtocolError("Donor query q must exclude held-out model target H.")
        if self.candidate_source in {self.model_target, self.query_center}:
            raise ProtocolError("Donor candidate e must exclude both H and q.")
        names = tuple(_text(value, "feature_name") for value in self.feature_names)
        values = tuple(_finite(value, "feature_value") for value in self.values)
        if not names or len(names) != len(values) or len(set(names)) != len(names):
            raise ProtocolError("Donor feature vector drifted.")
        if self.target.case_id != self.case_id or self.target.action_id != self.action_id:
            raise ProtocolError("Donor feature/target identity drifted.")
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
            "feature_names": list(self.feature_names),
            "values": list(self.values),
            "target": self.target.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DonorRow":
        target = payload["target"]
        if not isinstance(target, Mapping):
            raise ProtocolError("Donor target payload is not a mapping.")
        return cls(
            model_target=str(payload["model_target"]),
            query_center=str(payload["query_center"]),
            candidate_source=str(payload["candidate_source"]),
            case_id=str(payload["case_id"]),
            action_id=str(payload["action_id"]),
            feature_case_id=str(payload["feature_case_id"]),
            feature_names=tuple(str(v) for v in payload["feature_names"]),  # type: ignore[index]
            values=tuple(float(v) for v in payload["values"]),  # type: ignore[index]
            target=ContributionTarget.from_payload(target),
        )


@dataclass(frozen=True)
class HeadModel:
    intercept: float
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    residual_variance: float

    def __post_init__(self) -> None:
        coefficients = tuple(_finite(v, "coefficient") for v in self.coefficients)
        covariance = tuple(tuple(_finite(v, "covariance") for v in row) for row in self.covariance)
        dimension = len(coefficients) + 1
        if len(covariance) != dimension or any(len(row) != dimension for row in covariance):
            raise ProtocolError("Head covariance dimension drifted.")
        if self.residual_variance < 0.0:
            raise ProtocolError("Residual variance cannot be negative.")
        _finite(self.intercept, "intercept")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "covariance", covariance)

    def to_payload(self) -> dict[str, object]:
        return {
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "covariance": [list(row) for row in self.covariance],
            "residual_variance": self.residual_variance,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "HeadModel":
        return cls(
            intercept=float(payload["intercept"]),
            coefficients=tuple(float(v) for v in payload["coefficients"]),  # type: ignore[index]
            covariance=tuple(tuple(float(v) for v in row) for row in payload["covariance"]),  # type: ignore[index]
            residual_variance=float(payload["residual_variance"]),
        )


@dataclass(frozen=True)
class TwoHeadRidgeModel:
    model_target: str
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    alpha: float
    variance_floor: float
    tp_head: HeadModel
    tn_head: HeadModel
    training_case_clusters: tuple[str, ...]
    donor_query_centers: tuple[str, ...]
    donor_candidate_sources: tuple[str, ...]
    training_row_count: int
    provenance_hash: str
    model_hash: str = ""

    def __post_init__(self) -> None:
        _text(self.model_target, "model_target")
        names = tuple(_text(v, "feature_name") for v in self.feature_names)
        mean = tuple(_finite(v, "feature_mean") for v in self.feature_mean)
        scale = tuple(_finite(v, "feature_scale") for v in self.feature_scale)
        if not names or len(names) != len(mean) or len(names) != len(scale):
            raise ProtocolError("Model feature geometry drifted.")
        if any(v <= 0.0 for v in scale) or self.alpha != 1.0 or self.variance_floor < 1.0e-6:
            raise ProtocolError("Frozen ridge or variance contract drifted.")
        if len(self.tp_head.coefficients) != len(names) or len(self.tn_head.coefficients) != len(names):
            raise ProtocolError("Model head feature dimension drifted.")
        if self.model_target in self.donor_query_centers or self.model_target in self.donor_candidate_sources:
            raise ProtocolError("H-specific model provenance contains H.")
        if self.training_row_count <= 0 or not self.training_case_clusters:
            raise ProtocolError("Model training provenance is empty.")
        _sha256(self.provenance_hash, "provenance_hash")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        expected = canonical_hash(self._unhashed_payload())
        if self.model_hash:
            if _sha256(self.model_hash, "model_hash") != expected:
                raise ProtocolError("Two-head model hash drifted.")
        else:
            object.__setattr__(self, "model_hash", expected)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_target": self.model_target,
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "alpha": self.alpha,
            "variance_floor": self.variance_floor,
            "tp_head": self.tp_head.to_payload(),
            "tn_head": self.tn_head.to_payload(),
            "training_case_clusters": list(self.training_case_clusters),
            "donor_query_centers": list(self.donor_query_centers),
            "donor_candidate_sources": list(self.donor_candidate_sources),
            "training_row_count": self.training_row_count,
            "provenance_hash": self.provenance_hash,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "model_hash": self.model_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwoHeadRidgeModel":
        tp = payload["tp_head"]
        tn = payload["tn_head"]
        if not isinstance(tp, Mapping) or not isinstance(tn, Mapping):
            raise ProtocolError("Model head payload is invalid.")
        return cls(
            model_target=str(payload["model_target"]),
            feature_names=tuple(str(v) for v in payload["feature_names"]),  # type: ignore[index]
            feature_mean=tuple(float(v) for v in payload["feature_mean"]),  # type: ignore[index]
            feature_scale=tuple(float(v) for v in payload["feature_scale"]),  # type: ignore[index]
            alpha=float(payload["alpha"]),
            variance_floor=float(payload["variance_floor"]),
            tp_head=HeadModel.from_payload(tp),
            tn_head=HeadModel.from_payload(tn),
            training_case_clusters=tuple(str(v) for v in payload["training_case_clusters"]),  # type: ignore[index]
            donor_query_centers=tuple(str(v) for v in payload["donor_query_centers"]),  # type: ignore[index]
            donor_candidate_sources=tuple(str(v) for v in payload["donor_candidate_sources"]),  # type: ignore[index]
            training_row_count=int(payload["training_row_count"]),
            provenance_hash=str(payload["provenance_hash"]),
            model_hash=str(payload["model_hash"]),
        )


@dataclass(frozen=True)
class TwoHeadPrediction:
    model_target: str
    case_id: str
    action_id: str
    mean_delta_tp: float
    mean_delta_tn: float
    variance_delta_tp: float
    variance_delta_tn: float
    model_hash: str

    def __post_init__(self) -> None:
        for name in ("model_target", "case_id", "action_id"):
            _text(getattr(self, name), name)
        for name in ("mean_delta_tp", "mean_delta_tn", "variance_delta_tp", "variance_delta_tn"):
            _finite(getattr(self, name), name)
        if self.variance_delta_tp < 0.0 or self.variance_delta_tn < 0.0:
            raise ProtocolError("Prediction variance cannot be negative.")
        _sha256(self.model_hash, "model_hash")

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TwoHeadPrediction":
        return cls(
            model_target=str(payload["model_target"]),
            case_id=str(payload["case_id"]),
            action_id=str(payload["action_id"]),
            mean_delta_tp=float(payload["mean_delta_tp"]),
            mean_delta_tn=float(payload["mean_delta_tn"]),
            variance_delta_tp=float(payload["variance_delta_tp"]),
            variance_delta_tn=float(payload["variance_delta_tn"]),
            model_hash=str(payload["model_hash"]),
        )


@dataclass(frozen=True, order=True)
class CalibrationRow:
    case_id: str
    action_id: str
    raw_gain_0to1: float
    raw_gain_1to0: float
    exact_gain: float

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.action_id, "action_id")
        for name in ("raw_gain_0to1", "raw_gain_1to0", "exact_gain"):
            _finite(getattr(self, name), name)

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CalibrationRow":
        return cls(
            case_id=str(payload["case_id"]),
            action_id=str(payload["action_id"]),
            raw_gain_0to1=float(payload["raw_gain_0to1"]),
            raw_gain_1to0=float(payload["raw_gain_1to0"]),
            exact_gain=float(payload["exact_gain"]),
        )


@dataclass(frozen=True)
class DirectionSharedCalibration:
    gamma_0to1: float
    gamma_1to0: float
    n_positive: int
    n_negative: int
    row_count: int
    valid: bool
    calibration_hash: str = ""

    def __post_init__(self) -> None:
        _finite(self.gamma_0to1, "gamma_0to1")
        _finite(self.gamma_1to0, "gamma_1to0")
        if self.n_positive < 0 or self.n_negative < 0 or self.row_count < 0:
            raise ProtocolError("Calibration counts cannot be negative.")
        if self.valid != (self.n_positive > 0 and self.n_negative > 0 and self.row_count > 0):
            raise ProtocolError("Calibration validity disagrees with its support.")
        expected = canonical_hash(self._unhashed_payload())
        if self.calibration_hash:
            if _sha256(self.calibration_hash, "calibration_hash") != expected:
                raise ProtocolError("Calibration hash drifted.")
        else:
            object.__setattr__(self, "calibration_hash", expected)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "gamma_0to1": self.gamma_0to1,
            "gamma_1to0": self.gamma_1to0,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "row_count": self.row_count,
            "valid": self.valid,
            "zero_intercept": True,
            "target_source_action_intercepts": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "calibration_hash": self.calibration_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DirectionSharedCalibration":
        return cls(
            gamma_0to1=float(payload["gamma_0to1"]),
            gamma_1to0=float(payload["gamma_1to0"]),
            n_positive=int(payload["n_positive"]),
            n_negative=int(payload["n_negative"]),
            row_count=int(payload["row_count"]),
            valid=bool(payload["valid"]),
            calibration_hash=str(payload["calibration_hash"]),
        )


@dataclass(frozen=True)
class StaticSelection:
    action_id: str
    exact_gain: float
    runner_up_gain: float
    fallback_to_b: bool
    selection_hash: str = ""

    def __post_init__(self) -> None:
        _text(self.action_id, "action_id")
        _finite(self.exact_gain, "exact_gain")
        _finite(self.runner_up_gain, "runner_up_gain")
        if self.fallback_to_b != (self.action_id == "B"):
            raise ProtocolError("Static selection fallback flag drifted.")
        expected = canonical_hash(self._unhashed_payload())
        if self.selection_hash:
            if _sha256(self.selection_hash, "selection_hash") != expected:
                raise ProtocolError("Static selection hash drifted.")
        else:
            object.__setattr__(self, "selection_hash", expected)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "action_id": self.action_id,
            "exact_gain": self.exact_gain,
            "runner_up_gain": self.runner_up_gain,
            "fallback_to_b": self.fallback_to_b,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "selection_hash": self.selection_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "StaticSelection":
        return cls(
            action_id=str(payload["action_id"]),
            exact_gain=float(payload["exact_gain"]),
            runner_up_gain=float(payload["runner_up_gain"]),
            fallback_to_b=bool(payload["fallback_to_b"]),
            selection_hash=str(payload["selection_hash"]),
        )


@dataclass(frozen=True)
class CaseDecision:
    method_id: str
    case_id: str
    selected_action_id: str
    challenger_action_id: str
    predicted_gain: float
    standard_error: float
    lower_confidence_bound: float
    reason: str
    decision_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("method_id", "case_id", "selected_action_id", "challenger_action_id", "reason"):
            _text(getattr(self, name), name)
        if self.selected_action_id not in {"B", self.challenger_action_id}:
            raise ProtocolError("Case decision may choose only B or its sealed challenger.")
        for name in ("predicted_gain", "standard_error", "lower_confidence_bound"):
            _finite(getattr(self, name), name)
        if self.standard_error < 0.0:
            raise ProtocolError("Decision standard error cannot be negative.")
        expected = canonical_hash(self._unhashed_payload())
        if self.decision_hash:
            if _sha256(self.decision_hash, "decision_hash") != expected:
                raise ProtocolError("Case decision hash drifted.")
        else:
            object.__setattr__(self, "decision_hash", expected)

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "method_id": self.method_id,
            "case_id": self.case_id,
            "selected_action_id": self.selected_action_id,
            "challenger_action_id": self.challenger_action_id,
            "predicted_gain": self.predicted_gain,
            "standard_error": self.standard_error,
            "lower_confidence_bound": self.lower_confidence_bound,
            "reason": self.reason,
            "evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "decision_hash": self.decision_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CaseDecision":
        return cls(
            method_id=str(payload["method_id"]),
            case_id=str(payload["case_id"]),
            selected_action_id=str(payload["selected_action_id"]),
            challenger_action_id=str(payload["challenger_action_id"]),
            predicted_gain=float(payload["predicted_gain"]),
            standard_error=float(payload["standard_error"]),
            lower_confidence_bound=float(payload["lower_confidence_bound"]),
            reason=str(payload["reason"]),
            decision_hash=str(payload["decision_hash"]),
        )


def hash_decision_inputs(
    *,
    model: TwoHeadRidgeModel,
    calibration: DirectionSharedCalibration,
    selection: StaticSelection,
    features: Sequence[CaseActionFeatures],
) -> str:
    """Hash the complete pre-evaluation decision surface.

    There is intentionally no labels argument: held-evaluation labels cannot
    influence or even be represented in this seal.
    """

    return canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "model_hash": model.model_hash,
            "calibration_hash": calibration.calibration_hash,
            "selection_hash": selection.selection_hash,
            "features": [row.to_payload() for row in sorted(features)],
            "evaluation_labels_used": False,
        }
    )


__all__ = (
    "CalibrationRow",
    "CaseActionFeatures",
    "CaseDecision",
    "ContributionTarget",
    "DirectionSharedCalibration",
    "DonorRow",
    "HeadModel",
    "SCHEMA_VERSION",
    "StaticSelection",
    "TwoHeadPrediction",
    "TwoHeadRidgeModel",
    "canonical_hash",
    "hash_decision_inputs",
)
