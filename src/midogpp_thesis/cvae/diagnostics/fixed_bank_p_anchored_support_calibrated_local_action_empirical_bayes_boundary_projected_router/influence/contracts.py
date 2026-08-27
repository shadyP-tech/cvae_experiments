"""Immutable scientific contracts for SCALE-BP action influence estimation.

The sign convention is deliberately explicit: balanced-accuracy gain is
favourable when positive, while Brier and logarithmic *loss deltas* are
favourable when non-positive.  Contracts contain only scalars and tuples so
they remain deterministic and safe to move through spawned worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_FAMILIES, DIRECTIONS
from ..protocol import ProtocolError


def _finite_tuple(values: object, *, role: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"SCALE-BP {role} is not a numeric vector.") from exc
    if not result or not all(math.isfinite(value) for value in result):
        raise ProtocolError(f"SCALE-BP {role} must be a nonempty finite vector.")
    return result


@dataclass(frozen=True, slots=True)
class ActionMetricVector:
    """Three downstream action effects under one fixed denominator contract."""

    bacc_gain: float
    brier_loss_delta: float
    log_loss_delta: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.as_tuple())
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("SCALE-BP action metric vector is nonfinite.")
        object.__setattr__(self, "bacc_gain", values[0])
        object.__setattr__(self, "brier_loss_delta", values[1])
        object.__setattr__(self, "log_loss_delta", values[2])

    @classmethod
    def zeros(cls) -> "ActionMetricVector":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_iterable(cls, values: object) -> "ActionMetricVector":
        vector = _finite_tuple(values, role="metric vector")
        if len(vector) != 3:
            raise ProtocolError("SCALE-BP action metric vector has the wrong length.")
        return cls(*vector)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc_gain, self.brier_loss_delta, self.log_loss_delta

    def plus(self, other: "ActionMetricVector") -> "ActionMetricVector":
        return ActionMetricVector.from_iterable(
            left + right for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )

    def minus(self, other: "ActionMetricVector") -> "ActionMetricVector":
        return ActionMetricVector.from_iterable(
            left - right for left, right in zip(self.as_tuple(), other.as_tuple(), strict=True)
        )

    def scaled(self, weights: object) -> "ActionMetricVector":
        scale = _finite_tuple(weights, role="metric scaling")
        if len(scale) != 3:
            raise ProtocolError("SCALE-BP action metric scaling has the wrong length.")
        return ActionMetricVector.from_iterable(
            value * weight for value, weight in zip(self.as_tuple(), scale, strict=True)
        )

    def to_payload(self) -> dict[str, float]:
        return {
            "bacc_gain": self.bacc_gain,
            "brier_loss_delta": self.brier_loss_delta,
            "log_loss_delta": self.log_loss_delta,
        }


@dataclass(frozen=True, slots=True)
class MetricStandardError:
    """Non-negative uncertainty in the metric order used by SCALE-BP."""

    bacc: float
    brier: float
    log: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.as_tuple())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ProtocolError("SCALE-BP metric uncertainty is invalid.")
        object.__setattr__(self, "bacc", values[0])
        object.__setattr__(self, "brier", values[1])
        object.__setattr__(self, "log", values[2])

    @classmethod
    def zeros(cls) -> "MetricStandardError":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_iterable(cls, values: object) -> "MetricStandardError":
        vector = tuple(float(value) for value in values)  # type: ignore[arg-type]
        if len(vector) != 3:
            raise ProtocolError("SCALE-BP metric uncertainty has the wrong length.")
        return cls(*vector)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc, self.brier, self.log

    def to_payload(self) -> dict[str, float]:
        return {"bacc": self.bacc, "brier": self.brier, "log": self.log}


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    """Label-free fixed descriptor for one case/action candidate."""

    case_id: str
    action_id: str
    family: str
    direction: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    crossing_count: int
    row_count: int
    baseline_probability_hash: str
    action_probability_hash: str
    endpoint_probability_hash: str
    descriptor_hash: str = field(init=False)

    def __post_init__(self) -> None:
        case_id = str(self.case_id)
        action_id = str(self.action_id)
        family = str(self.family)
        direction = str(self.direction)
        names = tuple(str(name) for name in self.feature_names)
        values = _finite_tuple(self.values, role="descriptor values")
        crossing_count = int(self.crossing_count)
        row_count = int(self.row_count)
        if (
            not case_id
            or not action_id
            or family not in ACTION_FAMILIES
            or direction not in DIRECTIONS
            or not names
            or len(names) != len(set(names))
            or len(names) != len(values)
            or row_count <= 0
            or crossing_count <= 0
            or crossing_count > row_count
        ):
            raise ProtocolError("SCALE-BP action descriptor identity drifted.")
        baseline_hash = require_sha256(
            self.baseline_probability_hash, "baseline probability hash"
        )
        action_hash = require_sha256(
            self.action_probability_hash, "action probability hash"
        )
        endpoint_hash = require_sha256(
            self.endpoint_probability_hash, "endpoint probability hash"
        )
        payload = {
            "schema_version": "scale_bp_action_descriptor_v1",
            "case_id": case_id,
            "action_id": action_id,
            "family": family,
            "direction": direction,
            "feature_names": names,
            "values": values,
            "crossing_count": crossing_count,
            "row_count": row_count,
            "baseline_probability_hash": baseline_hash,
            "action_probability_hash": action_hash,
            "endpoint_probability_hash": endpoint_hash,
            "label_free": True,
        }
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "crossing_count", crossing_count)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "baseline_probability_hash", baseline_hash)
        object.__setattr__(self, "action_probability_hash", action_hash)
        object.__setattr__(self, "endpoint_probability_hash", endpoint_hash)
        object.__setattr__(self, "descriptor_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_action_descriptor_v1",
            "case_id": self.case_id,
            "action_id": self.action_id,
            "family": self.family,
            "direction": self.direction,
            "feature_names": self.feature_names,
            "values": self.values,
            "crossing_count": self.crossing_count,
            "row_count": self.row_count,
            "baseline_probability_hash": self.baseline_probability_hash,
            "action_probability_hash": self.action_probability_hash,
            "endpoint_probability_hash": self.endpoint_probability_hash,
            "label_free": True,
            "descriptor_hash": self.descriptor_hash,
        }


__all__ = (
    "ActionDescriptor",
    "ActionMetricVector",
    "MetricStandardError",
    "require_sha256",
)
