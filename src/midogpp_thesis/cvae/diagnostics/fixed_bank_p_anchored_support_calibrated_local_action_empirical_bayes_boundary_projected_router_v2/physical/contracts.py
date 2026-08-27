"""Pure identities and numeric contracts for the SCALE-BP v2 physical layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
PHYSICAL_ACTION_COUNT_PER_TARGET = 10
PHYSICAL_CELL_COUNT = 810
ACTION_FAMILIES = ("B", "I", "R")
DIRECTIONS = ("zero_to_one", "one_to_zero")
ACTION_IDS = tuple(
    f"{family}::{direction}"
    for family in ACTION_FAMILIES
    for direction in DIRECTIONS
)
P_METHOD_ID = "P_PROTECTED"
PRIMARY_METHOD_ID = "SCALE_BP_V2_PRIMARY"
HARD_THRESHOLD = 0.5
PORTFOLIO_I_WEIGHT = 3.0 / 5.0
PORTFOLIO_R_WEIGHT = 2.0 / 5.0
ROBUST_ARM_COUNT = 9


def candidate_sources(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise GovernanceError("SCALE-BP v2 target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def physical_action_ids(target_center: object) -> tuple[str, ...]:
    return (
        "B",
        "U",
        *(f"A1::source={source}" for source in candidate_sources(target_center)),
    )


def split_action_id(action_id: object) -> tuple[str, str]:
    text = str(action_id)
    try:
        family, direction = text.split("::", 1)
    except ValueError as exc:
        raise GovernanceError("SCALE-BP v2 action identity is malformed.") from exc
    if family not in ACTION_FAMILIES or direction not in DIRECTIONS:
        raise GovernanceError("SCALE-BP v2 action identity is outside the six-cell menu.")
    return family, direction


def probability_vector(
    values: object,
    *,
    expected_length: int | None = None,
    dtype: np.dtype[np.floating] | type[np.floating] = np.float64,
) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    if (
        array.ndim != 1
        or len(array) == 0
        or (expected_length is not None and len(array) != int(expected_length))
        or not np.isfinite(array).all()
        or np.any((array < 0.0) | (array > 1.0))
    ):
        raise GovernanceError("SCALE-BP v2 probability vector drifted.")
    array.setflags(write=False)
    return array


def finite_vector(values: object, *, expected_length: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or (expected_length is not None and len(array) != int(expected_length))
        or not np.isfinite(array).all()
    ):
        raise GovernanceError("SCALE-BP v2 finite vector drifted.")
    array.setflags(write=False)
    return array


def array_sha256(values: object) -> str:
    array = np.ascontiguousarray(values)
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return hashlib.sha256(header + memoryview(array).cast("B")).hexdigest()


def binary_entropy(probabilities: object) -> np.ndarray:
    values = probability_vector(probabilities)
    clipped = np.clip(values, 1.0e-12, 1.0 - 1.0e-12)
    result = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log1p(-clipped))
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MetricVector:
    """BACC gain and proper-score deltas relative to protected P."""

    bacc: float
    brier: float
    log: float

    def __post_init__(self) -> None:
        values = (float(self.bacc), float(self.brier), float(self.log))
        if not all(math.isfinite(value) for value in values):
            raise GovernanceError("SCALE-BP v2 metric vector is nonfinite.")
        object.__setattr__(self, "bacc", values[0])
        object.__setattr__(self, "brier", values[1])
        object.__setattr__(self, "log", values[2])

    @classmethod
    def zeros(cls) -> "MetricVector":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_array(cls, values: object) -> "MetricVector":
        array = finite_vector(values, expected_length=3)
        return cls(*(float(value) for value in array))

    def as_array(self) -> np.ndarray:
        return finite_vector((self.bacc, self.brier, self.log), expected_length=3)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc, self.brier, self.log

    def to_payload(self) -> dict[str, float]:
        return {"bacc": self.bacc, "brier": self.brier, "log": self.log}


@dataclass(frozen=True, slots=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    feature_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.names)
        values = tuple(float(value) for value in self.values)
        if (
            not names
            or len(names) != len(set(names))
            or len(values) != len(names)
            or not all(name for name in names)
            or not all(math.isfinite(value) for value in values)
        ):
            raise GovernanceError("SCALE-BP v2 feature vector drifted.")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "feature_hash",
            canonical_hash({"names": names, "values": values}),
        )

    def as_array(self) -> np.ndarray:
        return finite_vector(self.values, expected_length=len(self.names))


def freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(values))


__all__ = (
    "ACTION_FAMILIES",
    "ACTION_IDS",
    "CENTERS",
    "DIRECTIONS",
    "FeatureVector",
    "GENERATION_SEEDS",
    "HARD_THRESHOLD",
    "MetricVector",
    "PHYSICAL_ACTION_COUNT_PER_TARGET",
    "PHYSICAL_CELL_COUNT",
    "PORTFOLIO_I_WEIGHT",
    "PORTFOLIO_R_WEIGHT",
    "PRIMARY_METHOD_ID",
    "P_METHOD_ID",
    "ROBUST_ARM_COUNT",
    "TRAINING_SEEDS",
    "array_sha256",
    "binary_entropy",
    "candidate_sources",
    "finite_vector",
    "freeze_mapping",
    "physical_action_ids",
    "probability_vector",
    "split_action_id",
)
