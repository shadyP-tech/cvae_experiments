"""Leakage-auditable contracts for donor and route-local posterior fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import (
    ACTION_IDS,
    CENTERS,
    FeatureVector,
    MetricVector,
    candidate_sources,
)


@dataclass(frozen=True, slots=True)
class ScaleVector:
    bacc: float
    brier: float
    log: float

    def __post_init__(self) -> None:
        values = (float(self.bacc), float(self.brier), float(self.log))
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise GovernanceError("SCALE-BP v2 scale vector drifted.")
        object.__setattr__(self, "bacc", values[0])
        object.__setattr__(self, "brier", values[1])
        object.__setattr__(self, "log", values[2])

    @classmethod
    def zeros(cls) -> "ScaleVector":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_values(cls, values: object) -> "ScaleVector":
        rows = tuple(float(value) for value in values)  # type: ignore[arg-type]
        if len(rows) != 3:
            raise GovernanceError("SCALE-BP v2 scale vector width drifted.")
        return cls(*rows)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc, self.brier, self.log

    def to_payload(self) -> dict[str, float]:
        return {"bacc": self.bacc, "brier": self.brier, "log": self.log}


@dataclass(frozen=True, slots=True)
class DonorFitScope:
    """Strict H/J/d exclusion contract for one donor prediction regime."""

    outer_center: str
    prediction_center: str
    held_case_id: str
    training_case_ids_by_center: Mapping[str, tuple[str, ...]]
    source_excluded_centers: tuple[str, ...]
    role: str
    scope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer, prediction, held_case = (
            str(self.outer_center),
            str(self.prediction_center),
            str(self.held_case_id),
        )
        cases = {
            str(center): tuple(str(case) for case in center_cases)
            for center, center_cases in self.training_case_ids_by_center.items()
        }
        excluded = tuple(str(center) for center in self.source_excluded_centers)
        required_excluded = {outer, prediction}
        canonical_excluded = tuple(
            center for center in CENTERS if center in set(excluded)
        )
        expected_training_centers = tuple(
            center for center in CENTERS if center not in set(excluded)
        )
        if (
            outer not in CENTERS
            or prediction not in CENTERS
            or not held_case
            or self.role not in {"FINAL_H_C", "PSEUDO_H_J_D"}
            or (self.role == "PSEUDO_H_J_D" and outer == prediction)
            or (self.role == "FINAL_H_C" and outer != prediction)
            or tuple(cases) != expected_training_centers
            or any(not center_cases for center_cases in cases.values())
            or any(len(center_cases) != len(set(center_cases)) for center_cases in cases.values())
            or len(excluded) != len(set(excluded))
            or not required_excluded <= set(excluded)
            or excluded != canonical_excluded
            or len(expected_training_centers) < 3
        ):
            raise GovernanceError("SCALE-BP v2 donor H/J/d scope drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "prediction_center", prediction)
        object.__setattr__(self, "held_case_id", held_case)
        object.__setattr__(self, "training_case_ids_by_center", MappingProxyType(cases))
        object.__setattr__(self, "source_excluded_centers", excluded)
        object.__setattr__(
            self,
            "scope_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_donor_fit_scope_v1",
                    "outer_center": outer,
                    "prediction_center": prediction,
                    "held_case_id": held_case,
                    "training_case_ids_by_center": cases,
                    "source_excluded_centers": excluded,
                    "role": self.role,
                    "outer_center_excluded": True,
                    "prediction_center_excluded": True,
                    "held_case_excluded": True,
                }
            ),
        )

    @property
    def training_centers(self) -> tuple[str, ...]:
        return tuple(self.training_case_ids_by_center)


@dataclass(frozen=True, slots=True)
class DonorObservation:
    query_center: str
    case_id: str
    action_id: str
    descriptor: FeatureVector
    realized: MetricVector
    source_centers: tuple[str, ...]
    scope_hash: str
    observation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        sources = tuple(str(center) for center in self.source_centers)
        if (
            self.query_center not in CENTERS
            or not self.case_id
            or self.action_id not in ACTION_IDS
            or not sources
            or len(sources) != len(set(sources))
            or self.query_center in sources
            or any(source not in CENTERS for source in sources)
            or not self.scope_hash
        ):
            raise GovernanceError("SCALE-BP v2 donor observation drifted.")
        object.__setattr__(self, "source_centers", sources)
        object.__setattr__(
            self,
            "observation_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_donor_observation_v1",
                    "query_center": self.query_center,
                    "case_id": self.case_id,
                    "action_id": self.action_id,
                    "descriptor_hash": self.descriptor.feature_hash,
                    "realized": self.realized.to_payload(),
                    "source_centers": sources,
                    "scope_hash": self.scope_hash,
                    "raw_labels_persisted": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalResidualObservation:
    target_center: str
    route_case_id: str
    support_case_id: str
    action_id: str
    descriptor: FeatureVector
    residual: MetricVector
    donor_prediction_hash: str
    support_scope_hash: str
    endpoint_plan_hash: str
    support_excluded_case_ids: tuple[str, ...]
    outer_held_case_id: str
    observation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded_cases = tuple(str(value) for value in self.support_excluded_case_ids)
        outer_held = str(self.outer_held_case_id)
        if (
            self.target_center not in CENTERS
            or not self.route_case_id
            or not self.support_case_id
            or self.route_case_id == self.support_case_id
            or self.action_id not in ACTION_IDS
            or not self.donor_prediction_hash
            or not self.support_scope_hash
            or not self.endpoint_plan_hash
            or self.support_case_id not in excluded_cases
            or self.route_case_id not in excluded_cases
            or outer_held != self.route_case_id
            or len(excluded_cases) != len(set(excluded_cases))
        ):
            raise GovernanceError("SCALE-BP v2 local residual observation drifted.")
        object.__setattr__(self, "support_excluded_case_ids", excluded_cases)
        object.__setattr__(self, "outer_held_case_id", outer_held)
        object.__setattr__(
            self,
            "observation_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_local_residual_observation_v2",
                    "target_center": self.target_center,
                    "route_case_id": self.route_case_id,
                    "support_case_id": self.support_case_id,
                    "action_id": self.action_id,
                    "descriptor_hash": self.descriptor.feature_hash,
                    "residual": self.residual.to_payload(),
                    "donor_prediction_hash": self.donor_prediction_hash,
                    "support_scope_hash": self.support_scope_hash,
                    "endpoint_plan_hash": self.endpoint_plan_hash,
                    "support_excluded_case_ids": excluded_cases,
                    "outer_held_case_id": outer_held,
                    "route_local_only": True,
                }
            ),
        )


__all__ = (
    "DonorFitScope",
    "DonorObservation",
    "LocalResidualObservation",
    "ScaleVector",
)
