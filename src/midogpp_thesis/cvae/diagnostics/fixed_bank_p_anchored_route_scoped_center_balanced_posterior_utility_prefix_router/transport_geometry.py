"""Structural transport gate and zero-aware numeric transport audit.

Only immutable lineage and route noninterference authorize a decision.  Numeric
distance is persisted for diagnosis but is deliberately not an authorization
gate; zero-scale coordinates are dropped instead of divided by an epsilon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_hash


@dataclass(frozen=True)
class StructuralTransportGate:
    target_center: str
    probability_lineage_match: bool
    plan_lineage_match: bool
    target_excluded_from_fit: bool
    own_route_noninterference: bool
    finite_inputs: bool
    reason_codes: tuple[str, ...] = field(init=False)
    gate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.target_center:
            raise ProtocolError("CBPUPR structural transport target is empty.")
        checks = (
            ("PROBABILITY_LINEAGE_MISMATCH", self.probability_lineage_match),
            ("PLAN_LINEAGE_MISMATCH", self.plan_lineage_match),
            ("TARGET_NOT_EXCLUDED", self.target_excluded_from_fit),
            ("OWN_ROUTE_INTERFERENCE", self.own_route_noninterference),
            ("NONFINITE_INPUT", self.finite_inputs),
        )
        reasons = tuple(reason for reason, passed in checks if not passed)
        object.__setattr__(
            self,
            "reason_codes",
            ("STRUCTURAL_TRANSPORT_PASS",) if not reasons else reasons,
        )
        object.__setattr__(
            self,
            "gate_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_structural_transport_gate_v1",
                    "target_center": self.target_center,
                    "probability_lineage_match": self.probability_lineage_match,
                    "plan_lineage_match": self.plan_lineage_match,
                    "target_excluded_from_fit": self.target_excluded_from_fit,
                    "own_route_noninterference": self.own_route_noninterference,
                    "finite_inputs": self.finite_inputs,
                    "reason_codes": list(self.reason_codes),
                }
            ),
        )

    @property
    def passed(self) -> bool:
        return self.reason_codes == ("STRUCTURAL_TRANSPORT_PASS",)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "StructuralTransportGate":
        row = cls(
            str(payload["target_center"]),
            bool(payload["probability_lineage_match"]),
            bool(payload["plan_lineage_match"]),
            bool(payload["target_excluded_from_fit"]),
            bool(payload["own_route_noninterference"]),
            bool(payload["finite_inputs"]),
        )
        if "gate_hash" in payload and str(payload["gate_hash"]) != row.gate_hash:
            raise ProtocolError("CBPUPR structural transport hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "probability_lineage_match": self.probability_lineage_match,
            "plan_lineage_match": self.plan_lineage_match,
            "target_excluded_from_fit": self.target_excluded_from_fit,
            "own_route_noninterference": self.own_route_noninterference,
            "finite_inputs": self.finite_inputs,
            "reason_codes": list(self.reason_codes),
            "passed": self.passed,
            "gate_hash": self.gate_hash,
        }


@dataclass(frozen=True)
class NumericDimensionAudit:
    feature_name: str
    feature_kind: str
    target_value: float
    reference_median: float
    reference_mad: float
    active: bool
    standardized_distance: float | None
    zero_scale_novelty: bool
    sparse_reference_rate: float | None

    def __post_init__(self) -> None:
        if (
            not self.feature_name
            or self.feature_kind not in ("continuous", "sparse_pattern")
            or not all(
                math.isfinite(value)
                for value in (
                    self.target_value,
                    self.reference_median,
                    self.reference_mad,
                )
            )
            or self.reference_mad < 0.0
            or (self.active != (self.standardized_distance is not None))
            or (
                self.standardized_distance is not None
                and not math.isfinite(self.standardized_distance)
            )
            or (
                self.sparse_reference_rate is not None
                and not 0.0 <= self.sparse_reference_rate <= 1.0
            )
        ):
            raise ProtocolError("CBPUPR numeric transport dimension drifted.")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NumericDimensionAudit":
        distance = payload.get("standardized_distance")
        rate = payload.get("sparse_reference_rate")
        return cls(
            str(payload["feature_name"]),
            str(payload["feature_kind"]),
            float(payload["target_value"]),
            float(payload["reference_median"]),
            float(payload["reference_mad"]),
            bool(payload["active"]),
            None if distance is None else float(distance),
            bool(payload["zero_scale_novelty"]),
            None if rate is None else float(rate),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "feature_kind": self.feature_kind,
            "target_value": self.target_value,
            "reference_median": self.reference_median,
            "reference_mad": self.reference_mad,
            "active": self.active,
            "standardized_distance": self.standardized_distance,
            "zero_scale_novelty": self.zero_scale_novelty,
            "sparse_reference_rate": self.sparse_reference_rate,
        }


@dataclass(frozen=True)
class NumericTransportAudit:
    target_center: str
    reference_centers: tuple[str, ...]
    dimensions: tuple[NumericDimensionAudit, ...]
    active_continuous_dimension_count: int
    zero_scale_dimensions: tuple[str, ...]
    zero_scale_novelty_dimensions: tuple[str, ...]
    sparse_pattern_mismatch_count: int
    l2_distance: float
    maximum_absolute_distance: float
    audit_hash: str = field(init=False)

    def __post_init__(self) -> None:
        active = tuple(
            row.standardized_distance
            for row in self.dimensions
            if row.feature_kind == "continuous" and row.active
        )
        if (
            not self.target_center
            or len(set(self.reference_centers)) != len(self.reference_centers)
            or self.target_center in self.reference_centers
            or len({row.feature_name for row in self.dimensions}) != len(self.dimensions)
            or self.active_continuous_dimension_count != len(active)
            or not math.isfinite(self.l2_distance)
            or not math.isfinite(self.maximum_absolute_distance)
        ):
            raise ProtocolError("CBPUPR numeric transport audit drifted.")
        object.__setattr__(
            self,
            "audit_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_numeric_transport_audit_v1",
                    "target_center": self.target_center,
                    "reference_centers": list(self.reference_centers),
                    "dimensions": [row.to_payload() for row in self.dimensions],
                    "active_continuous_dimension_count": self.active_continuous_dimension_count,
                    "zero_scale_dimensions": list(self.zero_scale_dimensions),
                    "zero_scale_novelty_dimensions": list(
                        self.zero_scale_novelty_dimensions
                    ),
                    "sparse_pattern_mismatch_count": self.sparse_pattern_mismatch_count,
                    "l2_distance": self.l2_distance,
                    "maximum_absolute_distance": self.maximum_absolute_distance,
                    "authorization_gate": False,
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "NumericTransportAudit":
        row = cls(
            str(payload["target_center"]),
            tuple(str(value) for value in payload["reference_centers"]),  # type: ignore[index]
            tuple(
                NumericDimensionAudit.from_payload(value)
                for value in payload["dimensions"]  # type: ignore[index]
            ),
            int(payload["active_continuous_dimension_count"]),
            tuple(str(value) for value in payload["zero_scale_dimensions"]),  # type: ignore[index]
            tuple(
                str(value) for value in payload["zero_scale_novelty_dimensions"]  # type: ignore[index]
            ),
            int(payload["sparse_pattern_mismatch_count"]),
            float(payload["l2_distance"]),
            float(payload["maximum_absolute_distance"]),
        )
        if "audit_hash" in payload and str(payload["audit_hash"]) != row.audit_hash:
            raise ProtocolError("CBPUPR numeric transport audit hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "reference_centers": list(self.reference_centers),
            "dimensions": [row.to_payload() for row in self.dimensions],
            "active_continuous_dimension_count": self.active_continuous_dimension_count,
            "zero_scale_dimensions": list(self.zero_scale_dimensions),
            "zero_scale_novelty_dimensions": list(self.zero_scale_novelty_dimensions),
            "sparse_pattern_mismatch_count": self.sparse_pattern_mismatch_count,
            "l2_distance": self.l2_distance,
            "maximum_absolute_distance": self.maximum_absolute_distance,
            "authorization_gate": False,
            "audit_hash": self.audit_hash,
        }


def audit_numeric_transport(
    *,
    target_center: str,
    target_vector: Sequence[float],
    reference_vectors_by_center: Mapping[str, Sequence[Sequence[float]]],
    feature_names: Sequence[str],
    sparse_feature_names: Sequence[str] = (),
    scale_floor: float = 1.0e-12,
) -> NumericTransportAudit:
    """Build a robust, center-balanced, zero-aware transport diagnostic."""

    names = tuple(str(value) for value in feature_names)
    target = np.asarray(tuple(target_vector), dtype=np.float64)
    centers = tuple(sorted(str(value) for value in reference_vectors_by_center))
    if (
        not names
        or len(set(names)) != len(names)
        or target.shape != (len(names),)
        or not np.isfinite(target).all()
        or target_center in centers
        or len(centers) < 3
    ):
        raise ProtocolError("CBPUPR numeric transport input topology drifted.")
    center_medians: list[np.ndarray] = []
    for center in centers:
        matrix = np.asarray(reference_vectors_by_center[center], dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(names)
            or not len(matrix)
            or not np.isfinite(matrix).all()
        ):
            raise ProtocolError("CBPUPR numeric transport reference drifted.")
        center_medians.append(np.median(matrix, axis=0))
    reference = np.asarray(center_medians, dtype=np.float64)
    medians = np.median(reference, axis=0)
    mads = np.median(np.abs(reference - medians), axis=0)
    sparse = frozenset(str(value) for value in sparse_feature_names)
    if not sparse.issubset(names):
        raise ProtocolError("CBPUPR sparse transport feature is unknown.")

    rows: list[NumericDimensionAudit] = []
    distances: list[float] = []
    zero_scale: list[str] = []
    novelty: list[str] = []
    sparse_mismatches = 0
    for index, name in enumerate(names):
        median = float(medians[index])
        mad = float(mads[index])
        value = float(target[index])
        if name in sparse:
            reference_pattern = reference[:, index] != 0.0
            target_pattern = value != 0.0
            rate = float(np.mean(reference_pattern, dtype=np.float64))
            mismatch = target_pattern != (rate >= 0.5)
            sparse_mismatches += int(mismatch)
            rows.append(
                NumericDimensionAudit(
                    name,
                    "sparse_pattern",
                    value,
                    median,
                    mad,
                    False,
                    None,
                    bool(mismatch and mad <= float(scale_floor)),
                    rate,
                )
            )
            if mad <= float(scale_floor):
                zero_scale.append(name)
                if mismatch:
                    novelty.append(name)
            continue
        if mad <= float(scale_floor):
            is_novel = not math.isclose(value, median, rel_tol=0.0, abs_tol=scale_floor)
            zero_scale.append(name)
            if is_novel:
                novelty.append(name)
            rows.append(
                NumericDimensionAudit(
                    name,
                    "continuous",
                    value,
                    median,
                    mad,
                    False,
                    None,
                    is_novel,
                    None,
                )
            )
            continue
        distance = (value - median) / (1.4826 * mad)
        distances.append(float(distance))
        rows.append(
            NumericDimensionAudit(
                name,
                "continuous",
                value,
                median,
                mad,
                True,
                float(distance),
                False,
                None,
            )
        )
    return NumericTransportAudit(
        str(target_center),
        centers,
        tuple(rows),
        len(distances),
        tuple(zero_scale),
        tuple(novelty),
        sparse_mismatches,
        float(np.linalg.norm(np.asarray(distances, dtype=np.float64))),
        max((abs(value) for value in distances), default=0.0),
    )


__all__ = (
    "NumericDimensionAudit",
    "NumericTransportAudit",
    "StructuralTransportGate",
    "audit_numeric_transport",
)
