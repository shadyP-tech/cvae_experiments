"""Typed, hash-bound route, probability, phase, and label-role contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .constants import (
    CENTERS,
    ENDPOINT_METHOD_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_ROW_COUNT,
    SEED_PAIR_COUNT,
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_GEOMETRY_ID,
    physical_action_ids,
)
from .hashing import canonical_hash, require_digest, require_sha256


def _finite_probability_array(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or bool(np.any((array < 0.0) | (array > 1.0)))
    ):
        raise ProtocolError("PCSI-RACR probability array shape or range drifted.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CenterProbabilitySurface:
    """Label-free exact-nine probability arrays for one target center."""

    center: str
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    seed_probabilities: Mapping[str, np.ndarray]
    probability_store_hash: str
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        center = str(self.center)
        samples = tuple(str(value) for value in self.sample_ids)
        cases = tuple(str(value) for value in self.case_ids)
        actions = physical_action_ids(center)
        if (
            center not in CENTERS
            or not samples
            or len(samples) != len(cases)
            or len(samples) != len(set(samples))
            or any(not value for value in cases)
            or tuple(self.seed_probabilities) != actions
        ):
            raise ProtocolError("PCSI-RACR center probability identity or action order drifted.")
        require_digest(self.probability_store_hash, "probability_store_hash")
        arrays = {
            action: _finite_probability_array(
                self.seed_probabilities[action], shape=(SEED_PAIR_COUNT, len(samples))
            )
            for action in actions
        }
        payload = {
            "schema_version": "fixed_bank_pcsi_racr_center_surface_v1",
            "center": center,
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "probability_store_hash": self.probability_store_hash,
            "action_array_sha256": {
                action: sha256_array(arrays[action]) for action in actions
            },
            "storage_dtype": "float32",
            "reduction_dtype": "float64",
            "labels_used": False,
        }
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "seed_probabilities", MappingProxyType(arrays))
        object.__setattr__(self, "surface_hash", canonical_hash(payload))

    @property
    def cases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.case_ids))

    def positions(self, case_id: object) -> np.ndarray:
        case = str(case_id)
        result = np.flatnonzero(np.asarray(self.case_ids, dtype=object) == case)
        if not len(result):
            raise ProtocolError("PCSI-RACR requested case is absent from the center surface.")
        return result

    def exact_nine_mean(self, action_id: object) -> np.ndarray:
        try:
            values = self.seed_probabilities[str(action_id)]
        except KeyError as exc:
            raise ProtocolError("PCSI-RACR requested physical action is absent.") from exc
        return np.mean(values.astype(np.float64, copy=False), axis=0, dtype=np.float64)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            CenterProbabilitySurface,
            (
                self.center,
                self.sample_ids,
                self.case_ids,
                dict(self.seed_probabilities),
                self.probability_store_hash,
            ),
        )


@dataclass(frozen=True)
class PhysicalProbabilitySurface:
    centers: Mapping[str, CenterProbabilitySurface]
    probability_store_hash: str
    strict_canonical_topology: bool = True
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        require_digest(self.probability_store_hash, "probability_store_hash")
        rows = {str(key): value for key, value in self.centers.items()}
        if tuple(rows) != CENTERS or any(
            not isinstance(value, CenterProbabilitySurface)
            or value.center != center
            or value.probability_store_hash != self.probability_store_hash
            for center, value in rows.items()
        ):
            raise ProtocolError("PCSI-RACR physical surface must contain all nine centers in order.")
        if self.strict_canonical_topology:
            counts = {center: len(rows[center].cases) for center in CENTERS}
            if (
                counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER)
                or sum(len(rows[center].sample_ids) for center in CENTERS)
                != EXPECTED_TEST_ROW_COUNT
            ):
                raise ProtocolError("PCSI-RACR physical surface canonical topology drifted.")
        payload = {
            "schema_version": "fixed_bank_pcsi_racr_physical_surface_v1",
            "probability_store_hash": self.probability_store_hash,
            "center_surface_hashes": {
                center: rows[center].surface_hash for center in CENTERS
            },
            "strict_canonical_topology": self.strict_canonical_topology,
            "labels_used": False,
        }
        object.__setattr__(self, "centers", MappingProxyType(rows))
        object.__setattr__(self, "surface_hash", canonical_hash(payload))


@dataclass(frozen=True, order=True)
class BinaryLabel:
    center: str
    case_id: str
    sample_id: str
    value: int
    scope: str

    def __post_init__(self) -> None:
        if (
            self.center not in CENTERS
            or not self.case_id
            or not self.sample_id
            or self.value not in (0, 1)
            or not self.scope
        ):
            raise ProtocolError("PCSI-RACR scoped binary label drifted.")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.center, self.case_id, self.sample_id


@dataclass(frozen=True)
class EndpointCasePrediction:
    center: str
    case_id: str
    sample_ids: tuple[str, ...]
    probabilities: Mapping[str, tuple[float, ...]]
    state_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        rows = {
            str(method): tuple(float(value) for value in probabilities)
            for method, probabilities in self.probabilities.items()
        }
        if (
            self.center not in CENTERS
            or not self.case_id
            or not samples
            or len(samples) != len(set(samples))
            or tuple(rows) != ENDPOINT_METHOD_IDS
            or any(
                len(values) != len(samples)
                or any(
                    not math.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in values
                )
                for values in rows.values()
            )
        ):
            raise ProtocolError("PCSI-RACR endpoint prediction topology drifted.")
        require_sha256(self.state_hash, "endpoint_state_hash")
        payload = {
            "schema_version": "fixed_bank_pcsi_racr_endpoint_prediction_v1",
            "center": self.center,
            "case_id": self.case_id,
            "sample_ids": list(samples),
            "probabilities": {
                method: list(rows[method]) for method in ENDPOINT_METHOD_IDS
            },
            "state_hash": self.state_hash,
        }
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probabilities", MappingProxyType(rows))
        object.__setattr__(self, "prediction_hash", canonical_hash(payload))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            EndpointCasePrediction,
            (
                self.center,
                self.case_id,
                self.sample_ids,
                dict(self.probabilities),
                self.state_hash,
            ),
        )


@dataclass(frozen=True, order=True)
class TargetRouteKey:
    outer_center: str
    case_id: str

    def __post_init__(self) -> None:
        if self.outer_center not in CENTERS or not self.case_id:
            raise ProtocolError("PCSI-RACR target route identity drifted.")


@dataclass(frozen=True, order=True)
class PseudoRouteKey:
    outer_center: str
    donor_center: str
    case_id: str

    def __post_init__(self) -> None:
        if (
            self.outer_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_center == self.donor_center
            or not self.case_id
        ):
            raise ProtocolError("PCSI-RACR pseudo route identity drifted.")


@dataclass(frozen=True, order=True)
class TargetReferenceKey:
    outer_center: str
    reference_center: str
    case_id: str

    def __post_init__(self) -> None:
        if (
            self.outer_center not in CENTERS
            or self.reference_center not in CENTERS
            or self.outer_center == self.reference_center
            or not self.case_id
        ):
            raise ProtocolError("PCSI-RACR target-reference identity drifted.")


@dataclass(frozen=True, order=True)
class PseudoReferenceKey:
    outer_center: str
    donor_center: str
    reference_center: str
    case_id: str

    def __post_init__(self) -> None:
        if (
            any(
                center not in CENTERS
                for center in (
                    self.outer_center,
                    self.donor_center,
                    self.reference_center,
                )
            )
            or len(
                {
                    self.outer_center,
                    self.donor_center,
                    self.reference_center,
                }
            )
            != 3
            or not self.case_id
        ):
            raise ProtocolError("PCSI-RACR pseudo-reference identity drifted.")


@dataclass(frozen=True, order=True)
class GeometryKey:
    geometry_id: str

    def __post_init__(self) -> None:
        if self.geometry_id not in {
            PROJECTION_GEOMETRY_ID,
            UNPROJECTED_GEOMETRY_ID,
        }:
            raise ProtocolError("PCSI-RACR geometry identity drifted.")


@dataclass(frozen=True)
class PhaseSeal:
    schema_version: str
    phase: str
    parent_hash: str
    member_hashes: tuple[str, ...]
    count: int
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        members = tuple(self.member_hashes)
        if (
            not self.schema_version
            or not self.phase
            or self.count != len(members)
            or any(require_sha256(value, "phase_member_hash") != value for value in members)
        ):
            raise ProtocolError("PCSI-RACR phase seal drifted.")
        if self.parent_hash:
            require_sha256(self.parent_hash, "phase_parent_hash")
        payload = {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "parent_hash": self.parent_hash,
            "member_hashes": list(members),
            "count": self.count,
        }
        object.__setattr__(self, "member_hashes", members)
        object.__setattr__(self, "seal_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "parent_hash": self.parent_hash,
            "member_hashes": list(self.member_hashes),
            "count": self.count,
            "seal_hash": self.seal_hash,
        }


@dataclass(frozen=True, order=True)
class LabelRoleRecord:
    label_identity_hash: str
    label_center: str
    label_case_id: str
    outer_center: str
    route_case_id: str
    role: str
    phase: str
    permitted: bool
    record_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        require_sha256(self.label_identity_hash, "label_identity_hash")
        if (
            self.label_center not in CENTERS
            or self.outer_center not in CENTERS
            or not self.label_case_id
            or not self.route_case_id
            or not self.role
            or not self.phase
        ):
            raise ProtocolError("PCSI-RACR label-role record drifted.")
        payload = {
            "schema_version": "fixed_bank_pcsi_racr_label_role_record_v1",
            "label_identity_hash": self.label_identity_hash,
            "label_center": self.label_center,
            "label_case_id": self.label_case_id,
            "outer_center": self.outer_center,
            "route_case_id": self.route_case_id,
            "role": self.role,
            "phase": self.phase,
            "permitted": self.permitted,
            "raw_label_persisted": False,
        }
        object.__setattr__(self, "record_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_label_role_record_v1",
            "label_identity_hash": self.label_identity_hash,
            "label_center": self.label_center,
            "label_case_id": self.label_case_id,
            "outer_center": self.outer_center,
            "route_case_id": self.route_case_id,
            "role": self.role,
            "phase": self.phase,
            "permitted": self.permitted,
            "raw_label_persisted": False,
            "record_hash": self.record_hash,
        }


__all__ = (
    "BinaryLabel",
    "CenterProbabilitySurface",
    "EndpointCasePrediction",
    "GeometryKey",
    "LabelRoleRecord",
    "PhaseSeal",
    "PhysicalProbabilitySurface",
    "PseudoReferenceKey",
    "PseudoRouteKey",
    "TargetReferenceKey",
    "TargetRouteKey",
)
