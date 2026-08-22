"""Typed, hash-bound probability and scoped-label contracts for CBPUPR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_array
from .constants import (
    CANONICAL_PHYSICAL_ROW_ORDER,
    CENTERS,
    ENDPOINT_METHOD_IDS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_ROW_COUNT,
    SEED_PAIR_COUNT,
    physical_action_ids,
)
from .hashing import canonical_hash, require_digest, require_sha256
from .row_order import (
    require_canonical_center_row_order,
    require_canonical_sample_ids,
)


def _finite_probability_array(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float32)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or bool(np.any((array < 0.0) | (array > 1.0)))
    ):
        raise ProtocolError("CBPUPR probability array shape or range drifted.")
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
        identity_error = (
            "CBPUPR center probability identity, row, or action order drifted."
        )
        samples, cases = require_canonical_center_row_order(
            self.sample_ids,
            self.case_ids,
            error_message=identity_error,
        )
        actions = physical_action_ids(center)
        if (
            center not in CENTERS
            or tuple(self.seed_probabilities) != actions
        ):
            raise ProtocolError(identity_error)
        require_digest(self.probability_store_hash, "probability_store_hash")
        arrays = {
            action: _finite_probability_array(
                self.seed_probabilities[action], shape=(SEED_PAIR_COUNT, len(samples))
            )
            for action in actions
        }
        payload = {
            "schema_version": "fixed_bank_cbpupr_center_surface_v1",
            "center": center,
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "row_order": CANONICAL_PHYSICAL_ROW_ORDER,
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
            raise ProtocolError("CBPUPR requested case is absent from the center surface.")
        return result

    def exact_nine_mean(self, action_id: object) -> np.ndarray:
        try:
            values = self.seed_probabilities[str(action_id)]
        except KeyError as exc:
            raise ProtocolError("CBPUPR requested physical action is absent.") from exc
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
            raise ProtocolError("CBPUPR physical surface must contain all nine centers in order.")
        if self.strict_canonical_topology:
            counts = {center: len(rows[center].cases) for center in CENTERS}
            if (
                counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER)
                or sum(len(rows[center].sample_ids) for center in CENTERS)
                != EXPECTED_TEST_ROW_COUNT
            ):
                raise ProtocolError("CBPUPR physical surface canonical topology drifted.")
        payload = {
            "schema_version": "fixed_bank_cbpupr_physical_surface_v1",
            "probability_store_hash": self.probability_store_hash,
            "center_surface_hashes": {
                center: rows[center].surface_hash for center in CENTERS
            },
            "row_order": CANONICAL_PHYSICAL_ROW_ORDER,
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
            raise ProtocolError("CBPUPR scoped binary label drifted.")

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
        topology_error = "CBPUPR endpoint prediction topology drifted."
        samples = require_canonical_sample_ids(
            self.sample_ids,
            error_message=topology_error,
        )
        rows = {
            str(method): tuple(
                float(value)
                for value in _finite_probability_array(
                    probabilities, shape=(len(samples),)
                )
            )
            for method, probabilities in self.probabilities.items()
        }
        if (
            self.center not in CENTERS
            or not self.case_id
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
            raise ProtocolError(topology_error)
        require_sha256(self.state_hash, "endpoint_state_hash")
        payload = {
            "schema_version": "fixed_bank_cbpupr_endpoint_prediction_v1",
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


__all__ = (
    "BinaryLabel",
    "CenterProbabilitySurface",
    "EndpointCasePrediction",
    "PhysicalProbabilitySurface",
)
