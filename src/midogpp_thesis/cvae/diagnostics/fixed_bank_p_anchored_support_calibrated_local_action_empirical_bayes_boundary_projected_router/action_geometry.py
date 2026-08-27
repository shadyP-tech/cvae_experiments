"""Deterministic, minimally perturbing SCALE-BP action geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hashing import canonical_hash
from .identity import ACTION_FAMILIES, DIRECTIONS
from .protocol import ProtocolError


HARD_THRESHOLD = np.float32(0.5)
LOW_BOUNDARY = np.nextafter(HARD_THRESHOLD, np.float32(-np.inf), dtype=np.float32)
HIGH_BOUNDARY = np.nextafter(
    HARD_THRESHOLD, np.float32(np.inf), dtype=np.float32
)


def canonical_probabilities(
    values: object,
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return a contiguous read-only float32 probability vector."""

    array = np.ascontiguousarray(values, dtype=np.float32)
    if (
        array.ndim != 1
        or len(array) <= 0
        or (expected_length is not None and len(array) != int(expected_length))
        or not np.isfinite(array).all()
        or np.any((array < np.float32(0.0)) | (array > np.float32(1.0)))
    ):
        raise ProtocolError("SCALE-BP probability vector drifted.")
    array.setflags(write=False)
    return array


def probability_hash(values: object) -> str:
    probabilities = canonical_probabilities(values)
    return canonical_hash(
        {
            "schema_version": "scale_bp_float32_probability_vector_v1",
            "dtype": "float32",
            "row_count": len(probabilities),
            # Float hex strings bind exact float32 values without depending on
            # platform byte order or NumPy's object serialization.
            "values": tuple(float(value).hex() for value in probabilities),
        }
    )


@dataclass(frozen=True, slots=True)
class BoundaryProjection:
    """One hard-threshold action plus its full-endpoint sensitivity control."""

    family: str
    direction: str
    baseline_probabilities: tuple[float, ...]
    projected_probabilities: tuple[float, ...]
    full_endpoint_probabilities: tuple[float, ...]
    crossing_indices: tuple[int, ...]
    source_endpoint_hash: str
    baseline_probability_hash: str = field(init=False)
    projected_probability_hash: str = field(init=False)
    full_endpoint_probability_hash: str = field(init=False)
    geometry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        family = str(self.family)
        direction = str(self.direction)
        baseline = canonical_probabilities(self.baseline_probabilities)
        projected = canonical_probabilities(
            self.projected_probabilities, expected_length=len(baseline)
        )
        full_endpoint = canonical_probabilities(
            self.full_endpoint_probabilities, expected_length=len(baseline)
        )
        crossing_indices = tuple(int(index) for index in self.crossing_indices)
        if (
            family not in ACTION_FAMILIES
            or direction not in DIRECTIONS
            or crossing_indices != tuple(sorted(set(crossing_indices)))
            or any(index < 0 or index >= len(baseline) for index in crossing_indices)
        ):
            raise ProtocolError("SCALE-BP boundary-projection identity drifted.")

        crossing = np.zeros(len(baseline), dtype=bool)
        crossing[list(crossing_indices)] = True
        if not np.array_equal(projected[~crossing], baseline[~crossing]) or not np.array_equal(
            full_endpoint[~crossing], baseline[~crossing]
        ):
            raise ProtocolError("SCALE-BP action changed a non-crossing probability.")
        actual_crossing = (baseline >= HARD_THRESHOLD) != (projected >= HARD_THRESHOLD)
        full_crossing = (baseline >= HARD_THRESHOLD) != (
            full_endpoint >= HARD_THRESHOLD
        )
        if not np.array_equal(actual_crossing, crossing) or not np.array_equal(
            full_crossing, crossing
        ):
            raise ProtocolError("SCALE-BP action crossing mask drifted.")
        if crossing_indices:
            expected_boundary = (
                HIGH_BOUNDARY if direction == "zero_to_one" else LOW_BOUNDARY
            )
            if not np.all(projected[crossing] == expected_boundary):
                raise ProtocolError("SCALE-BP projected action left the nearest boundary.")
            if direction == "zero_to_one" and not np.all(baseline[crossing] < HARD_THRESHOLD):
                raise ProtocolError("SCALE-BP zero-to-one action direction drifted.")
            if direction == "one_to_zero" and not np.all(baseline[crossing] >= HARD_THRESHOLD):
                raise ProtocolError("SCALE-BP one-to-zero action direction drifted.")

        baseline_values = tuple(float(value) for value in baseline)
        projected_values = tuple(float(value) for value in projected)
        full_values = tuple(float(value) for value in full_endpoint)
        baseline_hash = probability_hash(baseline)
        projected_hash = probability_hash(projected)
        full_hash = probability_hash(full_endpoint)
        payload = {
            "schema_version": "scale_bp_boundary_projection_v1",
            "family": family,
            "direction": direction,
            "row_count": len(baseline),
            "crossing_indices": crossing_indices,
            "source_endpoint_hash": str(self.source_endpoint_hash),
            "baseline_probability_hash": baseline_hash,
            "projected_probability_hash": projected_hash,
            "full_endpoint_probability_hash": full_hash,
            "off_mask_exact_p": True,
            "boundary_float32_hex": float(
                HIGH_BOUNDARY if direction == "zero_to_one" else LOW_BOUNDARY
            ).hex(),
        }
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "baseline_probabilities", baseline_values)
        object.__setattr__(self, "projected_probabilities", projected_values)
        object.__setattr__(self, "full_endpoint_probabilities", full_values)
        object.__setattr__(self, "crossing_indices", crossing_indices)
        object.__setattr__(self, "source_endpoint_hash", str(self.source_endpoint_hash))
        object.__setattr__(self, "baseline_probability_hash", baseline_hash)
        object.__setattr__(self, "projected_probability_hash", projected_hash)
        object.__setattr__(self, "full_endpoint_probability_hash", full_hash)
        object.__setattr__(self, "geometry_hash", canonical_hash(payload))

    @property
    def row_count(self) -> int:
        return len(self.baseline_probabilities)

    @property
    def crossing_count(self) -> int:
        return len(self.crossing_indices)

    @property
    def is_exact_p(self) -> bool:
        return self.crossing_count == 0

    @property
    def action_id(self) -> str:
        return f"{self.family}::{self.direction}"

    def projected_array(self) -> np.ndarray:
        return canonical_probabilities(self.projected_probabilities)

    def full_endpoint_array(self) -> np.ndarray:
        return canonical_probabilities(self.full_endpoint_probabilities)

    def crossing_mask(self) -> np.ndarray:
        mask = np.zeros(self.row_count, dtype=bool)
        mask[list(self.crossing_indices)] = True
        mask.setflags(write=False)
        return mask

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_boundary_projection_v1",
            "family": self.family,
            "direction": self.direction,
            "action_id": self.action_id,
            "row_count": self.row_count,
            "crossing_indices": self.crossing_indices,
            "source_endpoint_hash": self.source_endpoint_hash,
            "baseline_probability_hash": self.baseline_probability_hash,
            "projected_probability_hash": self.projected_probability_hash,
            "full_endpoint_probability_hash": self.full_endpoint_probability_hash,
            "off_mask_exact_p": True,
            "geometry_hash": self.geometry_hash,
        }


def build_boundary_projection(
    portfolio: object,
    endpoint: object,
    *,
    family: str,
    direction: str,
) -> BoundaryProjection:
    """Project only threshold-crossing endpoint rows to the closest side.

    The full endpoint is retained only on the same crossing mask as a sealed
    sensitivity control.  If no row crosses, both actions are byte-identical
    to P and the returned crossing set is empty.
    """

    family_id = str(family)
    direction_id = str(direction)
    if family_id not in ACTION_FAMILIES or direction_id not in DIRECTIONS:
        raise ProtocolError("SCALE-BP action family or direction drifted.")
    baseline = canonical_probabilities(portfolio)
    endpoint_values = canonical_probabilities(endpoint, expected_length=len(baseline))
    if direction_id == "zero_to_one":
        crossing = (baseline < HARD_THRESHOLD) & (endpoint_values >= HARD_THRESHOLD)
        boundary = HIGH_BOUNDARY
    elif direction_id == "one_to_zero":
        crossing = (baseline >= HARD_THRESHOLD) & (endpoint_values < HARD_THRESHOLD)
        boundary = LOW_BOUNDARY
    else:  # defended above, retained to keep this branch fail-closed.
        raise ProtocolError("SCALE-BP action direction drifted.")

    projected = np.array(baseline, dtype=np.float32, copy=True, order="C")
    full_endpoint = np.array(baseline, dtype=np.float32, copy=True, order="C")
    projected[crossing] = boundary
    full_endpoint[crossing] = endpoint_values[crossing]
    crossing_indices = tuple(int(index) for index in np.flatnonzero(crossing))
    return BoundaryProjection(
        family=family_id,
        direction=direction_id,
        baseline_probabilities=tuple(float(value) for value in baseline),
        projected_probabilities=tuple(float(value) for value in projected),
        full_endpoint_probabilities=tuple(float(value) for value in full_endpoint),
        crossing_indices=crossing_indices,
        source_endpoint_hash=probability_hash(endpoint_values),
    )


__all__ = (
    "BoundaryProjection",
    "HARD_THRESHOLD",
    "HIGH_BOUNDARY",
    "LOW_BOUNDARY",
    "build_boundary_projection",
    "canonical_probabilities",
    "probability_hash",
)
