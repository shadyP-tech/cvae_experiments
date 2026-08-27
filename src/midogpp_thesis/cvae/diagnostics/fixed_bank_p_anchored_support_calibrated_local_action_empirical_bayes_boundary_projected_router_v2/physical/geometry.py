"""Nearest-threshold boundary projection with byte-exact P off-mask."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from .contracts import (
    ACTION_FAMILIES,
    DIRECTIONS,
    HARD_THRESHOLD,
    array_sha256,
    probability_vector,
)


LOW_BOUNDARY_FLOAT32 = float(
    np.nextafter(np.float32(0.5), np.float32(-np.inf), dtype=np.float32)
)
HIGH_BOUNDARY_FLOAT32 = float(
    np.nextafter(np.float32(0.5), np.float32(np.inf), dtype=np.float32)
)


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryAction:
    family: str
    direction: str
    protected_p: np.ndarray
    endpoint: np.ndarray
    projected: np.ndarray
    full_endpoint_control: np.ndarray
    crossing_indices: tuple[int, ...]
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        family, direction = str(self.family), str(self.direction)
        baseline = probability_vector(self.protected_p)
        endpoint = probability_vector(self.endpoint, expected_length=len(baseline))
        projected = probability_vector(self.projected, expected_length=len(baseline))
        full = probability_vector(
            self.full_endpoint_control, expected_length=len(baseline)
        )
        indices = tuple(int(index) for index in self.crossing_indices)
        if (
            family not in ACTION_FAMILIES
            or direction not in DIRECTIONS
            or indices != tuple(sorted(set(indices)))
            or any(index < 0 or index >= len(baseline) for index in indices)
        ):
            raise GovernanceError("SCALE-BP v2 boundary action identity drifted.")
        mask = np.zeros(len(baseline), dtype=bool)
        mask[list(indices)] = True
        if (
            not np.array_equal(projected[~mask], baseline[~mask])
            or not np.array_equal(full[~mask], baseline[~mask])
        ):
            raise GovernanceError("SCALE-BP v2 boundary action changed P off-mask.")
        expected_crossing = (
            (baseline < HARD_THRESHOLD) & (endpoint >= HARD_THRESHOLD)
            if direction == "zero_to_one"
            else (baseline >= HARD_THRESHOLD) & (endpoint < HARD_THRESHOLD)
        )
        if not np.array_equal(mask, expected_crossing):
            raise GovernanceError("SCALE-BP v2 boundary crossing mask drifted.")
        if indices:
            boundary = (
                HIGH_BOUNDARY_FLOAT32
                if direction == "zero_to_one"
                else LOW_BOUNDARY_FLOAT32
            )
            if not np.all(projected[mask] == boundary):
                raise GovernanceError("SCALE-BP v2 action missed the nearest float32 boundary.")
            if not np.array_equal(full[mask], endpoint[mask]):
                raise GovernanceError("SCALE-BP v2 full-endpoint control drifted.")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "protected_p", baseline)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "projected", projected)
        object.__setattr__(self, "full_endpoint_control", full)
        object.__setattr__(self, "crossing_indices", indices)
        object.__setattr__(
            self,
            "action_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_boundary_action_v1",
                    "action_id": f"{family}::{direction}",
                    "protected_p_sha256": array_sha256(baseline),
                    "endpoint_sha256": array_sha256(endpoint),
                    "projected_sha256": array_sha256(projected),
                    "full_endpoint_sha256": array_sha256(full),
                    "crossing_indices": indices,
                    "off_mask_exact_p": True,
                    "boundary_storage": "nearest_binary32_side_of_0.5",
                }
            ),
        )

    @property
    def action_id(self) -> str:
        return f"{self.family}::{self.direction}"

    @property
    def structural_noop(self) -> bool:
        return not self.crossing_indices


def build_boundary_action(
    protected_p: object,
    endpoint: object,
    *,
    family: object,
    direction: object,
) -> BoundaryAction:
    family_id, direction_id = str(family), str(direction)
    if family_id not in ACTION_FAMILIES or direction_id not in DIRECTIONS:
        raise GovernanceError("SCALE-BP v2 boundary action is outside the menu.")
    baseline = probability_vector(protected_p)
    alternative = probability_vector(endpoint, expected_length=len(baseline))
    if direction_id == "zero_to_one":
        crossing = (baseline < HARD_THRESHOLD) & (alternative >= HARD_THRESHOLD)
        boundary = HIGH_BOUNDARY_FLOAT32
    else:
        crossing = (baseline >= HARD_THRESHOLD) & (alternative < HARD_THRESHOLD)
        boundary = LOW_BOUNDARY_FLOAT32
    projected = np.array(baseline, dtype=np.float64, copy=True)
    full = np.array(baseline, dtype=np.float64, copy=True)
    projected[crossing] = boundary
    full[crossing] = alternative[crossing]
    return BoundaryAction(
        family_id,
        direction_id,
        baseline,
        alternative,
        projected,
        full,
        tuple(int(index) for index in np.flatnonzero(crossing)),
    )


__all__ = (
    "BoundaryAction",
    "HIGH_BOUNDARY_FLOAT32",
    "LOW_BOUNDARY_FLOAT32",
    "build_boundary_action",
)
