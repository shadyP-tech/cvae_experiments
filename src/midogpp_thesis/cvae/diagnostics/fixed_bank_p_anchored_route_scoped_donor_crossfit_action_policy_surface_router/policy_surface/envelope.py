"""Selection-aware descriptive OOF overprediction envelope.

This module deliberately does not call the correction a confidence bound and
makes no coverage claim.  It is an elementwise donor-center stress envelope
built only from nested out-of-fold policy responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import canonical_hash


@dataclass(frozen=True)
class PolicyOOFResidual:
    outer_center: str
    scored_center: str
    route_hash: str
    cell_hash: str
    predicted_utility: FavorableUtility
    realized_utility: FavorableUtility
    calibration_hash: str
    calibration_excluded_centers: tuple[str, ...]
    residual_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        scored = str(self.scored_center)
        excluded = tuple(sorted({str(value) for value in self.calibration_excluded_centers}))
        if (
            outer == scored
            or outer not in excluded
            or scored not in excluded
            or len(excluded) not in {2, 3}
        ):
            raise ProtocolError("P-DCAPS OOF residual did not exclude H/J.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "scored_center", scored)
        object.__setattr__(self, "calibration_excluded_centers", excluded)
        object.__setattr__(
            self,
            "residual_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_oof_residual_v1",
                    "outer_center": outer,
                    "scored_center": scored,
                    "route_hash": self.route_hash,
                    "cell_hash": self.cell_hash,
                    "predicted_utility": self.predicted_utility.to_payload(),
                    "realized_utility": self.realized_utility.to_payload(),
                    "calibration_hash": self.calibration_hash,
                    "calibration_excluded_centers": excluded,
                }
            ),
        )

    @property
    def positive_overprediction(self) -> FavorableUtility:
        delta = np.asarray(self.predicted_utility.as_tuple(), dtype=np.float64) - np.asarray(
            self.realized_utility.as_tuple(), dtype=np.float64
        )
        return FavorableUtility.from_array(np.maximum(delta, 0.0))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_oof_residual_v1",
            "outer_center": self.outer_center,
            "scored_center": self.scored_center,
            "route_hash": self.route_hash,
            "cell_hash": self.cell_hash,
            "predicted_utility": self.predicted_utility.to_payload(),
            "realized_utility": self.realized_utility.to_payload(),
            "positive_overprediction": self.positive_overprediction.to_payload(),
            "calibration_hash": self.calibration_hash,
            "calibration_excluded_centers": list(self.calibration_excluded_centers),
            "residual_hash": self.residual_hash,
        }


@dataclass(frozen=True)
class PolicyEnvelope:
    outer_center: str
    center_means: tuple[tuple[str, FavorableUtility], ...]
    full_equal_center_mean: FavorableUtility
    leave_one_center_means: tuple[tuple[str, FavorableUtility], ...]
    correction: FavorableUtility
    residual_hashes: tuple[str, ...]
    excluded_scored_center: str | None = None
    envelope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(str(center) for center, _ in self.center_means)
        omitted = tuple(str(center) for center, _ in self.leave_one_center_means)
        context = (
            None
            if self.excluded_scored_center is None
            else str(self.excluded_scored_center)
        )
        if (
            len(centers) < 2
            or len(set(centers)) != len(centers)
            or centers != tuple(sorted(centers))
            or omitted != centers
            or not self.residual_hashes
            or len(set(self.residual_hashes)) != len(self.residual_hashes)
            or context in centers
        ):
            raise ProtocolError("P-DCAPS policy envelope topology drifted.")
        expected = np.max(
            np.asarray(
                (
                    self.full_equal_center_mean.as_tuple(),
                    *(value.as_tuple() for _, value in self.leave_one_center_means),
                ),
                dtype=np.float64,
            ),
            axis=0,
        )
        if not np.allclose(
            expected,
            np.asarray(self.correction.as_tuple(), dtype=np.float64),
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ProtocolError("P-DCAPS policy envelope correction drifted.")
        object.__setattr__(self, "outer_center", str(self.outer_center))
        object.__setattr__(self, "excluded_scored_center", context)
        object.__setattr__(
            self,
            "envelope_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_envelope_v1",
                    "outer_center": self.outer_center,
                    "center_means": tuple(
                        (center, value.to_payload()) for center, value in self.center_means
                    ),
                    "full_equal_center_mean": self.full_equal_center_mean.to_payload(),
                    "leave_one_center_means": tuple(
                        (center, value.to_payload())
                        for center, value in self.leave_one_center_means
                    ),
                    "correction": self.correction.to_payload(),
                    "residual_hashes": self.residual_hashes,
                    "excluded_scored_center": context,
                    "descriptive_lower_envelope_only": True,
                    "finite_sample_coverage_claimed": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_envelope_v1",
            "outer_center": self.outer_center,
            "center_means": [
                [center, value.to_payload()] for center, value in self.center_means
            ],
            "full_equal_center_mean": self.full_equal_center_mean.to_payload(),
            "leave_one_center_means": [
                [center, value.to_payload()]
                for center, value in self.leave_one_center_means
            ],
            "correction": self.correction.to_payload(),
            "residual_hashes": list(self.residual_hashes),
            "excluded_scored_center": self.excluded_scored_center,
            "descriptive_lower_envelope_only": True,
            "finite_sample_coverage_claimed": False,
            "envelope_hash": self.envelope_hash,
        }


def build_policy_envelope(
    residuals: Sequence[PolicyOOFResidual],
    *,
    outer_center: str,
    excluded_scored_center: str | None = None,
) -> PolicyEnvelope:
    rows = tuple(residuals)
    outer = str(outer_center)
    context = (
        None if excluded_scored_center is None else str(excluded_scored_center)
    )
    if (
        not rows
        or len({row.residual_hash for row in rows}) != len(rows)
        or any(row.outer_center != outer for row in rows)
        or any(
            context is not None
            and context not in row.calibration_excluded_centers
            for row in rows
        )
        or any(row.scored_center == context for row in rows)
    ):
        raise ProtocolError("P-DCAPS policy envelope residual lineage drifted.")
    centers = tuple(sorted({row.scored_center for row in rows}))
    if len(centers) < 2:
        raise ProtocolError("P-DCAPS policy envelope needs multiple donor centers.")
    center_means = tuple(
        (center, _center_mean(tuple(row for row in rows if row.scored_center == center)))
        for center in centers
    )
    matrix = np.asarray(
        [value.as_tuple() for _, value in center_means], dtype=np.float64
    )
    full = FavorableUtility.from_array(matrix.mean(axis=0))
    leave_one = tuple(
        (
            center,
            FavorableUtility.from_array(np.delete(matrix, index, axis=0).mean(axis=0)),
        )
        for index, center in enumerate(centers)
    )
    correction = FavorableUtility.from_array(
        np.max(
            np.asarray(
                (full.as_tuple(), *(value.as_tuple() for _, value in leave_one)),
                dtype=np.float64,
            ),
            axis=0,
        )
    )
    return PolicyEnvelope(
        outer,
        center_means,
        full,
        leave_one,
        correction,
        tuple(row.residual_hash for row in rows),
        context,
    )


def apply_policy_envelope(
    predicted_utility: FavorableUtility,
    envelope: PolicyEnvelope,
    *,
    correction_applied_count: int = 0,
) -> tuple[FavorableUtility, int]:
    """Apply the descriptive correction exactly once."""

    if correction_applied_count != 0:
        raise ProtocolError("P-DCAPS policy envelope was applied more than once.")
    return predicted_utility - envelope.correction, 1


def _center_mean(rows: tuple[PolicyOOFResidual, ...]) -> FavorableUtility:
    routes = tuple(sorted({row.route_hash for row in rows}))
    if not rows or not routes:
        raise ProtocolError("P-DCAPS center residual group is empty.")
    route_means = []
    for route_hash in routes:
        route_rows = tuple(row for row in rows if row.route_hash == route_hash)
        values = np.asarray(
            [row.positive_overprediction.as_tuple() for row in route_rows],
            dtype=np.float64,
        )
        route_means.append(values.mean(axis=0))
    return FavorableUtility.from_array(np.asarray(route_means).mean(axis=0))


__all__ = (
    "PolicyEnvelope",
    "PolicyOOFResidual",
    "apply_policy_envelope",
    "build_policy_envelope",
)
