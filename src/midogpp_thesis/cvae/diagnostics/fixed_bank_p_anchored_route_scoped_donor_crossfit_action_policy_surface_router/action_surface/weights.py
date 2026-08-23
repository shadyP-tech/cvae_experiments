"""Exact equal-center -> equal-route -> equal-action training weights."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..identity import canonical_hash, require_sha256
from .contracts import ActionResponse


@dataclass(frozen=True)
class HierarchicalWeightAudit:
    row_response_hashes: tuple[str, ...]
    row_weights: tuple[float, ...]
    action_count_by_center_route: tuple[tuple[str, str, int], ...]
    route_count_by_center: tuple[tuple[str, int], ...]
    effective_total_by_center: tuple[tuple[str, float], ...]
    effective_total_by_center_route: tuple[tuple[str, str, float], ...]
    total_weight: float
    weight_audit_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = tuple(str(value) for value in self.row_response_hashes)
        weights = tuple(float(value) for value in self.row_weights)
        action_counts = tuple(
            (str(center), str(route_hash), int(count))
            for center, route_hash, count in self.action_count_by_center_route
        )
        route_counts = tuple(
            (str(center), int(count)) for center, count in self.route_count_by_center
        )
        center_totals = tuple(
            (str(center), float(total)) for center, total in self.effective_total_by_center
        )
        route_totals = tuple(
            (str(center), str(route_hash), float(total))
            for center, route_hash, total in self.effective_total_by_center_route
        )
        for digest in hashes:
            require_sha256(digest, "weighted response hash")
        for _center, route_hash, _value in (*action_counts, *route_totals):
            require_sha256(route_hash, "weighted route hash")
        if (
            not hashes
            or len(hashes) != len(set(hashes))
            or len(weights) != len(hashes)
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or any(center not in CENTERS or count <= 0 for center, _route, count in action_counts)
            or any(center not in CENTERS or count <= 0 for center, count in route_counts)
            or any(not math.isfinite(total) or total <= 0.0 for _center, total in center_totals)
            or any(
                not math.isfinite(total) or total <= 0.0
                for _center, _route, total in route_totals
            )
            or not math.isclose(float(self.total_weight), 1.0, abs_tol=1.0e-12)
            or not math.isclose(sum(weights), 1.0, abs_tol=1.0e-12)
        ):
            raise ProtocolError("P-DCAPS hierarchical weight audit drifted.")
        payload = {
            "schema_version": "pdcaps_hierarchical_weight_audit_v1",
            "row_response_hashes": hashes,
            "row_weights": weights,
            "action_count_by_center_route": action_counts,
            "route_count_by_center": route_counts,
            "effective_total_by_center": center_totals,
            "effective_total_by_center_route": route_totals,
            "total_weight": float(self.total_weight),
            "hierarchy": "equal_center_then_route_then_action",
        }
        object.__setattr__(self, "row_response_hashes", hashes)
        object.__setattr__(self, "row_weights", weights)
        object.__setattr__(self, "action_count_by_center_route", action_counts)
        object.__setattr__(self, "route_count_by_center", route_counts)
        object.__setattr__(self, "effective_total_by_center", center_totals)
        object.__setattr__(self, "effective_total_by_center_route", route_totals)
        object.__setattr__(self, "total_weight", float(self.total_weight))
        object.__setattr__(self, "weight_audit_hash", canonical_hash(payload))

    def as_array(self) -> np.ndarray:
        values = np.ascontiguousarray(self.row_weights, dtype=np.float64)
        values.setflags(write=False)
        return values

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_hierarchical_weight_audit_v1",
            "row_response_hashes": list(self.row_response_hashes),
            "row_weights": list(self.row_weights),
            "action_count_by_center_route": [list(row) for row in self.action_count_by_center_route],
            "route_count_by_center": [list(row) for row in self.route_count_by_center],
            "effective_total_by_center": [list(row) for row in self.effective_total_by_center],
            "effective_total_by_center_route": [
                list(row) for row in self.effective_total_by_center_route
            ],
            "total_weight": self.total_weight,
            "hierarchy": "equal_center_then_route_then_action",
            "weight_audit_hash": self.weight_audit_hash,
        }


def build_hierarchical_weights(
    responses: Sequence[ActionResponse],
) -> HierarchicalWeightAudit:
    """Give each represented center, route, and action equal nested mass."""

    rows = tuple(responses)
    if not rows or len({row.response_hash for row in rows}) != len(rows):
        raise ProtocolError("P-DCAPS weighted responses are empty or duplicated.")
    if any(row.key.route_key.surface_role != "pseudo" for row in rows):
        raise ProtocolError("P-DCAPS calibration weights may use pseudo routes only.")
    centers = tuple(center for center in CENTERS if center in {row.key.route_key.route_center for row in rows})
    route_by_row = tuple(row.key.route_key.exclusion_hash for row in rows)
    center_by_row = tuple(row.key.route_key.route_center for row in rows)
    routes_by_center = {
        center: tuple(
            sorted(
                {
                    route_hash
                    for candidate_center, route_hash in zip(center_by_row, route_by_row, strict=True)
                    if candidate_center == center
                }
            )
        )
        for center in centers
    }
    action_counts = Counter(zip(center_by_row, route_by_row, strict=True))
    center_count = len(centers)
    weights = tuple(
        1.0
        / center_count
        / len(routes_by_center[center])
        / action_counts[(center, route_hash)]
        for center, route_hash in zip(center_by_row, route_by_row, strict=True)
    )
    center_totals = tuple(
        (
            center,
            float(
                np.sum(
                    [weight for weight, candidate in zip(weights, center_by_row, strict=True) if candidate == center],
                    dtype=np.float64,
                )
            ),
        )
        for center in centers
    )
    route_totals = tuple(
        (
            center,
            route_hash,
            float(
                np.sum(
                    [
                        weight
                        for weight, candidate_center, candidate_route in zip(
                            weights, center_by_row, route_by_row, strict=True
                        )
                        if candidate_center == center and candidate_route == route_hash
                    ],
                    dtype=np.float64,
                )
            ),
        )
        for center in centers
        for route_hash in routes_by_center[center]
    )
    expected_center_total = 1.0 / center_count
    if any(
        not math.isclose(total, expected_center_total, abs_tol=1.0e-12)
        for _center, total in center_totals
    ):
        raise ProtocolError("P-DCAPS center-equal weights drifted.")
    return HierarchicalWeightAudit(
        tuple(row.response_hash for row in rows),
        weights,
        tuple(
            (center, route_hash, action_counts[(center, route_hash)])
            for center in centers
            for route_hash in routes_by_center[center]
        ),
        tuple((center, len(routes_by_center[center])) for center in centers),
        center_totals,
        route_totals,
        float(np.sum(weights, dtype=np.float64)),
    )


__all__ = ("HierarchicalWeightAudit", "build_hierarchical_weights")
