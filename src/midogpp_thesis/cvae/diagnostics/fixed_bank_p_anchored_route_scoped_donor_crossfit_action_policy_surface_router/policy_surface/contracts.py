"""Immutable descriptor and response contracts for the P-DCAPS policy surface.

The policy descriptor is sealed without realized utility.  Pseudo-center
responses are attached later and retain the byte-identical descriptor hashes;
this makes opening a response before the complete prefix surface is sealed
detectable rather than a convention left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import FavorableUtility, RouteKey
from ..identity import ACTION_STRATA, TIE_TOLERANCE, canonical_hash, require_sha256


@dataclass(frozen=True)
class PolicyAction:
    """One already calibrated, response-blind action entering policy ranking."""

    route_key: RouteKey
    case_id: str
    action_hash: str
    family: str
    direction: str
    predicted_utility: FavorableUtility
    action_calibration_hash: str
    action_descriptor_hash: str = field(init=False)

    def __post_init__(self) -> None:
        case_id = str(self.case_id)
        action_hash = require_sha256(self.action_hash, "policy action hash")
        calibration_hash = require_sha256(
            self.action_calibration_hash, "action calibration hash"
        )
        if (
            not case_id
            or case_id != self.route_key.held_case_id
            or (str(self.family), str(self.direction)) not in ACTION_STRATA
        ):
            raise ProtocolError("P-DCAPS policy action identity drifted.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "action_hash", action_hash)
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "direction", str(self.direction))
        object.__setattr__(self, "action_calibration_hash", calibration_hash)
        object.__setattr__(
            self,
            "action_descriptor_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_action_v1",
                    "route_key": self.route_key.to_payload(),
                    "case_id": case_id,
                    "action_hash": action_hash,
                    "family": self.family,
                    "direction": self.direction,
                    "predicted_utility": self.predicted_utility.to_payload(),
                    "action_calibration_hash": calibration_hash,
                    "realized_utility_available": False,
                }
            ),
        )

    @property
    def stratum(self) -> tuple[str, str]:
        return self.family, self.direction

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_action_v1",
            "route_key": self.route_key.to_payload(),
            "case_id": self.case_id,
            "action_hash": self.action_hash,
            "family": self.family,
            "direction": self.direction,
            "predicted_utility": self.predicted_utility.to_payload(),
            "action_calibration_hash": self.action_calibration_hash,
            "realized_utility_available": False,
            "action_descriptor_hash": self.action_descriptor_hash,
        }


@dataclass(frozen=True)
class PolicySurfaceProvenance:
    """H/J exclusions and action-lineage hashes shared by a whole surface."""

    surface_role: str
    outer_center: str
    route_center: str
    excluded_outer_center: str
    excluded_scored_center: str | None
    action_surface_seal_hash: str
    action_exclusion_hashes: tuple[str, ...]
    action_fit_scope_hashes: tuple[str, ...]
    provenance_hash: str = field(init=False)

    def __post_init__(self) -> None:
        role = str(self.surface_role)
        outer = str(self.outer_center)
        route = str(self.route_center)
        excluded_scored = (
            None
            if self.excluded_scored_center is None
            else str(self.excluded_scored_center)
        )
        if (
            role not in {"target", "pseudo"}
            or outer not in CENTERS
            or route not in CENTERS
            or str(self.excluded_outer_center) != outer
            or (role == "target" and (route != outer or excluded_scored is not None))
            or (role == "pseudo" and (route == outer or excluded_scored != route))
        ):
            raise ProtocolError("P-DCAPS policy H/J provenance drifted.")
        seal_hash = require_sha256(
            self.action_surface_seal_hash, "action-surface seal hash"
        )
        exclusion_hashes = tuple(str(value) for value in self.action_exclusion_hashes)
        fit_hashes = tuple(str(value) for value in self.action_fit_scope_hashes)
        if len(exclusion_hashes) != len(fit_hashes):
            raise ProtocolError("P-DCAPS policy action-lineage cardinality drifted.")
        for value in exclusion_hashes:
            require_sha256(value, "action exclusion hash")
        for value in fit_hashes:
            require_sha256(value, "action fit-scope hash")
        object.__setattr__(self, "surface_role", role)
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "route_center", route)
        object.__setattr__(self, "excluded_outer_center", outer)
        object.__setattr__(self, "excluded_scored_center", excluded_scored)
        object.__setattr__(self, "action_surface_seal_hash", seal_hash)
        object.__setattr__(self, "action_exclusion_hashes", exclusion_hashes)
        object.__setattr__(self, "action_fit_scope_hashes", fit_hashes)
        object.__setattr__(
            self,
            "provenance_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_surface_provenance_v1",
                    "surface_role": role,
                    "outer_center": outer,
                    "route_center": route,
                    "excluded_outer_center": outer,
                    "excluded_scored_center": excluded_scored,
                    "action_surface_seal_hash": seal_hash,
                    "action_exclusion_hashes": exclusion_hashes,
                    "action_fit_scope_hashes": fit_hashes,
                }
            ),
        )

    @property
    def excluded_centers(self) -> tuple[str, ...]:
        values = (self.outer_center, self.excluded_scored_center)
        return tuple(sorted(value for value in values if value is not None))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_surface_provenance_v1",
            "surface_role": self.surface_role,
            "outer_center": self.outer_center,
            "route_center": self.route_center,
            "excluded_outer_center": self.excluded_outer_center,
            "excluded_scored_center": self.excluded_scored_center,
            "excluded_centers": list(self.excluded_centers),
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "action_exclusion_hashes": list(self.action_exclusion_hashes),
            "action_fit_scope_hashes": list(self.action_fit_scope_hashes),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class PrefixCell:
    """One exact-P or nonzero prefix cell on a sealed policy surface."""

    provenance: PolicySurfaceProvenance
    k: int
    total_candidate_count: int
    ordered_action_hashes: tuple[str, ...]
    predicted_utility: FavorableUtility
    normalized_depth: float
    max_positive_candidate_share: float
    stratum_proportions: tuple[float, ...]
    realized_utility: FavorableUtility | None = None
    cell_hash: str = field(init=False)
    response_hash: str | None = field(init=False)

    def __post_init__(self) -> None:
        hashes = tuple(str(value) for value in self.ordered_action_hashes)
        proportions = tuple(float(value) for value in self.stratum_proportions)
        for value in hashes:
            require_sha256(value, "prefix action hash")
        values = np.asarray(
            (self.normalized_depth, self.max_positive_candidate_share, *proportions),
            dtype=np.float64,
        )
        expected_depth = 0.0 if self.total_candidate_count == 0 else self.k / self.total_candidate_count
        if (
            self.k < 0
            or self.total_candidate_count < 0
            or self.k > self.total_candidate_count
            or self.k != len(hashes)
            or len(set(hashes)) != len(hashes)
            or len(proportions) != len(ACTION_STRATA)
            or not np.isfinite(values).all()
            or abs(float(self.normalized_depth) - expected_depth) > TIE_TOLERANCE
            or not 0.0 <= float(self.max_positive_candidate_share) <= 1.0
            or any(value < 0.0 or value > 1.0 for value in proportions)
            or (
                self.k == 0
                and (
                    hashes
                    or self.predicted_utility != FavorableUtility.zeros()
                    or abs(sum(proportions)) > TIE_TOLERANCE
                    or abs(self.max_positive_candidate_share) > TIE_TOLERANCE
                    or (
                        self.realized_utility is not None
                        and self.realized_utility != FavorableUtility.zeros()
                    )
                )
            )
            or (
                self.k > 0
                and abs(sum(proportions) - 1.0) > TIE_TOLERANCE
            )
        ):
            raise ProtocolError("P-DCAPS prefix descriptor geometry drifted.")
        descriptor = {
            "schema_version": "pdcaps_prefix_cell_v1",
            "provenance_hash": self.provenance.provenance_hash,
            "k": int(self.k),
            "total_candidate_count": int(self.total_candidate_count),
            "ordered_action_hashes": hashes,
            "predicted_utility": self.predicted_utility.to_payload(),
            "normalized_depth": float(self.normalized_depth),
            "max_positive_candidate_share": float(
                self.max_positive_candidate_share
            ),
            "stratum_proportions": proportions,
            "realized_utility_available": False,
        }
        cell_hash = canonical_hash(descriptor)
        response_hash = (
            None
            if self.realized_utility is None
            else canonical_hash(
                {
                    "schema_version": "pdcaps_prefix_response_v1",
                    "cell_hash": cell_hash,
                    "realized_utility": self.realized_utility.to_payload(),
                }
            )
        )
        object.__setattr__(self, "ordered_action_hashes", hashes)
        object.__setattr__(self, "normalized_depth", float(self.normalized_depth))
        object.__setattr__(
            self,
            "max_positive_candidate_share",
            float(self.max_positive_candidate_share),
        )
        object.__setattr__(self, "stratum_proportions", proportions)
        object.__setattr__(self, "cell_hash", cell_hash)
        object.__setattr__(self, "response_hash", response_hash)

    @property
    def response_available(self) -> bool:
        return self.realized_utility is not None

    def with_realized_utility(self, value: FavorableUtility) -> "PrefixCell":
        if self.realized_utility is not None:
            raise ProtocolError("P-DCAPS prefix response was opened twice.")
        return replace(self, realized_utility=value)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_prefix_cell_v1",
            "provenance": self.provenance.to_payload(),
            "k": self.k,
            "total_candidate_count": self.total_candidate_count,
            "ordered_action_hashes": list(self.ordered_action_hashes),
            "predicted_utility": self.predicted_utility.to_payload(),
            "normalized_depth": self.normalized_depth,
            "max_positive_candidate_share": self.max_positive_candidate_share,
            "stratum_proportions": list(self.stratum_proportions),
            "realized_utility": (
                None
                if self.realized_utility is None
                else self.realized_utility.to_payload()
            ),
            "cell_hash": self.cell_hash,
            "response_hash": self.response_hash,
        }


@dataclass(frozen=True)
class PrefixSurface:
    provenance: PolicySurfaceProvenance
    ranked_actions: tuple[PolicyAction, ...]
    cells: tuple[PrefixCell, ...]
    surface_hash: str = field(init=False)
    response_surface_hash: str | None = field(init=False)

    def __post_init__(self) -> None:
        actions = tuple(self.ranked_actions)
        cells = tuple(self.cells)
        response_flags = tuple(cell.response_available for cell in cells)
        expected_prefixes = tuple(
            tuple(row.action_hash for row in actions[:k])
            for k in range(len(actions) + 1)
        )
        if (
            len({row.case_id for row in actions}) != len(actions)
            or len({row.action_hash for row in actions}) != len(actions)
            or tuple(cell.k for cell in cells) != tuple(range(len(actions) + 1))
            or tuple(cell.ordered_action_hashes for cell in cells) != expected_prefixes
            or any(cell.provenance != self.provenance for cell in cells)
            or any(cell.total_candidate_count != len(actions) for cell in cells)
            or self.provenance.action_exclusion_hashes
            != tuple(row.route_key.exclusion_hash for row in actions)
            or self.provenance.action_fit_scope_hashes
            != tuple(row.route_key.fit_scope_hash for row in actions)
            or (response_flags and any(response_flags) and not all(response_flags))
        ):
            raise ProtocolError("P-DCAPS prefix surface is incomplete or reordered.")
        descriptor_hash = canonical_hash(
            {
                "schema_version": "pdcaps_prefix_surface_v1",
                "provenance_hash": self.provenance.provenance_hash,
                "ranked_action_descriptor_hashes": tuple(
                    row.action_descriptor_hash for row in actions
                ),
                "cell_hashes": tuple(cell.cell_hash for cell in cells),
                "response_surface_available": False,
            }
        )
        response_hash = (
            canonical_hash(
                {
                    "schema_version": "pdcaps_prefix_response_surface_v1",
                    "surface_hash": descriptor_hash,
                    "response_hashes": tuple(cell.response_hash for cell in cells),
                }
            )
            if response_flags and all(response_flags)
            else None
        )
        object.__setattr__(self, "ranked_actions", actions)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "surface_hash", descriptor_hash)
        object.__setattr__(self, "response_surface_hash", response_hash)

    @property
    def responses_available(self) -> bool:
        return self.response_surface_hash is not None

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_prefix_surface_v1",
            "provenance": self.provenance.to_payload(),
            "ranked_actions": [row.to_payload() for row in self.ranked_actions],
            "cells": [row.to_payload() for row in self.cells],
            "surface_hash": self.surface_hash,
            "response_surface_hash": self.response_surface_hash,
        }


@dataclass(frozen=True)
class PolicyObservation:
    """A response-bearing prefix cell with its center/route weight groups."""

    cell: PrefixCell
    route_hash: str
    observation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route_hash = require_sha256(self.route_hash, "policy route hash")
        if self.cell.realized_utility is None or self.cell.response_hash is None:
            raise ProtocolError("P-DCAPS policy observation lacks a sealed response.")
        object.__setattr__(self, "route_hash", route_hash)
        object.__setattr__(
            self,
            "observation_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_observation_v1",
                    "cell_hash": self.cell.cell_hash,
                    "response_hash": self.cell.response_hash,
                    "route_hash": route_hash,
                    "center": self.center,
                }
            ),
        )

    @property
    def center(self) -> str:
        return self.cell.provenance.route_center

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_observation_v1",
            "cell": self.cell.to_payload(),
            "route_hash": self.route_hash,
            "center": self.center,
            "observation_hash": self.observation_hash,
        }


def favorable_utility_from_payload(payload: Mapping[str, object]) -> FavorableUtility:
    return FavorableUtility(
        float(payload["bacc_gain"]),
        float(payload["brier_gain"]),
        float(payload["log_gain"]),
    )


def require_complete_responses(surfaces: Sequence[PrefixSurface]) -> None:
    if not surfaces or any(not surface.responses_available for surface in surfaces):
        raise ProtocolError("P-DCAPS policy fitting requires complete sealed responses.")


__all__ = (
    "PolicyAction",
    "PolicyObservation",
    "PolicySurfaceProvenance",
    "PrefixCell",
    "PrefixSurface",
    "favorable_utility_from_payload",
    "require_complete_responses",
)
