"""Deterministic, pseudo-only legacy center-pooled prefix selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..engine import OuterActionPolicyResult
from ..identity import TIE_TOLERANCE
from ..policy_surface import PrefixCell, PrefixSurface
from .contracts import (
    LegacyControlDecision,
    LegacyControlSurface,
    LegacyTargetPolicyDecision,
)


def build_legacy_control_decision(
    surface: PrefixSurface,
    selected_k: int,
) -> LegacyControlDecision:
    """Bind an actual legacy choice to one complete current pseudo surface."""

    if (
        surface.provenance.surface_role != "pseudo"
        or not surface.responses_available
        or surface.response_surface_hash is None
        or not isinstance(selected_k, int)
        or isinstance(selected_k, bool)
        or not 0 <= selected_k < len(surface.cells)
    ):
        raise ProtocolError(
            "P-DCAPS legacy control requires a complete pseudo surface."
        )
    cell = surface.cells[selected_k]
    if cell.realized_utility is None or cell.response_hash is None:
        raise ProtocolError("P-DCAPS legacy selected response is absent.")
    oracle = max(
        row.realized_utility.bacc_gain
        for row in surface.cells
        if row.realized_utility is not None
    )
    realized = cell.realized_utility
    routed = cell.k > 0
    safe = bool(
        routed
        and realized.bacc_gain > 0.0
        and realized.brier_gain >= 0.0
        and realized.log_gain >= 0.0
    )
    return LegacyControlDecision(
        surface.provenance.outer_center,
        surface.provenance.route_center,
        surface.surface_hash,
        surface.response_surface_hash,
        cell.k,
        cell.cell_hash,
        cell.response_hash,
        cell.ordered_action_hashes,
        realized,
        routed,
        safe,
        float(oracle),
        abs(float(oracle) - realized.bacc_gain),
    )


def _nearest_prefix_k(surface: PrefixSurface, depth: float) -> int:
    return min(
        surface.cells,
        key=lambda row: (
            abs(row.normalized_depth - depth),
            row.k,
            row.cell_hash,
        ),
    ).k


def _mean_realized_utility(
    decisions: Sequence[LegacyControlDecision],
) -> FavorableUtility:
    values = np.asarray(
        [row.realized_utility.as_tuple() for row in decisions],
        dtype=np.float64,
    )
    means = np.mean(values, axis=0, dtype=np.float64)
    return FavorableUtility(*(float(value) for value in means))


def _center_pooled_target_choice(
    result: OuterActionPolicyResult,
) -> tuple[LegacyTargetPolicyDecision, tuple[LegacyControlDecision, ...]]:
    """Select a shared normalized depth by equal-center pseudo utility."""

    target = result.target_policy_surface
    pseudo_surfaces = tuple(result.pseudo_policy_response_surfaces)
    candidates: list[
        tuple[
            PrefixCell,
            tuple[LegacyControlDecision, ...],
            FavorableUtility,
        ]
    ] = []
    for target_cell in target.cells:
        decisions = tuple(
            build_legacy_control_decision(
                surface,
                _nearest_prefix_k(surface, target_cell.normalized_depth),
            )
            for surface in pseudo_surfaces
        )
        pooled = _mean_realized_utility(decisions)
        candidates.append((target_cell, decisions, pooled))
    feasible = tuple(
        row
        for row in candidates
        if row[0].k == 0
        or (
            row[2].bacc_gain > 0.0
            and row[2].brier_gain >= 0.0
            and row[2].log_gain >= 0.0
        )
    )
    maximum = max(row[2].bacc_gain for row in feasible)
    tied = tuple(
        row
        for row in feasible
        if abs(row[2].bacc_gain - maximum) <= TIE_TOLERANCE
    )
    selected_cell, decisions, pooled = min(
        tied,
        key=lambda row: (
            row[0].k,
            row[0].cell_hash,
        ),
    )
    target_decision = LegacyTargetPolicyDecision(
        result.outer_center,
        target.surface_hash,
        selected_cell.k,
        selected_cell.cell_hash,
        selected_cell.ordered_action_hashes,
        selected_cell.normalized_depth,
        pooled,
        tuple(row.decision_hash for row in decisions),
        bool(selected_cell.k > 0),
        (
            "LEGACY_CENTER_POOLED_PREFIX_SELECTED"
            if selected_cell.k > 0
            else "EXACT_P_CENTER_POOLED_PREFIX_K0"
        ),
    )
    return target_decision, decisions


def build_legacy_control_surface(
    result: OuterActionPolicyResult,
) -> LegacyControlSurface:
    """Build the deterministic center-pooled control from current responses."""

    donors = tuple(center for center in CENTERS if center != result.outer_center)
    surfaces = tuple(result.pseudo_policy_response_surfaces)
    if tuple(row.provenance.route_center for row in surfaces) != donors:
        raise ProtocolError("P-DCAPS legacy control response inventory drifted.")
    target_decision, decisions = _center_pooled_target_choice(result)
    return LegacyControlSurface(
        result.outer_center,
        result.result_hash,
        result.physical_surface_hash,
        result.action_surface_seal_hash,
        tuple(
            (surface.provenance.route_center, str(surface.response_surface_hash))
            for surface in surfaces
        ),
        decisions,
        target_decision,
    )


__all__ = (
    "build_legacy_control_decision",
    "build_legacy_control_surface",
)
