"""Adapter from a frozen utility policy to the residual-top-up primitive."""

from __future__ import annotations

from ...protocol import ProtocolError
from ..residual_topup.actions import build_single_source_tail_action
from ..residual_topup.contracts import (
    TARGET_BASE_PER_SOURCE,
    TARGET_TOPUP_TOTAL_PER_CLASS,
    ResidualTopupAction,
    TopupGeometry,
    target_topup_geometry,
)
from .policy_contracts import (
    BASE_ACTION_ID,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    UtilityAlignedPolicy,
)
from .row_contracts import TARGET_CANDIDATE_COUNT


def build_utility_aligned_action(
    policy: UtilityAlignedPolicy,
    *,
    geometry: TopupGeometry,
) -> ResidualTopupAction | None:
    """Adapt an eligible frozen policy to the neutral single-tail primitive.

    ``None`` is deliberate: it denotes exact B and prevents callers from
    accidentally materializing a residual action during abstention.
    """

    if not isinstance(policy, UtilityAlignedPolicy) or not isinstance(
        geometry, TopupGeometry
    ):
        raise ProtocolError("Utility-aligned action adapter received invalid contracts.")
    expected_geometry = target_topup_geometry(policy.candidate_sources)
    if (
        geometry != expected_geometry
        or geometry.source_count != TARGET_CANDIDATE_COUNT
        or geometry.base_per_source != TARGET_BASE_PER_SOURCE
        or geometry.topup_total_per_class != TARGET_TOPUP_TOTAL_PER_CLASS
    ):
        raise ProtocolError(
            "Utility-aligned action requires canonical 8x128 base plus 128 tail geometry."
        )
    if policy.action_id == BASE_ACTION_ID:
        if not policy.used_exact_base_fallback or policy.selected_source is not None:
            raise ProtocolError("Exact-base fallback policy is internally inconsistent.")
        return None
    if policy.action_id not in {
        GLOBAL_ACTION_ID,
        ROUTED_ACTION_ID,
        PERMUTATION_ACTION_ID,
    }:
        raise ProtocolError("Utility-aligned policy action ID is unsupported.")
    if (
        policy.used_exact_base_fallback
        or policy.selected_source is None
        or policy.selected_source not in geometry.source_order
    ):
        raise ProtocolError("Active utility-aligned policy has no legal selected source.")
    return build_single_source_tail_action(
        policy.selected_source,
        geometry=geometry,
    )


__all__ = ("build_utility_aligned_action",)
