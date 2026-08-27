"""Closed-world replay validation for SCALE-BP pseudo policies."""

from __future__ import annotations

import math
from typing import Sequence

from .controls import METHOD_IDS, P_PROTECTED, SCALE_BP_PRIMARY
from .identity import ACTION_IDS
from .protocol import ProtocolError
from .pseudo_evidence import PseudoRouteActionEvidence, PseudoRoutePolicyEvidence


ActionContext = tuple[str, str, str, str, str]
ReplayContext = tuple[str, str, str, str]


def group_actions(
    rows: Sequence[PseudoRouteActionEvidence],
) -> dict[ActionContext, tuple[PseudoRouteActionEvidence, ...]]:
    grouped: dict[ActionContext, list[PseudoRouteActionEvidence]] = {}
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, PseudoRouteActionEvidence):
            raise ProtocolError("SCALE-BP pseudo action evidence type drifted.")
        context = (
            row.outer_center,
            row.pseudo_center,
            row.case_id,
            row.scope.scope_hash,
            row.method_id,
        )
        identity = (*context, row.action_id)
        if identity in seen:
            raise ProtocolError("SCALE-BP pseudo action rectangle is duplicated.")
        seen.add(identity)
        grouped.setdefault(context, []).append(row)
    output = {
        key: tuple(sorted(values, key=lambda row: ACTION_IDS.index(row.action_id)))
        for key, values in grouped.items()
    }
    if any(
        tuple(row.action_id for row in values) != ACTION_IDS
        for values in output.values()
    ):
        raise ProtocolError("SCALE-BP pseudo action menu is incomplete.")
    return output


def group_policies(
    rows: Sequence[PseudoRoutePolicyEvidence],
) -> dict[ActionContext, PseudoRoutePolicyEvidence]:
    output: dict[ActionContext, PseudoRoutePolicyEvidence] = {}
    for row in rows:
        if not isinstance(row, PseudoRoutePolicyEvidence):
            raise ProtocolError("SCALE-BP pseudo policy evidence type drifted.")
        key = (
            row.outer_center,
            row.pseudo_center,
            row.case_id,
            row.scope.scope_hash,
            row.method_id,
        )
        if key in output:
            raise ProtocolError("SCALE-BP pseudo policy rectangle is duplicated.")
        output[key] = row
    return output


def validate_context_rectangle(
    actions: dict[str, tuple[PseudoRouteActionEvidence, ...]],
    policies: dict[str, PseudoRoutePolicyEvidence],
) -> None:
    """Validate one exact method/action rectangle and policy composition."""

    if tuple(actions) != METHOD_IDS or tuple(policies) != METHOD_IDS:
        raise ProtocolError("SCALE-BP frozen method menu is incomplete.")
    primary = actions[SCALE_BP_PRIMARY]
    opportunity = tuple(row.opportunity for row in primary)
    crossing = tuple(row.crossing_indices for row in primary)
    descriptor_hashes = tuple(row.descriptor_hash for row in primary)
    oracle_values = {policy.oracle_bacc_gain for policy in policies.values()}
    request_hashes = {
        row.replay_request_hash for rows in actions.values() for row in rows
    } | {row.replay_request_hash for row in policies.values()}
    label_hashes = {
        row.terminal_label_hash for rows in actions.values() for row in rows
    } | {row.terminal_label_hash for row in policies.values()}
    menu_hashes = {
        row.method_menu_hash for rows in actions.values() for row in rows
    } | {row.method_menu_hash for row in policies.values()}
    oracle_hashes = {
        row.oracle_hash for rows in actions.values() for row in rows
    } | {row.oracle_hash for row in policies.values()}
    if (
        len(oracle_values) != 1
        or len(request_hashes) != 1
        or len(label_hashes) != 1
        or len(menu_hashes) != 1
        or len(oracle_hashes) != 1
    ):
        raise ProtocolError("SCALE-BP pseudo policy oracle lineage drifted.")
    for method in METHOD_IDS:
        method_rows = actions[method]
        policy = policies[method]
        if (
            tuple(row.opportunity for row in method_rows) != opportunity
            or tuple(row.crossing_indices for row in method_rows) != crossing
            or tuple(row.descriptor_hash for row in method_rows) != descriptor_hashes
            or tuple(sorted(row.evidence_hash for row in method_rows))
            != policy.action_evidence_hashes
            or tuple(sorted(row.action_id for row in method_rows if row.selected))
            != policy.selected_action_ids
        ):
            raise ProtocolError("SCALE-BP pseudo policy/action replay lineage drifted.")
        selected = tuple(row for row in method_rows if row.selected)
        selected_indices = [set(row.crossing_indices) for row in selected]
        if len(selected_indices) == 2 and selected_indices[0].intersection(
            selected_indices[1]
        ):
            raise ProtocolError("SCALE-BP pseudo pair actions overlap.")
        expected = tuple(
            math.fsum(getattr(row, field_name) for row in selected)
            for field_name in (
                "realized_bacc_gain",
                "realized_brier_loss_delta",
                "realized_log_loss_delta",
            )
        )
        actual = (
            policy.realized_bacc_gain,
            policy.realized_brier_loss_delta,
            policy.realized_log_loss_delta,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(expected, actual, strict=True)
        ):
            raise ProtocolError("SCALE-BP pseudo policy metric replay drifted.")
    p_policy = policies[P_PROTECTED]
    if p_policy.selected_action_ids or any(
        value != 0.0
        for value in (
            p_policy.realized_bacc_gain,
            p_policy.realized_brier_loss_delta,
            p_policy.realized_log_loss_delta,
        )
    ):
        raise ProtocolError("SCALE-BP protected P policy is not exact P.")
    if not any(opportunity) and any(
        policy.selected_action_ids for policy in policies.values()
    ):
        raise ProtocolError("SCALE-BP no-opportunity pseudo case routed an action.")


__all__ = (
    "ActionContext",
    "ReplayContext",
    "group_actions",
    "group_policies",
    "validate_context_rectangle",
)
