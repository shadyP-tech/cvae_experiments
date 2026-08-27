"""Whole-case cyclic support-label permutation negative control."""

from __future__ import annotations

from dataclasses import replace

from ..engine import CaseRouteRequest
from ..hashing import canonical_hash
from ..identity import ACTION_IDS
from ..local_residual.contracts import LocalResidualRecord
from ..protocol import ProtocolError
from ..replay_scope import PseudoReplayScope


def build_support_label_permutation(
    request: CaseRouteRequest,
) -> tuple[CaseRouteRequest, str]:
    """Rotate whole-case outcomes once while preserving descriptors and lineage."""

    if not isinstance(request.route_scope, PseudoReplayScope):
        raise ProtocolError("SCALE-BP support permutation requires a pseudo scope.")
    rows = tuple(request.support_records)
    member_ids = tuple(member.member_id for member in request.support_plan.members)
    by_member: dict[str, dict[str, LocalResidualRecord]] = {
        member_id: {} for member_id in member_ids
    }
    for row in rows:
        if row.member_id not in by_member or row.action_id in by_member[row.member_id]:
            raise ProtocolError("SCALE-BP support permutation population drifted.")
        by_member[row.member_id][row.action_id] = row
    action_menus = {tuple(sorted(values)) for values in by_member.values()}
    if rows and (
        len(action_menus) != 1
        or any(action not in ACTION_IDS for action in next(iter(action_menus)))
    ):
        raise ProtocolError("SCALE-BP support permutation action menu drifted.")
    source_for_destination = {
        member_id: member_ids[(index + 1) % len(member_ids)]
        for index, member_id in enumerate(member_ids)
    }
    permuted: list[LocalResidualRecord] = []
    for row in rows:
        source = by_member[source_for_destination[row.member_id]][row.action_id]
        permuted.append(replace(row, realized_metrics=source.realized_metrics))
    ordered = tuple(
        sorted(permuted, key=lambda row: (row.member_id, row.action_id, row.record_hash))
    )
    permutation_hash = canonical_hash(
        {
            "schema_version": "scale_bp_whole_case_support_permutation_v1",
            "route_request_hash": request.request_hash,
            "scope_hash": request.route_scope.scope_hash,
            "source_for_destination": tuple(sorted(source_for_destination.items())),
            "original_record_hashes": tuple(sorted(row.record_hash for row in rows)),
            "permuted_record_hashes": tuple(sorted(row.record_hash for row in ordered)),
            "same_mapping_for_every_action": True,
            "held_case_labels_used": False,
        }
    )
    return (
        CaseRouteRequest(
            case_id=request.case_id,
            route_scope=request.route_scope,
            portfolio_probabilities=request.portfolio_probabilities,
            support_plan=request.support_plan,
            support_records=ordered,
            action_inputs=request.action_inputs,
        ),
        permutation_hash,
    )


__all__ = ("build_support_label_permutation",)
