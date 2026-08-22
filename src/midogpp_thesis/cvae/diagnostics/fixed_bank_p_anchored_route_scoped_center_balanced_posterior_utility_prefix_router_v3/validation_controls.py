"""Exact reconstruction of the two fixed identity-fingerprint controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .constants import (
    CANDIDATE_ONLY_METHOD_ID,
    CENTERS,
    OBSERVED_MAX_CONTROL_METHOD_ID,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
)
from .controls import (
    ControlPolicy,
    candidate_only_control,
    observed_maximum_prefix_control,
)
from .donor_replay_runtime import DonorReplayResult
from .hashing import canonical_hash
from .posterior_expected_utility import FavorableUtility
from .validation_candidates import CandidateTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, mapping_field


def reconstruct_control_policies(
    *,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    decisions: Mapping[tuple[str, str], Row],
    donor_case_replay_rows: Sequence[Row],
) -> dict[tuple[str, str], ControlPolicy]:
    replays = []
    for payload in donor_case_replay_rows:
        if payload.get("record_type") != "donor_case_replay":
            fail("control donor replay record type")
        try:
            replays.append(DonorReplayResult.from_payload(payload).replay)
        except (KeyError, TypeError, ValueError) as exc:
            raise _malformed_control() from exc

    result: dict[tuple[str, str], ControlPolicy] = {}
    for center in CENTERS:
        records = tuple(
            candidate_topology.selected_action_by_runtime[
                (center, center, case, PRIMARY_FINGERPRINT_CONTROL_ID)
            ]
            for case in plan_topology.cases_by_center[center]
        )
        actions = tuple(record.action for record in records if record is not None)
        structural = mapping_field(
            decisions[(center, PRIMARY_METHOD_ID)], "structural_transport"
        )
        passed = structural.get("passed") is True
        gate_hash = str(structural.get("gate_hash"))
        candidate_only = (
            candidate_only_control(actions)
            if passed
            else ControlPolicy(
                CANDIDATE_ONLY_METHOD_ID,
                (),
                FavorableUtility.zeros(),
                False,
                (gate_hash,),
            )
        )
        result[(CANDIDATE_ONLY_METHOD_ID, center)] = candidate_only

        center_replays = tuple(
            row
            for row in replays
            if row.outer_center == center
            and row.control_id == PRIMARY_FINGERPRINT_CONTROL_ID
        )
        if passed and center_replays:
            _selection, observed = observed_maximum_prefix_control(
                actions, center_replays
            )
        else:
            source_hash = canonical_hash(
                [
                    "CBPUPR_OBSERVED_MAX_UNSUPPORTED_OR_STRUCTURAL_FAILURE",
                    center,
                    gate_hash,
                ]
            )
            observed = ControlPolicy(
                OBSERVED_MAX_CONTROL_METHOD_ID,
                (),
                FavorableUtility.zeros(),
                False,
                (source_hash,),
            )
        result[(OBSERVED_MAX_CONTROL_METHOD_ID, center)] = observed
    return result


def _malformed_control() -> Exception:
    from ...protocol import ProtocolError

    return ProtocolError("CBPUPR persisted control donor replay is malformed.")


__all__ = ("reconstruct_control_policies",)
