"""Recompute the preterminal funnel and compact information diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .constants import (
    CENTERS,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    PRIMARY_METHOD_ID,
)
from .eligibility import BRIER_UNSAFE, LOG_UNSAFE, NONPOSITIVE_BACC
from .terminal_diagnostics import GateFunnel
from .validation_candidates import CandidateTopology
from .validation_shared import Row, fail, mapping_field, table_rows


def validate_preterminal_diagnostics(
    root: Path,
    *,
    candidate_topology: CandidateTopology,
    decisions: Mapping[tuple[str, str], Row],
) -> GateFunnel:
    primary_keys = tuple(
        key
        for key in candidate_topology.runtime_actions
        if key[0] == key[1] and key[3] == PRIMARY_FINGERPRINT_CONTROL_ID
    )
    routes = tuple(candidate_topology.runtime_actions[key] for key in primary_keys)
    runtime_rows = tuple(
        candidate_topology.targets[(outer, case, control)]
        for outer, _center, case, control in primary_keys
    )
    descriptors = sum(int(row["descriptor_count"]) for row in runtime_rows)
    crossing = sum(len(actions) for actions in routes)
    nonpositive = sum(
        NONPOSITIVE_BACC in record.eligibility.reason_codes
        for actions in routes
        for record in actions
    )
    positive = crossing - nonpositive
    proper_unsafe = sum(
        NONPOSITIVE_BACC not in record.eligibility.reason_codes
        and (
            BRIER_UNSAFE in record.eligibility.reason_codes
            or LOG_UNSAFE in record.eligibility.reason_codes
        )
        for actions in routes
        for record in actions
    )
    proper_safe = positive - proper_unsafe
    selected = sum(
        candidate_topology.selected_action_by_runtime[key] is not None
        for key in primary_keys
    )
    primary_decisions = tuple(decisions[(center, PRIMARY_METHOD_ID)] for center in CENTERS)
    structural = sum(
        mapping_field(row, "structural_transport").get("passed") is True
        for row in primary_decisions
    )
    feasible = sum(
        int(mapping_field(row, "prefix_selection")["selected_k"]) > 0
        for row in primary_decisions
    )
    routed_cases = sum(
        int(mapping_field(row, "prefix_selection")["selected_k"])
        for row in primary_decisions
    )
    exact_p = sum(
        mapping_field(row, "composition").get("exact_p") is True
        for row in primary_decisions
    )
    stages = (
        ("route_count", len(routes)),
        ("descriptor_count", descriptors),
        ("no_crossing_reject", descriptors - crossing),
        ("crossing_descriptor_count", crossing),
        ("nonpositive_bacc_reject", nonpositive),
        ("positive_bacc_descriptor_count", positive),
        ("proper_unsafe_reject", proper_unsafe),
        ("proper_safe_descriptor_count", proper_safe),
        ("selected_case_candidate_count", selected),
        ("structurally_admissible_center_count", structural),
        ("feasible_prefix_center_count", feasible),
        ("routed_case_count", routed_cases),
        ("exact_p_center_count", exact_p),
    )
    expected = GateFunnel(
        len(routes),
        descriptors,
        crossing,
        positive,
        proper_safe,
        selected,
        structural,
        feasible,
        routed_cases,
        exact_p,
        stages,
    )
    funnel_rows = table_rows(root, "gate_funnel")
    if len(funnel_rows) != 1 or dict(funnel_rows[0]) != expected.to_payload():
        fail("recomputed gate funnel")

    all_target_keys = tuple(
        key for key in candidate_topology.runtime_actions if key[0] == key[1]
    )
    all_selected = sum(
        candidate_topology.selected_action_by_runtime[key] is not None
        for key in all_target_keys
    )
    information = {
        "target_route_control_count": len(all_target_keys),
        "selected_case_control_count": all_selected,
        "candidate_selection_rate": all_selected / len(all_target_keys),
        "numeric_transport_is_authorization_gate": False,
        "formal_claim_authorized": False,
    }
    information_rows = table_rows(root, "information_diagnostics")
    if len(information_rows) != 1 or dict(information_rows[0]) != information:
        fail("information diagnostic reconstruction")
    return expected


__all__ = ("validate_preterminal_diagnostics",)
