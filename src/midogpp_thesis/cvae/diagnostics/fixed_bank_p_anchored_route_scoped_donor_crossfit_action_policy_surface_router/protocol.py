"""Frozen scientific protocol for the terminal P-DCAPS diagnostic."""

from __future__ import annotations

from typing import Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .identity import (
    ACTION_STRATA,
    DIRECT_INPUT_ROLES,
    EXPERIMENT_ID,
    METHOD_MENU,
    PUBLICATION_STATUS,
    RIDGE_ALPHA,
    TERMINAL_DECISION,
    TIE_TOLERANCE,
    canonical_hash,
)
from .target_local_runtime import POSTERIOR_CONTROL_IDS


PROTOCOL_SCHEMA = "pdcaps_terminal_protocol_v1"


def frozen_protocol_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "split": "test",
        "split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "execution_authorized": False,
        "authorization_is_separate_from_implementation_request": True,
        "source_code_or_implementation_request_alone_authorizes_execution": False,
        "execution_requires_external_authorized_config_and_ledger": True,
        "input_roles": list(DIRECT_INPUT_ROLES),
        "input_count": len(DIRECT_INPUT_ROLES),
        "physical_probability_surface_recomputed_from_original_inputs": True,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
        "held_unit": "whole_case_or_group",
        "outer_center_excluded_from_every_scientific_fit": True,
        "pseudo_center_excluded_from_own_prediction": True,
        "held_case_d_role": "scored_response_only_after_surface_seal",
        "action_strata": [list(row) for row in ACTION_STRATA],
        "ridge_alpha": RIDGE_ALPHA,
        "hyperparameter_selection_used": False,
        "fit_only_standardization": True,
        "hierarchical_weighting": "equal_center_then_route_then_cell",
        "minimum_reliability_center_count": 6,
        "all_action_surface_sealed_before_pseudo_response_access": True,
        "posterior_control_ids": list(POSTERIOR_CONTROL_IDS),
        "identity_and_cyclic_action_surfaces_jointly_sealed_before_pseudo_response_access": True,
        "pseudo_label_capability_opened_once_per_route_for_all_posterior_controls": True,
        "all_prefix_cells_sealed_before_policy_response_access": True,
        "per_outer_center_admission": True,
        "pooled_admission_can_affect_routes": False,
        "descriptive_lower_envelope_only": True,
        "finite_sample_coverage_claimed": False,
        "method_menu": list(METHOD_MENU),
        "tie_tolerance": TIE_TOLERANCE,
        "exact_p_fallback_required": True,
        "target_labels_open_only_after_preterminal_attestation": True,
        "all_fixed_method_decisions_and_compositions_sealed_before_target_labels": True,
        "raw_labels_may_be_persisted": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "deployment_claimed": False,
    }
    return {**payload, "protocol_hash": canonical_hash(payload)}


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    expected = frozen_protocol_payload()
    if dict(payload) != expected:
        raise ProtocolError("P-DCAPS frozen protocol drifted.")


def validate_nested_exclusions(
    *,
    outer_center: object,
    scored_center: object,
    excluded_centers: Sequence[object],
) -> tuple[str, str]:
    outer = str(outer_center)
    scored = str(scored_center)
    excluded = tuple(sorted({str(value) for value in excluded_centers}))
    if outer not in CENTERS or scored not in CENTERS or outer == scored:
        raise ProtocolError("P-DCAPS nested H/J identity drifted.")
    if set(excluded) != {outer, scored}:
        raise ProtocolError("P-DCAPS nested fit must exclude exactly outer H and scored J.")
    return outer, scored


def validate_held_case_role(
    held_case_id: object,
    *,
    fit_case_ids: Sequence[object],
    role: str,
) -> str:
    held = str(held_case_id)
    if not held or held in {str(value) for value in fit_case_ids}:
        raise ProtocolError("P-DCAPS held case d entered a fitted/support role.")
    if role != "SCORED_RESPONSE_ONLY_AFTER_SURFACE_SEAL":
        raise ProtocolError("P-DCAPS held case d role drifted.")
    return held


__all__ = (
    "PROTOCOL_SCHEMA",
    "frozen_protocol_payload",
    "validate_held_case_role",
    "validate_nested_exclusions",
    "validate_protocol_payload",
)
