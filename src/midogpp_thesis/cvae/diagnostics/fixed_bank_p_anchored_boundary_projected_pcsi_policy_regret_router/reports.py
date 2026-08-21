"""Reconstructive protocol, leakage, and terminal claim reports."""

from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    PUBLICATION_STATUS,
    STAGE_ID,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash


_TRANSPORT_SEMANTICS = "support_conditioned_endpoint_reconstructed_P_B_I_R"
_TRANSPORT_ENDPOINT_SUPPORT_SCOPE = "endpoint_target_T_minus_held_case_c"
_TRANSPORT_ACTUAL_SOURCE_PRIOR_SCOPE = (
    "q_not_in_endpoint_target_T_or_source_e"
)
_TRANSPORT_DONOR_SOURCE_PRIOR_SCOPE = (
    "q_not_in_outer_H_or_endpoint_target_T_or_source_e"
)
_TRANSPORT_LINEAGE_KEYS = frozenset(
    {
        "schema_version",
        "transport_semantics",
        "outer_target_center",
        "endpoint_target_center",
        "endpoint_support_scope",
        "source_prior_scope",
        "endpoint_state_matrix_hash",
        "source_prior_labels_used_upstream",
        "route_local_support_labels_used_upstream",
        "held_case_evaluation_capability_used_directly",
        "pseudo_evaluation_capability_used_directly",
        "terminal_evaluation_capability_used_directly",
        "label_free_claim",
        "uses_pre_equivalence_endpoint_crossing_rates",
        "identity_level_route_noninterference_required",
        "identity_level_route_noninterference_proven",
        "authorization_valid",
        "protocol_status",
        "raw_labels_persisted",
        "lineage_hash",
    }
)


def validate_transport_endpoint_lineage_payload(
    payload: Mapping[str, object],
    *,
    outer_target_center: str,
    endpoint_target_center: str,
    endpoint_state_matrix_hash: str,
) -> Mapping[str, object]:
    """Validate one exact Option-B transport lineage payload."""

    row = dict(payload)
    unhashed = {key: value for key, value in row.items() if key != "lineage_hash"}
    expected_source_scope = (
        _TRANSPORT_ACTUAL_SOURCE_PRIOR_SCOPE
        if outer_target_center == endpoint_target_center
        else _TRANSPORT_DONOR_SOURCE_PRIOR_SCOPE
    )
    if (
        set(row) != _TRANSPORT_LINEAGE_KEYS
        or row.get("schema_version")
        != "fixed_bank_pcsi_parc_transport_endpoint_lineage_v2"
        or row.get("transport_semantics") != _TRANSPORT_SEMANTICS
        or row.get("outer_target_center") != outer_target_center
        or row.get("endpoint_target_center") != endpoint_target_center
        or row.get("endpoint_support_scope")
        != _TRANSPORT_ENDPOINT_SUPPORT_SCOPE
        or row.get("source_prior_scope") != expected_source_scope
        or row.get("endpoint_state_matrix_hash") != endpoint_state_matrix_hash
        or row.get("source_prior_labels_used_upstream") is not True
        or row.get("route_local_support_labels_used_upstream") is not True
        or row.get("held_case_evaluation_capability_used_directly") is not False
        or row.get("pseudo_evaluation_capability_used_directly") is not False
        or row.get("terminal_evaluation_capability_used_directly") is not False
        or row.get("label_free_claim") is not False
        or row.get("uses_pre_equivalence_endpoint_crossing_rates") is not True
        or row.get("identity_level_route_noninterference_required") is not True
        or row.get("identity_level_route_noninterference_proven") is not False
        or row.get("authorization_valid") is not False
        or row.get("protocol_status")
        != "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
        or row.get("raw_labels_persisted") is not False
        or row.get("lineage_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("PCSI-PARC transport endpoint lineage drifted.")
    return MappingProxyType(row)


def validate_transport_lineage_evidence(
    preterminal: object,
) -> Mapping[str, object]:
    """Reconstruct the complete Option-B descriptor/screen/seal lineage."""

    runtime = getattr(preterminal, "policy_runtime")
    centers = tuple(getattr(preterminal, "surface").centers)
    endpoint_by_center = {
        row.target_center: row for row in getattr(preterminal, "endpoint_products")
    }
    donor_products = getattr(preterminal, "donor_endpoint_products")
    descriptor_payloads: dict[tuple[str, str], dict[str, object]] = {}
    lineage_hashes: dict[tuple[str, str], str] = {}
    for outer in centers:
        for endpoint in centers:
            descriptor = runtime.transport_descriptors_by_outer_candidate[
                (outer, endpoint)
            ]
            products = (
                endpoint_by_center[outer]
                if endpoint == outer
                else donor_products[(outer, endpoint)]
            )
            payload = descriptor.to_payload()
            lineage = validate_transport_endpoint_lineage_payload(
                payload.get("transport_lineage", {}),
                outer_target_center=outer,
                endpoint_target_center=endpoint,
                endpoint_state_matrix_hash=canonical_hash(
                    [list(row) for row in products.state_hashes]
                ),
            )
            unhashed = {
                key: value
                for key, value in payload.items()
                if key != "descriptor_hash"
            }
            if (
                payload.get("schema_version")
                != "fixed_bank_pcsi_parc_transport_descriptor_v2"
                or payload.get("center") != endpoint
                or payload.get("descriptor_hash") != canonical_hash(unhashed)
            ):
                raise ProtocolError("PCSI-PARC transport descriptor lineage drifted.")
            descriptor_payloads[(outer, endpoint)] = payload
            lineage_hashes[(outer, endpoint)] = str(lineage["lineage_hash"])

    screen_payloads: dict[tuple[str, str | None], dict[str, object]] = {}
    for outer in centers:
        for pseudo in (None, *(center for center in centers if center != outer)):
            key = outer, pseudo
            screen = runtime.transport_screens[key]
            payload = screen.to_payload()
            candidate = outer if pseudo is None else pseudo
            references = tuple(
                center
                for center in centers
                if center != outer and (pseudo is None or center != pseudo)
            )
            unhashed = {
                name: value
                for name, value in payload.items()
                if name != "screen_hash"
            }
            if (
                payload.get("schema_version")
                != "fixed_bank_pcsi_parc_transport_screen_v2"
                or payload.get("transport_semantics") != _TRANSPORT_SEMANTICS
                or payload.get("outer_target_center") != outer
                or payload.get("candidate_center") != candidate
                or payload.get("screen_role")
                != ("target" if pseudo is None else "pseudo")
                or tuple(payload.get("reference_centers", ())) != references
                or payload.get("candidate_descriptor_hash")
                != descriptor_payloads[(outer, candidate)]["descriptor_hash"]
                or tuple(payload.get("reference_descriptor_hashes", ()))
                != tuple(
                    descriptor_payloads[(outer, center)]["descriptor_hash"]
                    for center in references
                )
                or payload.get("candidate_lineage_hash")
                != lineage_hashes[(outer, candidate)]
                or tuple(payload.get("reference_lineage_hashes", ()))
                != tuple(lineage_hashes[(outer, center)] for center in references)
                or not _transport_flags_are_exact(payload)
                or payload.get("equality_passes") is not True
                or payload.get("screen_hash") != canonical_hash(unhashed)
            ):
                raise ProtocolError("PCSI-PARC transport screen lineage drifted.")
            screen_payloads[key] = payload

    seal = runtime.transport_seal.to_payload()
    descriptor_matrix_hash = canonical_hash(
        [
            {
                "outer_target_center": outer,
                "endpoint_target_center": endpoint,
                "descriptor_hash": descriptor_payloads[(outer, endpoint)][
                    "descriptor_hash"
                ],
                "lineage_hash": lineage_hashes[(outer, endpoint)],
            }
            for outer in centers
            for endpoint in centers
        ]
    )
    screen_matrix_hash = canonical_hash(
        [
            {
                "outer_target_center": outer,
                "pseudo_target_center": pseudo,
                "screen_hash": screen_payloads[(outer, pseudo)][
                    "screen_hash"
                ],
            }
            for outer in centers
            for pseudo in (None, *(center for center in centers if center != outer))
        ]
    )
    seal_unhashed = {
        key: value for key, value in seal.items() if key != "transport_hash"
    }
    expected_count = len(centers) * len(centers)
    if (
        seal.get("schema_version")
        != "fixed_bank_pcsi_parc_transport_runtime_seal_v2"
        or seal.get("transport_semantics") != _TRANSPORT_SEMANTICS
        or seal.get("descriptor_count") != expected_count
        or seal.get("screen_count") != expected_count
        or seal.get("descriptor_hash") != descriptor_matrix_hash
        or seal.get("screen_hash") != screen_matrix_hash
        or not _transport_flags_are_exact(seal)
        or seal.get("screens_sealed_before_pseudo_evaluation_capability_open")
        is not True
        or seal.get("screens_sealed_before_terminal_evaluation_capability_open")
        is not True
        or seal.get("raw_labels_persisted") is not False
        or seal.get("transport_hash") != canonical_hash(seal_unhashed)
        or runtime.transport_hash != seal.get("transport_hash")
    ):
        raise ProtocolError("PCSI-PARC transport runtime lineage seal drifted.")
    evidence = {
        "transport_semantics": _TRANSPORT_SEMANTICS,
        "transport_label_free_claim": False,
        "transport_source_prior_labels_used_upstream": True,
        "transport_route_local_support_labels_used_upstream": True,
        "transport_held_case_evaluation_capability_used_directly": False,
        "transport_pseudo_evaluation_capability_used_directly": False,
        "transport_terminal_evaluation_capability_used_directly": False,
        "transport_authorization_valid": False,
        "transport_held_case_own_route_nonuse_validated": False,
        "transport_identity_feedback_detected": True,
        "transport_runtime_protocol_status": (
            "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
        ),
        "canonical_persistence_prohibited": True,
        "transport_descriptor_count": expected_count,
        "transport_screen_count": expected_count,
        "transport_runtime_hash": str(seal["transport_hash"]),
        "transport_lineage_reconstructed": True,
    }
    return MappingProxyType(
        {**evidence, "transport_lineage_evidence_hash": canonical_hash(evidence)}
    )


def assert_transport_authorization_lineage_valid(preterminal: object) -> None:
    """Prevent persistence while own-route held-label invariance is unproved."""

    evidence = validate_transport_lineage_evidence(preterminal)
    if evidence.get("transport_authorization_valid") is not True:
        raise ProtocolError(
            "PCSI-PARC transport authorization is invalid under "
            "BLOCKED_IDENTITY_FEEDBACK: the center-wide endpoint descriptor "
            "is not invariant to held-case or pseudo-target label poison. "
            "Canonical persistence is prohibited."
        )


def _transport_flags_are_exact(payload: Mapping[str, object]) -> bool:
    return bool(
        payload.get("source_prior_labels_used_upstream") is True
        and payload.get("route_local_support_labels_used_upstream") is True
        and payload.get("held_case_evaluation_capability_used_directly") is False
        and payload.get("pseudo_evaluation_capability_used_directly") is False
        and payload.get("terminal_evaluation_capability_used_directly") is False
        and payload.get("label_free_claim") is False
        and payload.get("uses_pre_equivalence_endpoint_crossing_rates") is True
        and payload.get("identity_level_route_noninterference_required") is True
        and payload.get("identity_level_route_noninterference_proven") is False
        and payload.get("authorization_valid") is False
        and payload.get("protocol_status")
        == "BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK"
    )


def protocol_manifest_payload(
    config: object,
    *,
    protocol: object,
    provenance: Mapping[str, Mapping[str, object]],
    cache_binding_hash: str,
    pre_gpu_firewall: Mapping[str, object],
) -> dict[str, object]:
    input_hashes = {
        artifact_id: canonical_hash(dict(row))
        for artifact_id, row in provenance.items()
    }
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_protocol_manifest_v1",
        "experiment_id": str(getattr(config, "experiment_id")),
        "output_artifact_id": str(getattr(config, "output_artifact_id")),
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": str(getattr(protocol, "protocol_hash")),
        "stage": STAGE_ID,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "input_artifact_hashes": input_hashes,
        "input_artifact_count": len(input_hashes),
        "cache_binding_hash": cache_binding_hash,
        "pre_gpu_firewall": dict(pre_gpu_firewall),
        "exact_six_original_inputs": True,
        "previous_stage90_output_or_checkpoint_used": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "publication_status": PUBLICATION_STATUS,
    }
    return {**payload, "protocol_manifest_hash": canonical_hash(payload)}


def leakage_report_payload(
    *,
    probability_surface_hash: str,
    preterminal: object,
    capability_report: Mapping[str, object],
) -> dict[str, object]:
    donor_runtime = getattr(preterminal, "donor_runtime")
    policy_runtime = getattr(preterminal, "policy_runtime")
    transport_lineage = validate_transport_lineage_evidence(preterminal)
    double_plans = getattr(preterminal, "plans").double_by_key
    pseudo_model_scopes_match_plans = all(
        runtime.pseudo_full_models[pair].training_centers
        == double_plans[pair].model_training_centers
        and set(runtime.pseudo_delete_models[pair])
        == set(double_plans[pair].model_training_centers)
        and all(
            model.training_centers
            == tuple(
                center
                for center in double_plans[pair].model_training_centers
                if center != deleted
            )
            for deleted, model in runtime.pseudo_delete_models[pair].items()
        )
        for runtime in donor_runtime.geometry_results.values()
        for pair in double_plans
    )
    expected_triples = {
        (outer, pseudo, donor)
        for outer, pseudo in double_plans
        for donor in getattr(preterminal, "surface").centers
        if donor not in {outer, pseudo}
    }
    pseudo_prior_scopes_match_triples = bool(
        set(donor_runtime.pseudo_prior_provenance) == expected_triples
        and set(donor_runtime.pseudo_donor_endpoint_products) == expected_triples
        and all(
            all(
                outer not in centers
                and pseudo not in centers
                and donor not in centers
                and source not in centers
                for source, centers in provenance.query_centers_by_source
            )
            and donor_runtime.pseudo_donor_endpoint_products[
                (outer, pseudo, donor)
            ].target_center
            == donor
            and all(
                dict(state.donor_priors) == dict(provenance.prior_values)
                for _case, state in donor_runtime.pseudo_donor_endpoint_products[
                    (outer, pseudo, donor)
                ].states
            )
            for (outer, pseudo, donor), provenance in (
                donor_runtime.pseudo_prior_provenance.items()
            )
        )
    )
    scope_status = bool(
        pseudo_model_scopes_match_plans and pseudo_prior_scopes_match_triples
    )
    if not scope_status:
        raise ProtocolError("PCSI-PARC H/J/K/e exclusion evidence drifted.")
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_leakage_report_v1",
        "status": "NEEDS_EVIDENCE",
        "h_j_k_e_exclusion_scope_status": "PASS",
        "probability_surface_hash": probability_surface_hash,
        "outer_plan_seal_hash": str(getattr(preterminal, "plans").seal_hash),
        "decision_barrier_hash": str(
            getattr(preterminal, "decision_barrier")["decision_barrier_hash"]
        ),
        "aggregate_preterminal_seal_hash": str(
            getattr(preterminal, "aggregate_seal")["aggregate_seal_hash"]
        ),
        "donor_runtime_hash": str(donor_runtime.runtime_hash),
        "transport_hash": str(policy_runtime.transport_hash),
        "policy_replay_runtime_hash": str(policy_runtime.runtime_hash),
        "policy_menu_seal_hash": str(
            policy_runtime.policy_menu_seal["policy_menu_seal_hash"]
        ),
        "capability_report_hash": canonical_hash(capability_report),
        "all_physical_probabilities_sealed_before_any_label_access": True,
        "all_218_routes_globally_sealed_before_terminal_labels": True,
        "all_target_and_144_pseudo_policies_globally_sealed_before_any_pseudo_evaluation_label": True,
        "outer_case_labels_excluded_from_own_route": True,
        "outer_support_labels_scoped_to_H_minus_c": True,
        "physical_fingerprint_sealed_before_label_access": True,
        "target_posterior_is_route_local_and_not_shared": True,
        "target_posterior_is_not_the_final_classifier": True,
        "whole_policy_H_J_double_exclusion_used": True,
        "double_exclusion_pair_count": len(getattr(preterminal, "plans").double_exclusion_plans),
        "double_exclusion_state_count": len(policy_runtime.pseudo_candidate_policies),
        "policy_replay_count": len(policy_runtime.replays),
        "outer_target_center_labels_excluded_from_all_donor_features": True,
        "pseudo_utility_model_training_scopes_match_H_J_plans": (
            pseudo_model_scopes_match_plans
        ),
        "pseudo_endpoint_prior_scopes_match_H_J_K_e_exclusion": (
            pseudo_prior_scopes_match_triples
        ),
        "double_excluded_prior_scope_count": len(
            donor_runtime.pseudo_prior_provenance
        ),
        "actual_donor_feature_source_prior_scope": (
            "q_not_in_outer_H_or_training_donor_K_or_source_e"
        ),
        "pseudo_donor_feature_source_prior_scope": (
            "q_not_in_outer_H_or_pseudo_target_J_or_training_donor_K_or_source_e"
        ),
        "utility_response_grant_scope": "donor_J_not_equal_outer_H",
        "pseudo_model_response_row_scope": (
            "donor_q_not_in_outer_H_or_pseudo_J"
        ),
        "utility_features_are_label_free": True,
        "structural_zero_utility_rows_retained": True,
        "projected_action_equivalence_collapsed_before_features": True,
        "projected_selection_uses_target_influence_not_donor_BACC_veto": True,
        "proper_loss_predictions_constrain_selection": True,
        "projected_per_cell_BACC_veto_used": False,
        "complete_delete_one_donor_families_used": True,
        "held_donor_residual_calibration_used": True,
        "whole_policy_regret_computed_after_selection": True,
        **dict(transport_lineage),
        "transport_protocol_status": "BLOCKED_IDENTITY_FEEDBACK",
        "transport_diagnostic_scope_valid": False,
        "terminal_diagnostic_bundle_valid": False,
        "target_expert_used": False,
        "source_or_shared_model_updated": False,
        "held_case_evaluation_capability_used_before_route_seal": False,
        "raw_labels_persisted": False,
        "sample_or_image_paths_persisted": False,
        "terminal_diagnostics_may_change_same_surface_routes": False,
        "may_feed_another_experiment": False,
        "fresh_evidence": False,
    }
    return {**payload, "leakage_report_hash": canonical_hash(payload)}


def publication_decision_payload(terminal: object) -> dict[str, object]:
    summary = dict(getattr(terminal, "diagnostic_summary"))
    return {
        "schema_version": "fixed_bank_pcsi_parc_publication_decision_v1",
        "status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "terminal_evaluation_seal_hash": str(
            getattr(terminal, "terminal_seal")["terminal_seal_hash"]
        ),
        "diagnostic_summary": summary,
        "unconfirmed_thesis_specific_mechanism_hypothesis": (
            "P-anchored_boundary-projected_target-influence_selection_with_"
            "H-J-double-excluded_whole-policy_regret_and_transport_abstention"
        ),
        "generic_ensemble_or_calibration_method_novelty_claimed": False,
        "terminal_information_success_gate_defined": False,
        "terminal_information_is_formal_risk_control": False,
        "routing_success_claim_authorized": False,
        "routing_quality_claim_authorized": False,
        "target_performance_claim_authorized": False,
        "nominal_significance_claim_authorized": False,
        "transport_protocol_status": "BLOCKED_IDENTITY_FEEDBACK",
        "completed_canonical_run_exists": False,
        "terminal_diagnostic_bundle_valid": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "fresh_evidence": False,
    }


def run_state_payload(
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pcsi_parc_run_state_v1",
        "status": status,
        "phase": phase,
        "error": error,
        "error_class": error_class,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
    }


__all__ = (
    "assert_transport_authorization_lineage_valid",
    "leakage_report_payload",
    "protocol_manifest_payload",
    "publication_decision_payload",
    "run_state_payload",
    "validate_transport_endpoint_lineage_payload",
    "validate_transport_lineage_evidence",
)
