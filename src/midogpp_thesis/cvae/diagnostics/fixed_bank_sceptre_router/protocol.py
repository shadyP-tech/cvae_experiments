"""Frozen scientific protocol for scoped SCEPTRE development and evaluation."""

from __future__ import annotations

from collections.abc import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from .experiment_contracts import claim_boundary_payload, direct_input_policy_payload
from .hashing import canonical_hash
from .identity import EXPERIMENT_ID


def frozen_protocol_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "sceptre_v1_scoped_terminal_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "development_surface": (
            "IMMUTABLE_SOURCE_INNER_UTILITY_AND_PRELABEL_PREDICTION_PACKET_V1"
        ),
        "development_surface_previously_consumed": True,
        "development_surface_use": "ADAPTIVE_DESCRIPTIVE_MODEL_DEVELOPMENT_ONLY",
        "development_surface_members": [
            "locks/policy_consumption_lock.json",
            "tables/candidate_utility.csv",
            "tables/case_confusions.csv",
            "arrays/candidate_predictions.npz",
            "manifests/prediction_index.json",
            "tables/classifier_fits.csv",
            "tables/evaluation_rows.csv",
        ],
        "prelabel_prediction_packet_contains_labels": False,
        "historical_source_inner_reuse_authorized": True,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "candidate_unit": "source_center_family",
        "candidate_inventory": "EXACT_C_MINUS_H",
        "candidate_family_training_seeds": list(TRAINING_SEEDS),
        "candidate_family_generation_seeds": list(GENERATION_SEEDS),
        "seed_cell_count_per_family": 9,
        "seed_cells_are_nuisance_replications": True,
        "seed_cells_are_independent_observations": False,
        "seed_selection_allowed": False,
        "outer_target_centers": list(CENTERS),
        "strict_outer_center_exclusion": {
            "rule": "DELETE_ALL_ROWS_WITH_QUERY_H_OR_CANDIDATE_H",
            "q_equal_H_removed": True,
            "e_equal_H_removed": True,
            "q_equal_e_forbidden": True,
            "applied_before_feature_transforms": True,
            "applied_before_normalization": True,
            "applied_before_fitting": True,
            "applied_before_hyperparameter_tuning": True,
        },
        "nested_lodo": {
            "complete_rotation_required": True,
            "held_K_query_rows_removed_from_fit": True,
            "held_K_candidate_rows_removed_before_fit_and_validation": True,
            "held_K_removed_from_training_before_training_feature_transform": True,
            "held_K_removed_from_training_before_training_normalization": True,
            "validation_q_equal_K_is_transformed_separately_label_free": True,
            "validation_q_equal_K_may_influence_training_transforms": False,
            "hyperparameters_selected_by_equal_center_descriptive_regret": True,
            "seed_cells_not_resampled_as_independent_units": True,
            "p_values_or_confidence_intervals_allowed": False,
        },
        "architecture": {
            "name": "SCEPTRE",
            "label_free_core": "SOURCE_FAMILY_COMPATIBILITY_EVIDENCE_AND_EXACT_B",
            "historical_development_adapter": (
                "LOW_CAPACITY_PAIRWISE_BACC_CONTRAST_WITH_CANDIDATE_EVIDENCE_INTERACTIONS"
            ),
            "historical_label_free_evidence": (
                "PREDICTIVE_ENTROPY_VOTE_DISAGREEMENT_AND_PROXY_ENERGY_NOT_NELBO"
            ),
            "complete_prelabel_freeze": (
                "EXACT_NINE_H_MODELS_MENUS_B_CONTROLS_THRESHOLDS_PARTITION_AND_POLICY"
            ),
            "g_proposal_scope": "ONE_TARGET_GLOBAL_LABEL_FREE_PROPOSAL_PER_H",
            "g_fold_receipts_are_phase_barrier_attachments_only": True,
            "target_selection": "G_PROPOSAL_ONLY_SUPPORT_COMPARISON_AGAINST_EXACT_B",
            "target_uncertainty": (
                "PAIRED_WHOLE_CASE_DIRICHLET_G_PROPOSAL_VS_EXACT_B_FIXED_0_8_GATE"
            ),
            "target_calibration": (
                "DISJOINT_WHOLE_CASE_UNCERTAINTY_AND_POINT_SAFETY_GATE"
            ),
            "final_route_policy": (
                "CANONICAL_EXACT_45_FOLD_SAME_EXPERT_OR_EXACT_B_TABLE"
            ),
            "final_route_policy_embeds_g_proposal_hash_and_candidate": True,
            "terminal_evaluation": "DISJOINT_WHOLE_CASE_EVALUATION_FOLD",
            "fallback": "B_EXACT_EQUAL_UNION",
            "exact_B_is_first_class_abstention": True,
            "proxy_semantics": "PROXY_ENERGY_RANK",
            "proxy_is_exact_nelbo": False,
            "labels_consumed_by_core": False,
            "support_may_switch_to_another_expert": False,
            "uncertainty_may_switch_to_another_expert": False,
            "calibration_may_switch_or_revive_an_expert": False,
            "only_legal_transitions": ["G_TO_SAME_EXPERT", "ANY_STAGE_TO_EXACT_B"],
        },
        "future_consumed_test_execution": {
            "support_fold_count": 5,
            "selection_calibration_evaluation_whole_case_disjoint": True,
            "every_case_evaluated_exactly_once": True,
            "complete_router_and_thresholds_frozen_before_test_label_access": True,
            "manager_owned_calibration_uncertainty_registration_required": True,
            "calibration_must_match_registered_uncertainty_hash_route_and_gate": True,
            "target_global_g_must_match_across_all_five_fold_attachments": True,
            "final_route_policy_serialized_before_terminal_evaluation": True,
            "final_route_policy_exact_fold_count": 45,
            "terminal_capability_binds_final_route_policy_hash": True,
            "global_phase_order": [
                "ALL_G_LABEL_FREE_DECISIONS_SEALED",
                "ALL_45_SELECTION_DECISIONS_SEALED",
                "ALL_45_CALIBRATION_ROUTE_OR_B_DECISIONS_SEALED",
                "EXACT_45_FOLD_ROUTE_POLICY_SERIALIZED",
                "DURABLE_PRETERMINAL_ATTESTATION",
                "TERMINAL_EVALUATION_LABEL_ACCESS",
            ],
            "raw_labels_may_be_persisted": False,
            "phase_capabilities_are_lineage_only": True,
            "phase_capabilities_authorize_raw_label_access": False,
            "one_shot_label_reader_implemented": False,
            "label_access_execution_layer_status": "ABSENT_FUTURE_IDENTITY_ONLY",
            "durable_preterminal_validation_processes": 2,
            "fresh_validation_process_ids_must_differ": True,
            "fresh_reconstructions_must_be_byte_identical": True,
            "execution_authorized": False,
            "consumed_test_reuse_authorized": False,
            "test_cache_or_manifest_registered": False,
            "test_consumption_ledger_registered": False,
            "genuine_subprocess_attestation_implemented": False,
            "separate_future_execution_identity_required": True,
        },
        "exact_B_fallback_required": True,
        "ties_fall_back_to_exact_B": True,
        "unsupported_or_nonfinite_states_fall_back_to_exact_B": True,
        "execution_authorized": False,
        "implementation_authorizes_execution": False,
        "output_or_scratch_resolution_allowed": False,
        "cross_run_recovery_allowed": False,
        "direct_input_policy": direct_input_policy_payload(),
        "publication_and_claim_boundary": claim_boundary_payload(),
        "may_feed_another_experiment": False,
    }
    payload["protocol_hash"] = canonical_hash(payload)
    return payload


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    expected = frozen_protocol_payload()
    if dict(payload) != expected:
        raise ProtocolError("SCEPTRE scientific protocol drifted.")
    if (
        payload.get("execution_authorized") is not False
        or payload.get("may_feed_another_experiment") is not False
        or payload.get("exact_B_fallback_required") is not True
    ):
        raise ProtocolError("SCEPTRE protocol firewall drifted.")


__all__ = ("frozen_protocol_payload", "validate_protocol_payload")
