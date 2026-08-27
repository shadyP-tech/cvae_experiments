"""Frozen scientific and claim protocol for executable SCEPTRE v2."""

from __future__ import annotations

from collections.abc import Mapping

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from .experiment_contracts import INPUT_ARTIFACT_IDS
from .identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    canonical_hash,
)


PROTOCOL_SCHEMA = "sceptre_v2_executable_terminal_protocol_v1"


def frozen_protocol_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2_3840",
        "domain_axis": "scanner_center",
        "candidate_unit": "source_center_family",
        "candidate_inventory": "EXACT_C_MINUS_H",
        "centers": list(CENTERS),
        "training_seeds": list(TRAINING_SEEDS),
        "generation_seeds": list(GENERATION_SEEDS),
        "seed_cells_per_family": 9,
        "seed_cells_are_nuisance_replications": True,
        "seed_selection_allowed": False,
        "strict_outer_center_exclusion": (
            "delete_all_q_equal_H_or_e_equal_H_before_transform_fit_or_tuning"
        ),
        "nested_lodo_hyperparameter_selection": True,
        "source_inner_use": "ADAPTIVE_DESCRIPTIVE_DEVELOPMENT_ONLY",
        "source_inner_member_count": 7,
        "source_inner_self_pairs_forbidden": True,
        "classifier": {
            "family": "sklearn_logistic_regression",
            "C": 0.01,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": None,
            "random_state": 23,
            "l1_ratio": None,
            "threshold_policy": "predict",
            "scaler_fit": "synthetic_train_only",
        },
        "physical_surface": {
            "full_source_rows_per_class": 1024,
            "exact_B_source_count": 8,
            "exact_B_rows_per_source_per_class": 128,
            "target_expert_excluded": True,
            "candidate_target_rows_scored": False,
            "candidate_target_row_storage": "SEALED_MINUS_ONE_SENTINEL",
            "generation_lock_frozen": True,
            "expert_bank_frozen": True,
        },
        "target_protocol": {
            "fold_count_per_center": 5,
            "selection_calibration_evaluation_whole_case_disjoint": True,
            "prelabel_router_model_thresholds_and_G_frozen_before_label_access": True,
            "per_fold_policy_uses_disjoint_selection_and_calibration_labels": True,
            "selection_calibration_metric_family": (
                "downstream_classifier_BACC_Brier_log_loss_not_NELBO"
            ),
            "selection_compares_only_target_global_G_against_exact_B": True,
            "uncertainty_compares_only_same_G_against_exact_B": True,
            "calibration_may_keep_same_G_or_return_exact_B_only": True,
            "ties_or_nonfinite_states_return_exact_B": True,
            "exact_fold_count": 45,
            "raw_labels_may_be_persisted": False,
            "durable_preterminal_attestation_required": True,
            "fresh_preterminal_validation_process_count": 2,
            "terminal_evaluation_after_route_policy_seal_only": True,
        },
        "phase_order": [
            "ALL_G_LABEL_FREE_DECISIONS_SEALED",
            "ALL_45_SELECTION_DECISIONS_SEALED",
            "ALL_45_CALIBRATION_DECISIONS_SEALED",
            "EXACT_45_FOLD_ROUTE_POLICY_SERIALIZED",
            "DURABLE_PRETERMINAL_ATTESTATION",
            "TERMINAL_EVALUATION_LABEL_ACCESS",
            "TWO_FRESH_FINAL_VALIDATIONS",
            "POSTVALIDATION_INDEX_AUTHENTICATED",
        ],
        "execution_authorized_only_by_exact_amendment_bytes": True,
        "single_use_execution_identity": True,
        "cross_run_recovery_allowed": False,
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "significance_claimed": False,
        "promotion_allowed": False,
        "deployment_claimed": False,
        "may_feed_another_experiment": False,
    }
    payload["protocol_hash"] = canonical_hash(payload)
    return payload


def claim_boundary_payload() -> dict[str, object]:
    return {
        "schema_version": "sceptre_v2_claim_boundary_v1",
        "experiment_id": EXPERIMENT_ID,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": "diagnostic_only",
        "bounded_question": (
            "can_the_architecture_recover_downstream_utility_selection_structure_"
            "already_present_on_the_consumed_midogpp_test_surface"
        ),
        "fresh_evidence": False,
        "consumed_test_reuse_authorized": True,
        "execution_authorized": True,
        "implementation_authorizes_execution": False,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "adaptive_architecture_comparison": True,
        "comparisons_are_descriptive_only": True,
        "new_center_generalization_claimed": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "confidence_bound_claimed": False,
        "significance_claimed": False,
        "thesis_confirmatory_improvement_claimed": False,
        "promotion_allowed": False,
        "deployment_claimed": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
    }


def validate_protocol_payload(payload: Mapping[str, object]) -> None:
    if dict(payload) != frozen_protocol_payload():
        raise ProtocolError("SCEPTRE v2 scientific protocol drifted.")
    if (
        payload.get("fresh_evidence") is not False
        or payload.get("routing_success_claimed") is not False
        or payload.get("nelbo_compatibility_claimed") is not False
        or payload.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError("SCEPTRE v2 claim firewall drifted.")


__all__ = (
    "PROTOCOL_SCHEMA",
    "claim_boundary_payload",
    "frozen_protocol_payload",
    "validate_protocol_payload",
)
