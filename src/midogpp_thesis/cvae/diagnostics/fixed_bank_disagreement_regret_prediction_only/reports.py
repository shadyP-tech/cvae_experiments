"""Claim-bound reports for a terminal label-free prediction artifact."""

from __future__ import annotations

from typing import Mapping

from ...protocol import ProtocolError
from .hashing import canonical_hash


def leakage_report_payload(
    *,
    source_prediction_seal_hash: str,
    test_prediction_seal_hash: str,
    source_label_capability_report: Mapping[str, object],
    model_bank_hash: str,
    frozen_test_prediction_hash: str,
) -> dict[str, object]:
    report = dict(source_label_capability_report)
    required = {
        "source_labels_opened": True,
        "source_labels_opened_after_complete_prediction_seal": True,
        "raw_source_labels_persisted": False,
        "test_labels_opened": False,
        "test_labels_available": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ProtocolError("Prediction-only label capability report drifted.")
    unhashed = {
        "schema_version": "midogpp_disagreement_regret_prediction_only_leakage_v1",
        "status": "PASS",
        "source_prediction_seal_hash": source_prediction_seal_hash,
        "test_prediction_seal_hash": test_prediction_seal_hash,
        "model_bank_hash": model_bank_hash,
        "frozen_test_prediction_hash": frozen_test_prediction_hash,
        "source_labels_opened_after_complete_prediction_seal": True,
        "source_labels_used_for_training_only": True,
        "source_labels_persisted": False,
        "test_labels_opened": False,
        "test_labels_available_to_runner": False,
        "test_scoring_manifest_opened": False,
        "test_bacc_or_regret_computed": False,
        "test_oracle_computed": False,
        "test_predictions_fed_back_to_training": False,
        "target_expert_used_on_its_own_center": False,
        "outer_H_query_q_candidate_e_exclusions_enforced": True,
        "source_oof_query_q_excluded_before_every_classifier_fit": True,
        "source_oof_oriented_prediction_cell_count": 10_368,
        "source_oof_physical_classifier_fit_count": 5_184,
        "target_compatible_classifier_fit_count": 1_458,
        "test_phase_classifier_fit_count": 0,
        "source_oof_mass_normalization_label_tuned": False,
        "family_hyperparameters_frozen_before_source_labels": True,
        "model_bank_frozen_before_test_inference": True,
        "prior_stage90_prediction_or_result_consumed": False,
        "raw_source_labels_persisted": False,
        "raw_test_labels_persisted": False,
    }
    return {**unhashed, "leakage_hash": canonical_hash(unhashed)}


def publication_decision_payload(*, frozen_test_prediction_hash: str) -> dict[str, object]:
    unhashed = {
        "schema_version": "midogpp_disagreement_regret_prediction_only_publication_v1",
        "status": "TERMINAL_LABEL_FREE_PREDICTION_DIAGNOSTIC_DO_NOT_PROMOTE",
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "claim_role": "posthoc_source_oof_trained_consumed_test_prediction_only_diagnostic",
        "frozen_test_prediction_hash": frozen_test_prediction_hash,
        "whole_test_rows_predicted": 9_928,
        "test_labels_used": False,
        "test_metrics_computed": False,
        "routing_success_claimed": False,
        "routing_quality_claimed": False,
        "target_performance_claimed": False,
        "fresh_evidence": False,
        "consumed_test_data": True,
        "source_method_development_is_posthoc": True,
        "may_authorize_routing": False,
        "may_authorize_action_selection": False,
        "may_authorize_policy_update": False,
        "may_authorize_model_update": False,
        "may_authorize_expert_update": False,
        "may_authorize_promotion": False,
        "may_authorize_deployment": False,
        "may_feed_stage50": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_another_stage90": False,
        "may_feed_another_experiment": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }
    return {**unhashed, "publication_hash": canonical_hash(unhashed)}


__all__ = ("leakage_report_payload", "publication_decision_payload")
