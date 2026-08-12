"""Fail-closed protocol contract for the target-static endpoint diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .experiment_contracts import (
    ACTION_IDS,
    CENTERS,
    DEVELOPMENT_RESPONSE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNT,
    EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_SUPPORT_ROW_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    PRIMARY_CONTRASTS,
    SUPPORT_BOOTSTRAP_REPLICATES,
    SUPPORT_BOOTSTRAP_SEED,
    SUPPORT_CASE_COUNT_PER_CENTER,
    SUPPORT_PARTITION_NAMESPACE,
)


@dataclass(frozen=True)
class ConsumedTestEndpointRouterProtocol:
    evidence_status: str
    consumed_test_data: bool
    fresh_evidence: bool
    support_labels_used: bool
    target_actions_are_static: bool
    same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal: bool
    may_authorize_routing: bool
    may_feed_another_experiment: bool
    workstation_profile: str
    gpu_workers: int
    cpu_workers: int
    threads_per_worker: int
    contract_hash: str

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                "midogpp_utility_aligned_consumed_test_endpoint_router_protocol_v1"
            ),
            "dataset_family": "MIDOG++",
            "claim_role": (
                "posthoc_utility_aligned_target_static_endpoint_router_diagnostic"
            ),
            "evidence_status": self.evidence_status,
            "consumed_test_data": self.consumed_test_data,
            "fresh_evidence": self.fresh_evidence,
            "eligible_test_row_count": EXPECTED_TEST_ROW_COUNT,
            "eligible_test_case_count": EXPECTED_TOTAL_CASE_COUNT,
            "eligible_test_case_counts_by_center": dict(
                EXPECTED_CASE_COUNTS_BY_CENTER
            ),
            "centers": list(CENTERS),
            "support_partition_rule": (
                "canonical_sort_then_first_eight_whole_cases_per_center"
            ),
            "support_partition_is_seed_independent": True,
            "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
            "support_case_count_per_center": SUPPORT_CASE_COUNT_PER_CENTER,
            "support_case_count_total": EXPECTED_SUPPORT_CASE_COUNT,
            "evaluation_case_count_total": EXPECTED_EVALUATION_CASE_COUNT,
            "evaluation_case_counts_by_center": dict(
                EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER
            ),
            "support_row_count": EXPECTED_SUPPORT_ROW_COUNT,
            "evaluation_row_count": EXPECTED_EVALUATION_ROW_COUNT,
            "evaluation_row_counts_by_center": dict(
                EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER
            ),
            "class_coverage_checked_only_after_partition_membership_frozen": True,
            "whole_case_support_evaluation_disjoint": True,
            "all_cases_participate_once_as_support_or_evaluation": True,
            "support_labels_used": self.support_labels_used,
            "development_response_unit": (
                "candidate_H_q_e_exact_nine_probability_ensemble"
            ),
            "development_response": "exact_nine_probability_ensemble_bacc_delta",
            "development_response_count": DEVELOPMENT_RESPONSE_COUNT,
            "strict_H_q_e_exclusion": True,
            "development_predictions_sealed_before_development_labels": True,
            "cross_center_evaluation_labels_used_as_development_q_labels_after_development_seal": True,
            "global_source_control_provenance": "experiment_manifest_only",
            "separate_metadata_profile_artifact_used": False,
            "metadata_identity_or_label_predictor_used": False,
            "action_ids": list(ACTION_IDS),
            "target_actions_are_static": self.target_actions_are_static,
            "case_level_routing_used": False,
            "source_inner_cardinality_transfer_gate_required": True,
            "routed_selected_gain_lcb_strictly_positive_required": True,
            "exact_B_fallback_on_any_policy_gate_failure": True,
            "simultaneous_prelabel_lcb_vs_U_G_P_required": False,
            "target_support_bootstrap_replicates": SUPPORT_BOOTSTRAP_REPLICATES,
            "target_support_bootstrap_seed": SUPPORT_BOOTSTRAP_SEED,
            "same_outer_H_evaluation_labels_used_for_plan_H": False,
            "same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal": (
                self.same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal
            ),
            "primary_contrasts": list(PRIMARY_CONTRASTS),
            "Hxe_role": "terminal_descriptive_oracle_rank_only",
            "previous_stage90_outputs_or_amendments_used": False,
            "stage50_stage60_or_stage70_results_used": False,
            "source_expert_updated": False,
            "target_expert_used": False,
            "target_labels_update_shared_model": False,
            "may_authorize_routing": self.may_authorize_routing,
            "may_authorize_action_or_policy": False,
            "may_authorize_model_or_expert_update": False,
            "may_authorize_promotion_or_deployment": False,
            "may_feed_another_experiment": self.may_feed_another_experiment,
            "workstation": {
                "profile": self.workstation_profile,
                "generation_devices": ["cuda:0", "cuda:1"],
                "gpu_workers": self.gpu_workers,
                "cpu_workers": self.cpu_workers,
                "threads_per_worker": self.threads_per_worker,
                "gpu_then_cpu_phase_order": True,
                "gpu_and_cpu_phases_disjoint": True,
                "parent_cuda_context_forbidden_during_cpu_phase": True,
                "array_storage_dtype": "float32",
                "scientific_reduction_dtype": "float64",
                "multiprocessing_start_method": "spawn",
            },
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "contract_hash": self.contract_hash}


def canonical_consumed_test_protocol() -> ConsumedTestEndpointRouterProtocol:
    provisional = ConsumedTestEndpointRouterProtocol(
        evidence_status="EXPLORATORY_CONSUMED_DATA_ONLY",
        consumed_test_data=True,
        fresh_evidence=False,
        support_labels_used=False,
        target_actions_are_static=True,
        same_outer_H_evaluation_labels_open_only_after_plan_H_and_global_target_prediction_seal=True,
        may_authorize_routing=False,
        may_feed_another_experiment=False,
        workstation_profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        gpu_workers=2,
        cpu_workers=4,
        threads_per_worker=3,
        contract_hash="",
    )
    return ConsumedTestEndpointRouterProtocol(
        **{
            **provisional.__dict__,
            "contract_hash": canonical_sha256(provisional._unhashed_payload()),
        }
    )


def assert_consumed_test_diagnostic_only(
    protocol: ConsumedTestEndpointRouterProtocol,
) -> None:
    expected = canonical_consumed_test_protocol()
    if (
        protocol.to_payload() != expected.to_payload()
        or protocol.contract_hash != canonical_sha256(protocol._unhashed_payload())
    ):
        raise ProtocolError(
            "Target-static endpoint router escaped its consumed-test boundary."
        )


__all__ = (
    "ConsumedTestEndpointRouterProtocol",
    "assert_consumed_test_diagnostic_only",
    "canonical_consumed_test_protocol",
)
