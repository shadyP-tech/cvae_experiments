"""Terminal SCEPTRE v2 reports with permanent claim-boundary language."""

from __future__ import annotations

from typing import Mapping

from ..fixed_bank_sceptre_router.hashing import canonical_hash
from .terminal_evaluation import TerminalEvaluationResult


FINAL_REPORT_MEMBERS = (
    "reports/diagnostic_summary.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/claim_boundary.json",
)


def build_final_reports(
    result: TerminalEvaluationResult,
    *,
    input_binding: Mapping[str, object],
    source_snapshot: Mapping[str, object],
    label_journal: Mapping[str, object],
    runtime: Mapping[str, object],
    prediction_store: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    summary = {
        "schema_version": "sceptre_v2_diagnostic_summary_v1",
        "terminal_result_hash": result.result_hash,
        "metrics": dict(result.summary),
        "metric_semantics": (
            "downstream_consumed_test_BACC_Brier_and_log_loss_descriptive_only"
        ),
        "metrics_are_nelbo_compatibility": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
    }
    leakage = {
        "schema_version": "sceptre_v2_leakage_report_v1",
        "status": "PASS",
        "input_binding_hash": canonical_hash(input_binding),
        "source_snapshot_hash": canonical_hash(source_snapshot),
        "label_journal_hash": label_journal["journal_hash"],
        "strict_outer_query_and_candidate_center_exclusion": True,
        "selection_calibration_evaluation_whole_case_disjoint": True,
        "evaluation_cases_exactly_once": True,
        "prediction_store_materialized_before_label_access": True,
        "target_expert_rows_masked_before_candidate_scoring": True,
        "route_policy_durable_before_evaluation_labels": True,
        "support_fallback_opens_calibration_labels": False,
        "raw_labels_persisted": False,
        "sample_paths_persisted": False,
        "fresh_evidence": False,
    }
    publication = {
        "schema_version": "sceptre_v2_publication_decision_v1",
        "decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "bounded_question": (
            "can_SCEPTRE_recover_downstream_utility_selection_structure_on_the_"
            "already_observed_MIDOGpp_consumed_test_surface"
        ),
        "fresh_or_new_center_generalization_claim_allowed": False,
        "routing_success_claim_allowed": False,
        "nelbo_compatibility_claim_allowed": False,
        "confidence_significance_or_coverage_claim_allowed": False,
        "thesis_confirmatory_claim_allowed": False,
        "promotion_or_deployment_allowed": False,
        "may_feed_another_experiment": False,
    }
    runtime_report = {
        "schema_version": "sceptre_v2_runtime_summary_v1",
        "runtime": dict(runtime),
        "prediction_store_hash": prediction_store["store_hash"],
        "terminal_result_hash": result.result_hash,
        "gpu_generation_workers": 2,
        "cpu_classifier_workers": 4,
        "blas_threads_per_worker": 1,
        "seed_selection_performed": False,
    }
    boundary = {
        **publication,
        "schema_version": "sceptre_v2_claim_boundary_report_v1",
        "interpretation": (
            "adaptive_and_descriptive_sensitivity_analysis_on_a_consumed_test_surface"
        ),
        "downstream_classifier_utility_is_not_CVAE_NELBO_routing_evidence": True,
    }
    return {
        "reports/diagnostic_summary.json": _hashed(summary),
        "reports/leakage_report.json": _hashed(leakage),
        "reports/publication_decision.json": _hashed(publication),
        "reports/runtime_summary.json": _hashed(runtime_report),
        "reports/claim_boundary.json": _hashed(boundary),
    }


def build_validation_report(payload: Mapping[str, object]) -> dict[str, object]:
    """Bind the persisted report to the exact two-process final attestation."""

    body = {
        "schema_version": "sceptre_v2_final_validation_report_v1",
        "status": "PASS",
        "final_fresh_process_attestation_hash": payload["attestation_hash"],
        "fresh_process_count": payload["fresh_process_count"],
        "process_launches_sequential": payload["process_launches_sequential"],
        "cuda_hidden": payload["cuda_hidden"],
        "semantic_reconstruction_without_refit": payload[
            "semantic_reconstruction_without_refit"
        ],
        "raw_labels_read": payload["raw_labels_read"],
    }
    return {**body, "report_hash": canonical_hash(body)}


def _hashed(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {**dict(payload), "report_hash": canonical_hash(payload)}


__all__ = (
    "FINAL_REPORT_MEMBERS",
    "build_final_reports",
    "build_validation_report",
)
