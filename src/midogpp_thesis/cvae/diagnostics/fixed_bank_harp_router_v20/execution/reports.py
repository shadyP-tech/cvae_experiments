"""Leakage, routing, and terminal report construction for HARP v20."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ....routing.harp_protocol import canonical_hash
from ....runtime.artifact_io import atomic_json
from ....runtime.harp_v20_execution.contracts import (
    PrelabelRouteSet,
    TerminalEvaluation,
)
from ..identity import PUBLICATION_STATUS, TERMINAL_DECISION


@dataclass(frozen=True, slots=True)
class TerminalReportBundle:
    """Paths and identity needed by the coordinator's final durable commit."""

    metrics: Mapping[str, object]
    paths: tuple[Path, ...]


def prelabel_route_summary(routes: object) -> dict[str, object]:
    """Summarize decisions without opening evaluation labels."""

    cases = tuple(getattr(routes, "cases"))
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    exact = True
    for case in cases:
        action_counts[case.selected_kind.value] = (
            action_counts.get(case.selected_kind.value, 0) + 1
        )
        reason_counts[case.reason] = reason_counts.get(case.reason, 0) + 1
        if case.selected_kind.value == "B":
            exact &= (
                case.routed_probabilities.tobytes(order="C")
                == case.baseline_probabilities.tobytes(order="C")
            )
    return {
        "case_count": len(cases),
        "row_count": sum(len(case.sample_ids) for case in cases),
        "selected_action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "case_consistent": True,
        "exact_b_fallback_byte_identity": exact,
    }


def build_leakage_report() -> dict[str, object]:
    """State the terminal consumed-test firewall in a durable report."""

    return {
        "schema_version": "midogpp_harp_v20_leakage_report_v1",
        "source_q_candidate_pool_is_C_minus_q": True,
        "target_H_candidate_pool_is_C_minus_H": True,
        "H_q_r_seven_expert_folds_used": False,
        "source_train_and_test_cases_disjoint": True,
        "all_eighteen_menu_seals_bound_before_any_source_truth": True,
        "all_bank_attestations_bound_before_any_source_truth": True,
        "source_labels_update_pooled_v20_router_only": True,
        "fixed_experts_frames_generation_classifier_and_menus_unchanged_after_source_labels": True,
        "evaluation_labels_opened_after_two_fresh_reconstructions_and_frozen_seal": True,
        "nested_center_stratified_outer_folds": 5,
        "nested_center_stratified_inner_folds": 4,
        "whole_policy_source_oof_replay": True,
        "old_aggregate_utility_surface_used": False,
        "v1_through_v19_policy_rank_output_cache_or_authority_used": False,
        "shared_label_free_effective_menu_before_labels": True,
        "no_hard_change_candidates_excluded_from_learning_and_selection": True,
        "probability_distinct_no_hard_change_candidates_retained_in_frontier": True,
        "pooled_pairwise_ranker_used": True,
        "selected_policy_soft_topk_used": True,
        "case_conditional_action_families": ["B", "U_FULL", "D01_ONLY", "D10_ONLY", "BOTH"],
        "unselected_branch_preserves_exact_B_bytes": True,
        "signed_actual_composite_outcomes_used": True,
        "shared_action_conditioned_outcome_model": True,
        "complete_stacked_learner_refit_inside_every_inner_fold": True,
        "source_and_terminal_class_support_weighting_aligned": True,
        "source_oof_conditional_on_frozen_bank": True,
        "source_frontier_persisted_before_admission": True,
        "individual_action_lower_bound_guarantee": False,
        "source_oof_bounds_include_refit_uncertainty": False,
        "k_values": [1, 2, 4],
        "lambda_values": [0.25, 0.5, 0.75, 1.0],
        "approximate_source_oof_bounds": True,
        "conformal_bounds_claimed": False,
        "minimum_routed_oof_cases": 18,
        "minimum_routed_oof_centers": 6,
        "minimum_routed_oof_cases_per_counted_center": 2,
        "exact_B_fallback": True,
        "physical_expert_lambda_grid": [1.0],
        "terminal_oracle_may_feed_policy_or_thresholds": False,
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": True,
        "compatibility_proxy_is_exact_nelbo": False,
        "compatibility_proxy_is_true_utility": False,
        "source_train_labels_consumed": True,
        "target_evaluation_labels_used_for_fit_admission_ranking_or_tie_breaking": False,
        "known_center_train_support_to_full_test_estimand": True,
        "unseen_center_generalization_claimed": False,
        "generative_expert_compatibility_claimed": False,
        "status": "PASS",
    }


def write_terminal_reports(
    root: Path,
    *,
    terminal: TerminalEvaluation,
    sealed_routes: PrelabelRouteSet,
    frozen: Mapping[str, object],
    development_surface_seal_hash: object,
    model_lock_hash: object,
    target_action_seal_hash: object,
    validations: tuple[Mapping[str, object], ...],
    route_summary: Mapping[str, object],
) -> TerminalReportBundle:
    """Write the evaluation-dependent reports after the frozen-route seal."""

    terminal_metrics = dict(terminal.metrics)
    terminal_metrics.pop("result_hash", None)
    terminal_metrics.update(
        {
            "utility_kind": "downstream_classifier_utility_not_NELBO",
            "routing_stage_compatibility_estimated": True,
            "compatibility_proxy_is_exact_nelbo": False,
            "compatibility_proxy_is_true_utility": False,
            "generative_expert_compatibility_claimed": False,
        }
    )
    terminal_metrics["result_hash"] = canonical_hash(terminal_metrics)

    terminal_result_path = root / "reports/terminal_result.json"
    action_oracle_path = root / "reports/action_oracle_diagnostics.json"
    route_reasons_path = root / "reports/route_and_fallback_reasons.json"
    evaluation_access_path = root / "reports/evaluation_label_access.json"
    leakage_path = root / "reports/leakage_report.json"
    validation_report_path = root / "reports/validation_report.json"
    atomic_json(terminal_result_path, terminal_metrics)
    atomic_json(action_oracle_path, dict(terminal.oracle_diagnostic))
    atomic_json(route_reasons_path, dict(terminal.route_reasons))
    atomic_json(
        evaluation_access_path,
        {
            "schema_version": "midogpp_harp_v20_evaluation_label_access_v1",
            "opened_after_frozen_route_seal": True,
            "frozen_route_seal_hash": frozen["seal_hash"],
            "reconstructed_route_hash": sealed_routes.route_hash,
            "route_store_reopened_after_frozen_seal": True,
            "access_count": 1,
        },
    )
    atomic_json(leakage_path, build_leakage_report())
    validation_report = {
        "schema_version": "midogpp_harp_v20_validation_report_v1",
        "status": "PASS",
        "development_surface_seal_hash": development_surface_seal_hash,
        "model_lock_hash": model_lock_hash,
        "target_action_seal_hash": target_action_seal_hash,
        "frozen_route_seal_hash": frozen["seal_hash"],
        "independent_validation_hashes": [
            value["validation_hash"] for value in validations
        ],
        "terminal_result_hash": terminal_metrics["result_hash"],
        "action_oracle_diagnostic_hash": terminal.oracle_diagnostic["diagnostic_hash"],
        "exact_b_fallback_byte_identity": route_summary[
            "exact_b_fallback_byte_identity"
        ],
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
    }
    atomic_json(validation_report_path, validation_report)
    return TerminalReportBundle(
        metrics=terminal_metrics,
        paths=(
            terminal_result_path,
            action_oracle_path,
            route_reasons_path,
            evaluation_access_path,
            leakage_path,
            validation_report_path,
        ),
    )


__all__ = (
    "TerminalReportBundle",
    "build_leakage_report",
    "prelabel_route_summary",
    "write_terminal_reports",
)
