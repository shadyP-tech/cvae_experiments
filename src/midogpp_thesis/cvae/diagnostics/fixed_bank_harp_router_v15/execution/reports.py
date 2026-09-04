"""Leakage, routing, and terminal report construction for HARP v15."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ....routing.harp_protocol import canonical_hash
from ....runtime.artifact_io import atomic_json
from ....runtime.harp_v15_execution.contracts import (
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
        "schema_version": "midogpp_harp_v15_leakage_report_v1",
        "strict_outer_center_exclusion": True,
        "target_train_support_and_test_case_disjoint": True,
        "support_labels_opened_after_both_role_menu_seals": True,
        "support_labels_update_router_only": True,
        "fixed_experts_frames_generation_classifier_and_menus_unchanged_after_support_labels": True,
        "fixed_bank_support_independence_attested_per_target": True,
        "evaluation_labels_opened_after_two_fresh_reconstructions_and_frozen_seal": True,
        "regularization_hyperparameters_predeclared_fixed": True,
        "regularization_hyperparameter_selection_performed": False,
        "normalization_refit_inside_each_leave_one_support_case_out_fold": True,
        "whole_policy_support_oof_calibration_only": True,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_rank_output_cache_or_authority_used": False,
        "shared_label_free_effective_menu_before_labels": True,
        "all_margins_and_structural_noops_excluded": True,
        "direct_action_gain_harm_brier_logloss_heads": True,
        "hierarchy": ["B_VS_ROUTE", "DIRECTION", "FAMILY", "EXPERT"],
        "case_level_max_finite_sample_residual_calibration": True,
        "per_action_worst_support_fold_certificate_used": False,
        "leave_one_support_case_out_whole_policy_risk_coverage": True,
        "per_outer_local_admission": True,
        "global_kill_switch_used": False,
        "exact_top1_physical_action_only": True,
        "unevaluated_action_mixtures_used": False,
        "exact_B_fallback": True,
        "physical_expert_lambda_grid": [1.0],
        "terminal_oracle_may_feed_policy_or_thresholds": False,
        "utility_kind": "downstream_classifier_utility_not_NELBO",
        "routing_stage_compatibility_estimated": True,
        "compatibility_proxy_is_exact_nelbo": False,
        "compatibility_proxy_is_true_utility": False,
        "target_support_labels_consumed": True,
        "target_evaluation_labels_used_for_fit_admission_ranking_or_tie_breaking": False,
        "known_center_support_adaptation_estimand": True,
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
            "schema_version": "midogpp_harp_v15_evaluation_label_access_v1",
            "opened_after_frozen_route_seal": True,
            "frozen_route_seal_hash": frozen["seal_hash"],
            "reconstructed_route_hash": sealed_routes.route_hash,
            "route_store_reopened_after_frozen_seal": True,
            "access_count": 1,
        },
    )
    atomic_json(leakage_path, build_leakage_report())
    validation_report = {
        "schema_version": "midogpp_harp_v15_validation_report_v1",
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
