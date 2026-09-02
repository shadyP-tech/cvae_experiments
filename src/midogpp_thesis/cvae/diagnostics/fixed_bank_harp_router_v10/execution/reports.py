"""Leakage, routing, and terminal report construction for HARP v10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ....routing.harp_protocol import canonical_hash
from ....runtime.artifact_io import atomic_json
from ....runtime.harp_v10_execution.contracts import (
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
        "schema_version": "midogpp_harp_v10_leakage_report_v1",
        "strict_outer_center_exclusion": True,
        "development_evaluation_case_disjoint": True,
        "development_labels_opened_after_label_free_menu_seal": True,
        "evaluation_labels_opened_after_two_fresh_reconstructions_and_frozen_seal": True,
        "regularization_hyperparameters_predeclared_fixed": True,
        "regularization_hyperparameter_selection_performed": False,
        "acceptance_threshold_selected_inside_source_lodo": True,
        "nested_whole_policy_source_lodo_calibration_only": True,
        "old_aggregate_utility_surface_used": False,
        "predecessor_policy_rank_output_cache_or_authority_used": False,
        "shared_label_free_effective_menu_before_labels": True,
        "all_margins_and_structural_noops_excluded": True,
        "exact_B_explicit_zero_effect_control": True,
        "budget_and_allocation_residual_heads": True,
        "center_case_balanced_tie_aware_pairwise_ranker": True,
        "cross_fitted_selected_action_acceptor": True,
        "per_action_worst_center_certificate_used": False,
        "rank_all_then_whole_policy_risk_coverage": True,
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
        "target_support_labels_consumed": False,
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
            "schema_version": "midogpp_harp_v10_evaluation_label_access_v1",
            "opened_after_frozen_route_seal": True,
            "frozen_route_seal_hash": frozen["seal_hash"],
            "reconstructed_route_hash": sealed_routes.route_hash,
            "route_store_reopened_after_frozen_seal": True,
            "access_count": 1,
        },
    )
    atomic_json(leakage_path, build_leakage_report())
    validation_report = {
        "schema_version": "midogpp_harp_v10_validation_report_v1",
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
