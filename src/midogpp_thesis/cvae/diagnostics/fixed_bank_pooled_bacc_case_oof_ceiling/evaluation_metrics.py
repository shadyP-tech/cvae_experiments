"""Pooled fold/center metric assembly and equal-center result inference."""

from __future__ import annotations

from typing import Sequence

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .core_contracts import BinaryLabelRow, SealedProbabilitySurface, SufficientStatisticSurface
from .core_hashing import canonical_hash, finite
from .decisions import DecisionSeal
from .evaluation_contracts import (
    ActionSelectionMetricRow,
    CenterEvaluationMetric,
    FoldEvaluationMetric,
    PooledCeilingEvaluationResult,
    center_equal_ci95,
    equal_center_rows,
    mean_values,
    result_payload,
)
from .evaluation_null import score_permutation_null
from .permutation_plan import PermutationDecisionPlan
from .pooled_metrics import (
    pooled_exact_bacc,
    score_evaluation_statistics_after_preevaluation_seals,
)
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, TERMINAL_DECISION, action_ids


def evaluate_decision_seal(
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    permutation_plan: PermutationDecisionPlan,
    tie_tolerance: float = 1.0e-12,
    confidence_level: float = 0.95,
) -> PooledCeilingEvaluationResult:
    if (
        decision_seal.partition_hash != partition.partition_hash
        or decision_seal.probability_surface_hash != probabilities.surface_hash
        or permutation_plan.partition_hash != partition.partition_hash
        or permutation_plan.probability_surface_hash != probabilities.surface_hash
    ):
        raise ProtocolError("Terminal evaluation inputs are not bound to one sealed run.")
    statistics = score_evaluation_statistics_after_preevaluation_seals(
        probabilities,
        labels,
        decision_seal=decision_seal,
        permutation_plan=permutation_plan,
    )
    return evaluate_statistics_seal(
        decision_seal,
        partition,
        statistics,
        permutation_plan=permutation_plan,
        tie_tolerance=tie_tolerance,
        confidence_level=confidence_level,
    )


def evaluate_statistics_seal(
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    statistics: SufficientStatisticSurface,
    *,
    permutation_plan: PermutationDecisionPlan,
    tie_tolerance: float = 1.0e-12,
    confidence_level: float = 0.95,
) -> PooledCeilingEvaluationResult:
    """Recompute the terminal result from persisted sufficient statistics."""

    if (
        decision_seal.partition_hash != partition.partition_hash
        or permutation_plan.partition_hash != partition.partition_hash
        or decision_seal.probability_surface_hash != permutation_plan.probability_surface_hash
        or decision_seal.all_fold_decisions_sealed_before_evaluation_labels is not True
        or permutation_plan.sealed_before_evaluation_labels is not True
        or permutation_plan.evaluation_labels_used_to_generate_actions is not False
    ):
        raise ProtocolError("Persisted terminal inputs are not one pre-evaluation sealed run.")
    if abs(float(confidence_level) - 0.95) > 1.0e-12:
        raise ProtocolError("Only the predeclared 95% confidence level is allowed.")
    tolerance = finite(tie_tolerance, "tie_tolerance")
    if tolerance != 1.0e-12:
        raise ProtocolError("Terminal fixed-action tie tolerance drifted.")
    expected_prerequisite = canonical_hash(
        {
            "schema_version": "fixed_bank_pooled_bacc_preevaluation_seal_pair_v2",
            "decision_seal_hash": decision_seal.decision_seal_hash,
            "permutation_plan_hash": permutation_plan.plan_hash,
            "all_observed_and_null_actions_sealed": True,
        }
    )
    expected_case_keys = {
        (identity.target_center, identity.case_id) for identity in partition.identities
    }
    if (
        statistics.label_scope
        != "terminal_evaluation_after_observed_and_null_decisions_sealed"
        or statistics.prerequisite_seal_hash != expected_prerequisite
        or set(statistics.allowed_case_keys) != expected_case_keys
    ):
        raise ProtocolError("Persisted evaluation statistics escaped the sealed label capability.")
    identities_per_case: dict[tuple[str, str], int] = {}
    for identity in partition.identities:
        identities_per_case[identity.case_key] = identities_per_case.get(identity.case_key, 0) + 1
    baseline_by_case = {
        row.case_key: row for row in statistics.rows if row.action_id == BASELINE_ACTION_ID
    }
    if any(
        baseline_by_case[key].row_count != count for key, count in identities_per_case.items()
    ):
        raise ProtocolError("Evaluation statistics drifted from partition row counts.")
    lookup = statistics.by_key()

    fold_metrics: list[FoldEvaluationMetric] = []
    for fold in partition.folds:
        cases = fold.evaluation_case_ids
        decision = decision_seal.decision(fold.target_center, fold.fold_ordinal)
        scores = {
            action: pooled_exact_bacc(
                tuple(lookup[(fold.target_center, case, action)] for case in cases)
            )
            for action in action_ids(fold.target_center)
        }
        baseline = scores[BASELINE_ACTION_ID]
        global_metric = scores[decision.global_action_id]
        routed = scores[decision.routed_action_id]
        oracle_value = max(value.exact_bacc for value in scores.values())
        oracle_actions = tuple(
            action
            for action in action_ids(fold.target_center)
            if oracle_value - scores[action].exact_bacc <= tolerance
        )
        headroom = oracle_value - baseline.exact_bacc
        regret = oracle_value - routed.exact_bacc
        fold_metrics.append(
            FoldEvaluationMetric(
                target_center=fold.target_center,
                fold_ordinal=fold.fold_ordinal,
                row_count=baseline.row_count,
                case_count=len(cases),
                n_positive=baseline.n_positive,
                n_negative=baseline.n_negative,
                baseline_bacc=baseline.exact_bacc,
                global_bacc=global_metric.exact_bacc,
                routed_bacc=routed.exact_bacc,
                oracle_action_ids=oracle_actions,
                oracle_bacc=oracle_value,
                global_minus_baseline=global_metric.exact_bacc - baseline.exact_bacc,
                routed_minus_global=routed.exact_bacc - global_metric.exact_bacc,
                routed_minus_baseline=routed.exact_bacc - baseline.exact_bacc,
                oracle_regret=regret,
                normalized_regret=0.0 if headroom <= tolerance else regret / headroom,
                top1_accuracy=float(decision.routed_action_id == oracle_actions[0]),
                tie_aware_top1_accuracy=float(decision.routed_action_id in oracle_actions),
                global_action_id=decision.global_action_id,
                routed_action_id=decision.routed_action_id,
                route_tier=decision.route_tier,
            )
        )
    canonical_folds = tuple(fold_metrics)

    center_metrics: list[CenterEvaluationMetric] = []
    selections_by_case: list[tuple[str, str]] = []
    for center in MIDOGPP_CENTERS:
        cases = tuple(
            sorted(case for row_center, case in statistics.allowed_case_keys if row_center == center)
        )
        decisions = {
            case: decision_seal.decision(
                center, partition.evaluation_fold_for_case(center, case).fold_ordinal
            )
            for case in cases
        }
        baseline_rows = tuple(lookup[(center, case, BASELINE_ACTION_ID)] for case in cases)
        global_rows = tuple(
            lookup[(center, case, decisions[case].global_action_id)] for case in cases
        )
        routed_rows = tuple(
            lookup[(center, case, decisions[case].routed_action_id)] for case in cases
        )
        baseline = pooled_exact_bacc(baseline_rows)
        global_metric = pooled_exact_bacc(global_rows)
        routed = pooled_exact_bacc(routed_rows)
        fixed_scores = {
            action: pooled_exact_bacc(
                tuple(lookup[(center, case, action)] for case in cases)
            ).exact_bacc
            for action in action_ids(center)
        }
        fixed_best = max(fixed_scores.values())
        best_actions = tuple(
            action
            for action in action_ids(center)
            if fixed_best - fixed_scores[action] <= tolerance
        )
        center_folds = tuple(row for row in canonical_folds if row.target_center == center)
        local_count = sum(
            decisions[case].routed_action_id != decisions[case].global_action_id
            for case in cases
        )
        center_metrics.append(
            CenterEvaluationMetric(
                target_center=center,
                row_count=baseline.row_count,
                case_count=len(cases),
                n_positive=baseline.n_positive,
                n_negative=baseline.n_negative,
                baseline_bacc=baseline.exact_bacc,
                global_bacc=global_metric.exact_bacc,
                routed_bacc=routed.exact_bacc,
                best_fixed_action_ids=best_actions,
                best_fixed_action_bacc=fixed_best,
                global_minus_baseline=global_metric.exact_bacc - baseline.exact_bacc,
                routed_minus_global=routed.exact_bacc - global_metric.exact_bacc,
                routed_minus_baseline=routed.exact_bacc - baseline.exact_bacc,
                routed_regret_to_best_fixed=fixed_best - routed.exact_bacc,
                local_route_coverage=local_count / len(cases),
                mean_fold_normalized_regret=mean_values(
                    row.normalized_regret for row in center_folds
                ),
                mean_fold_top1_accuracy=mean_values(
                    row.top1_accuracy for row in center_folds
                ),
                mean_fold_tie_aware_top1_accuracy=mean_values(
                    row.tie_aware_top1_accuracy for row in center_folds
                ),
            )
        )
        selections_by_case.extend(
            (decisions[case].global_action_id, decisions[case].routed_action_id)
            for case in cases
        )
    canonical_centers = tuple(center_metrics)
    action_selection_rows = _action_selection_rows(selections_by_case)
    gmb = center_equal_ci95(tuple(value.global_minus_baseline for value in canonical_centers))
    rmg = center_equal_ci95(tuple(value.routed_minus_global for value in canonical_centers))
    rmb = center_equal_ci95(tuple(value.routed_minus_baseline for value in canonical_centers))
    observed_coverage = sum(
        value.local_route_coverage * value.case_count for value in canonical_centers
    ) / sum(value.case_count for value in canonical_centers)
    permutation_summary = score_permutation_null(
        permutation_plan=permutation_plan,
        decision_seal=decision_seal,
        partition=partition,
        lookup=lookup,
        observed_center_equal_r_minus_g=rmg[0],
        observed_route_coverage=observed_coverage,
    )
    values = {
        "fold_metric_rows": canonical_folds,
        "center_metrics": canonical_centers,
        "equal_center_inference_rows": equal_center_rows(canonical_centers),
        "action_selection_rows": action_selection_rows,
        "permutation_null_summary_rows": (permutation_summary,),
        "decision_seal_hash": decision_seal.decision_seal_hash,
        "permutation_plan_hash": permutation_plan.plan_hash,
        "evaluation_statistics_surface_hash": statistics.statistics_surface_hash,
        "total_row_count": sum(value.row_count for value in canonical_centers),
        "total_case_count": sum(value.case_count for value in canonical_centers),
        "mean_center_baseline_bacc": mean_values(value.baseline_bacc for value in canonical_centers),
        "mean_center_global_bacc": mean_values(value.global_bacc for value in canonical_centers),
        "mean_center_routed_bacc": mean_values(value.routed_bacc for value in canonical_centers),
        "mean_center_global_minus_baseline": gmb[0],
        "global_minus_baseline_ci95_lower": gmb[1],
        "global_minus_baseline_ci95_upper": gmb[2],
        "mean_center_routed_minus_global": rmg[0],
        "routed_minus_global_ci95_lower": rmg[1],
        "routed_minus_global_ci95_upper": rmg[2],
        "mean_center_routed_minus_baseline": rmb[0],
        "routed_minus_baseline_ci95_lower": rmb[1],
        "routed_minus_baseline_ci95_upper": rmb[2],
        "mean_center_normalized_regret": mean_values(
            value.mean_fold_normalized_regret for value in canonical_centers
        ),
        "mean_center_top1_accuracy": mean_values(
            value.mean_fold_top1_accuracy for value in canonical_centers
        ),
        "mean_center_tie_aware_top1_accuracy": mean_values(
            value.mean_fold_tie_aware_top1_accuracy for value in canonical_centers
        ),
        "local_route_coverage": observed_coverage,
        "decision": TERMINAL_DECISION,
        "consumed_test_data": True,
        "policy_update_authorized": False,
    }
    return PooledCeilingEvaluationResult(
        **values, scientific_result_hash=canonical_hash(result_payload(values))
    )


def _action_selection_rows(
    selections_by_case: Sequence[tuple[str, str]],
) -> tuple[ActionSelectionMetricRow, ...]:
    actions = (BASELINE_ACTION_ID, *MIDOGPP_CENTERS)
    total = len(selections_by_case)
    return tuple(
        ActionSelectionMetricRow(
            method_id=method,
            action_id=action,
            selection_count=sum(
                (global_action if method == "G_H" else routed_action) == action
                for global_action, routed_action in selections_by_case
            ),
            total_case_count=total,
            selection_share=sum(
                (global_action if method == "G_H" else routed_action) == action
                for global_action, routed_action in selections_by_case
            )
            / total,
        )
        for method in ("G_H", "R")
        for action in actions
    )


__all__ = ("evaluate_decision_seal", "evaluate_statistics_seal")
