"""Vectorized held-fold scoring for the presealed permutation action plan."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .core_contracts import CaseActionSufficientStatistics
from .decisions import DecisionSeal
from .evaluation_contracts import PermutationNullSummaryRow
from .permutation_plan import PermutationDecisionPlan
from .pooled_metrics import pooled_exact_bacc
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, action_ids


def score_permutation_null(
    *,
    permutation_plan: PermutationDecisionPlan,
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    lookup: Mapping[tuple[str, str, str], CaseActionSufficientStatistics],
    observed_center_equal_r_minus_g: float,
    observed_route_coverage: float,
) -> PermutationNullSummaryRow:
    count = permutation_plan.permutation_count
    null_center_contrasts = np.empty((count, len(MIDOGPP_CENTERS)), dtype=np.float64)
    global_codes = np.empty(len(permutation_plan.fold_keys), dtype=np.uint8)
    fold_case_weights = np.empty(len(permutation_plan.fold_keys), dtype=np.float64)
    for column, (center, fold_ordinal) in enumerate(permutation_plan.fold_keys):
        decision = decision_seal.decision(center, fold_ordinal)
        global_codes[column] = action_ids(center).index(decision.global_action_id)
        fold_case_weights[column] = len(
            partition.fold(center, fold_ordinal).evaluation_case_ids
        )
    for center_index, center in enumerate(MIDOGPP_CENTERS):
        cases = tuple(
            dict.fromkeys(
                sorted(case for row_center, case, _action in lookup if row_center == center)
            )
        )
        n_positive = sum(
            lookup[(center, case, BASELINE_ACTION_ID)].n_positive for case in cases
        )
        n_negative = sum(
            lookup[(center, case, BASELINE_ACTION_ID)].n_negative for case in cases
        )
        if n_positive <= 0 or n_negative <= 0:
            raise ProtocolError("Permutation evaluation center lacks a pooled binary class.")
        true_positive = np.zeros(count, dtype=np.float64)
        true_negative = np.zeros(count, dtype=np.float64)
        global_rows: list[CaseActionSufficientStatistics] = []
        for case in cases:
            fold = partition.evaluation_fold_for_case(center, case)
            column = center_index * 5 + fold.fold_ordinal
            codes = permutation_plan.action_codes[:, column]
            actions = action_ids(center)
            tp = np.asarray(
                [lookup[(center, case, action)].true_positive for action in actions]
            )
            tn = np.asarray(
                [lookup[(center, case, action)].true_negative for action in actions]
            )
            true_positive += tp[codes]
            true_negative += tn[codes]
            decision = decision_seal.decision(center, fold.fold_ordinal)
            global_rows.append(lookup[(center, case, decision.global_action_id)])
        null_bacc = 0.5 * (true_positive / n_positive + true_negative / n_negative)
        global_bacc = pooled_exact_bacc(tuple(global_rows)).exact_bacc
        null_center_contrasts[:, center_index] = null_bacc - global_bacc
    null_values = null_center_contrasts.mean(axis=1)
    weighted_switches = (
        permutation_plan.action_codes != global_codes[None, :]
    ) * fold_case_weights[None, :]
    null_coverage = weighted_switches.sum(axis=1) / fold_case_weights.sum()
    null_mean = float(null_values.mean())
    null_sd = float(null_values.std(ddof=1)) if count > 1 else 0.0
    one_sided = (
        1.0 + float(np.count_nonzero(null_values >= observed_center_equal_r_minus_g))
    ) / (count + 1.0)
    lower_tail = (
        1.0 + float(np.count_nonzero(null_values <= observed_center_equal_r_minus_g))
    ) / (count + 1.0)
    two_sided = min(1.0, 2.0 * min(one_sided, lower_tail))
    quantiles = np.quantile(null_values, (0.025, 0.5, 0.975))
    return PermutationNullSummaryRow(
        contrast="R-G_H",
        observed_center_equal_value=observed_center_equal_r_minus_g,
        permutation_count=count,
        permutation_seed=permutation_plan.permutation_seed,
        null_mean=null_mean,
        null_standard_deviation=null_sd,
        null_q025=float(quantiles[0]),
        null_q500=float(quantiles[1]),
        null_q975=float(quantiles[2]),
        one_sided_p_value=one_sided,
        lower_tail_p_value=lower_tail,
        two_sided_p_value=two_sided,
        observed_route_coverage=observed_route_coverage,
        null_mean_route_coverage=float(null_coverage.mean()),
        permutation_plan_hash=permutation_plan.plan_hash,
    )


__all__ = ("score_permutation_null",)
