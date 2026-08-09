"""Typed contracts for terminal pooled center evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .core_hashing import canonical_hash, finite, require_sha256
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, TERMINAL_DECISION, action_ids


STUDENT_T_975_DF8 = 2.306004135204166


@dataclass(frozen=True, order=True)
class FoldEvaluationMetric:
    target_center: str
    fold_ordinal: int
    row_count: int
    case_count: int
    n_positive: int
    n_negative: int
    baseline_bacc: float
    global_bacc: float
    routed_bacc: float
    oracle_action_ids: tuple[str, ...]
    oracle_bacc: float
    global_minus_baseline: float
    routed_minus_global: float
    routed_minus_baseline: float
    oracle_regret: float
    normalized_regret: float
    top1_accuracy: float
    tie_aware_top1_accuracy: float
    global_action_id: str
    routed_action_id: str
    route_tier: str
    metric_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or not 0 <= self.fold_ordinal < 5:
            raise ProtocolError("Fold evaluation metric identity is invalid.")
        if (
            self.row_count <= 0
            or self.case_count <= 0
            or self.n_positive <= 0
            or self.n_negative <= 0
            or self.row_count != self.n_positive + self.n_negative
        ):
            raise ProtocolError("Fold metric requires both pooled classes and valid counts.")
        actions = action_ids(self.target_center)
        oracle_actions = tuple(self.oracle_action_ids)
        if (
            not oracle_actions
            or any(action not in actions for action in oracle_actions)
            or self.global_action_id not in actions
            or self.routed_action_id not in actions
        ):
            raise ProtocolError("Fold metric action identity drifted.")
        expected_tier = (
            "R"
            if self.routed_action_id != self.global_action_id
            else "G"
            if self.global_action_id != BASELINE_ACTION_ID
            else "B"
        )
        if self.route_tier != expected_tier:
            raise ProtocolError("Fold metric route tier drifted.")
        for name in (
            "baseline_bacc",
            "global_bacc",
            "routed_bacc",
            "oracle_bacc",
            "global_minus_baseline",
            "routed_minus_global",
            "routed_minus_baseline",
            "oracle_regret",
            "normalized_regret",
            "top1_accuracy",
            "tie_aware_top1_accuracy",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        headroom = self.oracle_bacc - self.baseline_bacc
        expected_normalized = 0.0 if headroom <= 1.0e-12 else (
            self.oracle_bacc - self.routed_bacc
        ) / headroom
        expected_top1 = float(self.routed_action_id == oracle_actions[0])
        expected_tie_top1 = float(self.routed_action_id in oracle_actions)
        if (
            abs(self.global_minus_baseline - (self.global_bacc - self.baseline_bacc)) > 1.0e-12
            or abs(self.routed_minus_global - (self.routed_bacc - self.global_bacc)) > 1.0e-12
            or abs(self.routed_minus_baseline - (self.routed_bacc - self.baseline_bacc)) > 1.0e-12
            or abs(self.oracle_regret - (self.oracle_bacc - self.routed_bacc)) > 1.0e-12
            or abs(self.normalized_regret - expected_normalized) > 1.0e-12
            or abs(self.top1_accuracy - expected_top1) > 1.0e-12
            or abs(self.tie_aware_top1_accuracy - expected_tie_top1) > 1.0e-12
        ):
            raise ProtocolError("Fold pooled metric arithmetic drifted.")
        object.__setattr__(self, "oracle_action_ids", oracle_actions)
        object.__setattr__(self, "metric_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_fold_metric_v2",
            "target_center": self.target_center,
            "fold_ordinal": self.fold_ordinal,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "baseline_bacc": self.baseline_bacc,
            "global_bacc": self.global_bacc,
            "routed_bacc": self.routed_bacc,
            "oracle_action_ids": list(self.oracle_action_ids),
            "oracle_bacc": self.oracle_bacc,
            "global_minus_baseline": self.global_minus_baseline,
            "routed_minus_global": self.routed_minus_global,
            "routed_minus_baseline": self.routed_minus_baseline,
            "oracle_regret": self.oracle_regret,
            "normalized_regret": self.normalized_regret,
            "top1_accuracy": self.top1_accuracy,
            "tie_aware_top1_accuracy": self.tie_aware_top1_accuracy,
            "global_action_id": self.global_action_id,
            "routed_action_id": self.routed_action_id,
            "route_tier": self.route_tier,
            "utility_scope": "whole_held_fold_pooled_exact_bacc",
            "zero_headroom_normalized_regret": 0.0,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metric_hash": self.metric_hash}


@dataclass(frozen=True, order=True)
class CenterEvaluationMetric:
    target_center: str
    row_count: int
    case_count: int
    n_positive: int
    n_negative: int
    baseline_bacc: float
    global_bacc: float
    routed_bacc: float
    best_fixed_action_ids: tuple[str, ...]
    best_fixed_action_bacc: float
    global_minus_baseline: float
    routed_minus_global: float
    routed_minus_baseline: float
    routed_regret_to_best_fixed: float
    local_route_coverage: float
    mean_fold_normalized_regret: float
    mean_fold_top1_accuracy: float
    mean_fold_tie_aware_top1_accuracy: float
    metric_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Center metric uses an unknown MIDOG++ target.")
        if (
            self.row_count <= 0
            or self.case_count <= 0
            or self.n_positive <= 0
            or self.n_negative <= 0
            or self.row_count != self.n_positive + self.n_negative
        ):
            raise ProtocolError("Center metric requires both pooled classes and valid counts.")
        best_actions = tuple(self.best_fixed_action_ids)
        if not best_actions or any(action not in action_ids(self.target_center) for action in best_actions):
            raise ProtocolError("Center fixed-action oracle is invalid.")
        for name in (
            "baseline_bacc",
            "global_bacc",
            "routed_bacc",
            "best_fixed_action_bacc",
            "global_minus_baseline",
            "routed_minus_global",
            "routed_minus_baseline",
            "routed_regret_to_best_fixed",
            "local_route_coverage",
            "mean_fold_normalized_regret",
            "mean_fold_top1_accuracy",
            "mean_fold_tie_aware_top1_accuracy",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if (
            abs(self.global_minus_baseline - (self.global_bacc - self.baseline_bacc)) > 1.0e-12
            or abs(self.routed_minus_global - (self.routed_bacc - self.global_bacc)) > 1.0e-12
            or abs(self.routed_minus_baseline - (self.routed_bacc - self.baseline_bacc)) > 1.0e-12
            or abs(self.routed_regret_to_best_fixed - (self.best_fixed_action_bacc - self.routed_bacc))
            > 1.0e-12
            or not 0.0 <= self.local_route_coverage <= 1.0
            or not 0.0 <= self.mean_fold_top1_accuracy <= 1.0
            or not 0.0 <= self.mean_fold_tie_aware_top1_accuracy <= 1.0
        ):
            raise ProtocolError("Center metric arithmetic drifted.")
        object.__setattr__(self, "best_fixed_action_ids", best_actions)
        object.__setattr__(self, "metric_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_center_metric_v2",
            "target_center": self.target_center,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "baseline_bacc": self.baseline_bacc,
            "global_bacc": self.global_bacc,
            "routed_bacc": self.routed_bacc,
            "best_fixed_action_ids": list(self.best_fixed_action_ids),
            "best_fixed_action_bacc": self.best_fixed_action_bacc,
            "global_minus_baseline": self.global_minus_baseline,
            "routed_minus_global": self.routed_minus_global,
            "routed_minus_baseline": self.routed_minus_baseline,
            "routed_regret_to_best_fixed": self.routed_regret_to_best_fixed,
            "local_route_coverage": self.local_route_coverage,
            "mean_fold_normalized_regret": self.mean_fold_normalized_regret,
            "mean_fold_top1_accuracy": self.mean_fold_top1_accuracy,
            "mean_fold_tie_aware_top1_accuracy": self.mean_fold_tie_aware_top1_accuracy,
            "primary_endpoint": "pooled_exact_bacc",
            "per_case_bacc_used": False,
            "whole_case_cluster_uncertainty_used_for_routing": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metric_hash": self.metric_hash}


@dataclass(frozen=True, order=True)
class EqualCenterInferenceRow:
    endpoint: str
    center_count: int
    estimate: float
    ci95_lower: float
    ci95_upper: float
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if not self.endpoint or self.center_count != len(MIDOGPP_CENTERS):
            raise ProtocolError("Equal-center inference identity drifted.")
        for name in ("estimate", "ci95_lower", "ci95_upper"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if not self.ci95_lower <= self.estimate <= self.ci95_upper:
            raise ProtocolError("Equal-center estimate lies outside its interval.")
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_equal_center_inference_v2",
            "endpoint": self.endpoint,
            "center_count": self.center_count,
            "estimate": self.estimate,
            "ci95_lower": self.ci95_lower,
            "ci95_upper": self.ci95_upper,
            "inference_unit": "target_center_equal_weight_n9",
            "student_t_df": 8,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True, order=True)
class ActionSelectionMetricRow:
    method_id: str
    action_id: str
    selection_count: int
    total_case_count: int
    selection_share: float
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.method_id not in ("G_H", "R"):
            raise ProtocolError("Action-selection row uses an unknown method.")
        if self.action_id not in (BASELINE_ACTION_ID, *MIDOGPP_CENTERS):
            raise ProtocolError("Action-selection row uses an unknown action.")
        if (
            isinstance(self.selection_count, bool)
            or self.selection_count < 0
            or isinstance(self.total_case_count, bool)
            or self.total_case_count <= 0
            or self.selection_count > self.total_case_count
        ):
            raise ProtocolError("Action-selection counts are invalid.")
        share = finite(self.selection_share, "selection_share")
        if abs(share - self.selection_count / self.total_case_count) > 1.0e-12:
            raise ProtocolError("Action-selection share differs from its exact count.")
        object.__setattr__(self, "selection_share", share)
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_action_selection_metric_v2",
            "method_id": self.method_id,
            "action_id": self.action_id,
            "selection_count": self.selection_count,
            "total_case_count": self.total_case_count,
            "selection_share": self.selection_share,
            "selection_unit": "whole_oof_case",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True, order=True)
class PermutationNullSummaryRow:
    contrast: str
    observed_center_equal_value: float
    permutation_count: int
    permutation_seed: int
    null_mean: float
    null_standard_deviation: float
    null_q025: float
    null_q500: float
    null_q975: float
    one_sided_p_value: float
    lower_tail_p_value: float
    two_sided_p_value: float
    observed_route_coverage: float
    null_mean_route_coverage: float
    permutation_plan_hash: str
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.contrast != "R-G_H" or self.permutation_count <= 0:
            raise ProtocolError("Permutation summary contrast/count drifted.")
        require_sha256(self.permutation_plan_hash, "permutation_plan_hash")
        for name in (
            "observed_center_equal_value",
            "null_mean",
            "null_standard_deviation",
            "null_q025",
            "null_q500",
            "null_q975",
            "one_sided_p_value",
            "lower_tail_p_value",
            "two_sided_p_value",
            "observed_route_coverage",
            "null_mean_route_coverage",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if not (
            0.0 <= self.one_sided_p_value <= 1.0
            and 0.0 <= self.lower_tail_p_value <= 1.0
            and 0.0 <= self.two_sided_p_value <= 1.0
            and 0.0 <= self.observed_route_coverage <= 1.0
            and 0.0 <= self.null_mean_route_coverage <= 1.0
        ):
            raise ProtocolError("Permutation p-value/coverage domain drifted.")
        if (
            self.null_standard_deviation < 0.0
            or not self.null_q025 <= self.null_q500 <= self.null_q975
            or abs(
                self.two_sided_p_value
                - min(1.0, 2.0 * min(self.one_sided_p_value, self.lower_tail_p_value))
            )
            > 1.0e-12
        ):
            raise ProtocolError("Permutation uncertainty/tail arithmetic drifted.")
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_permutation_null_summary_v2",
            "contrast": self.contrast,
            "observed_center_equal_value": self.observed_center_equal_value,
            "permutation_count": self.permutation_count,
            "permutation_seed": self.permutation_seed,
            "null_mean": self.null_mean,
            "null_standard_deviation": self.null_standard_deviation,
            "null_q025": self.null_q025,
            "null_q500": self.null_q500,
            "null_q975": self.null_q975,
            "one_sided_p_value": self.one_sided_p_value,
            "lower_tail_p_value": self.lower_tail_p_value,
            "two_sided_p_value": self.two_sided_p_value,
            "observed_route_coverage": self.observed_route_coverage,
            "null_mean_route_coverage": self.null_mean_route_coverage,
            "permutation_plan_hash": self.permutation_plan_hash,
            "blocking": "full_candidate_statistics_within_H_fold_support_case",
            "center_equal_inference": True,
            "null_actions_sealed_before_evaluation_labels": True,
            "permutation_primary_statistic": "equal_center_R_minus_G_H",
            "upper_tail_formula": "(1+count(null>=observed))/(K+1)",
            "lower_tail_formula": "(1+count(null<=observed))/(K+1)",
            "two_sided_formula": "min(1,2*min(upper,lower))",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class PooledCeilingEvaluationResult:
    fold_metric_rows: tuple[FoldEvaluationMetric, ...]
    center_metrics: tuple[CenterEvaluationMetric, ...]
    equal_center_inference_rows: tuple[EqualCenterInferenceRow, ...]
    action_selection_rows: tuple[ActionSelectionMetricRow, ...]
    permutation_null_summary_rows: tuple[PermutationNullSummaryRow, ...]
    decision_seal_hash: str
    permutation_plan_hash: str
    evaluation_statistics_surface_hash: str
    total_row_count: int
    total_case_count: int
    mean_center_baseline_bacc: float
    mean_center_global_bacc: float
    mean_center_routed_bacc: float
    mean_center_global_minus_baseline: float
    global_minus_baseline_ci95_lower: float
    global_minus_baseline_ci95_upper: float
    mean_center_routed_minus_global: float
    routed_minus_global_ci95_lower: float
    routed_minus_global_ci95_upper: float
    mean_center_routed_minus_baseline: float
    routed_minus_baseline_ci95_lower: float
    routed_minus_baseline_ci95_upper: float
    mean_center_normalized_regret: float
    mean_center_top1_accuracy: float
    mean_center_tie_aware_top1_accuracy: float
    local_route_coverage: float
    scientific_result_hash: str
    decision: str = TERMINAL_DECISION
    consumed_test_data: bool = True
    policy_update_authorized: bool = False

    def __post_init__(self) -> None:
        folds = tuple(self.fold_metric_rows)
        metrics = tuple(self.center_metrics)
        inference = tuple(self.equal_center_inference_rows)
        selections = tuple(self.action_selection_rows)
        permutations = tuple(self.permutation_null_summary_rows)
        if tuple(value.target_center for value in metrics) != MIDOGPP_CENTERS:
            raise ProtocolError("Terminal result requires nine center inference units.")
        expected_fold_keys = tuple(
            (center, fold) for center in MIDOGPP_CENTERS for fold in range(5)
        )
        if tuple((row.target_center, row.fold_ordinal) for row in folds) != expected_fold_keys:
            raise ProtocolError("Terminal result requires 45 canonical pooled fold metrics.")
        if not inference or len({row.endpoint for row in inference}) != len(inference):
            raise ProtocolError("Terminal result requires unique equal-center inference rows.")
        for name in (
            "decision_seal_hash",
            "permutation_plan_hash",
            "evaluation_statistics_surface_hash",
            "scientific_result_hash",
        ):
            require_sha256(getattr(self, name), name)
        if (
            self.total_row_count != sum(value.row_count for value in metrics)
            or self.total_case_count != sum(value.case_count for value in metrics)
            or len(permutations) != 1
            or permutations[0].permutation_plan_hash != self.permutation_plan_hash
            or self.decision != TERMINAL_DECISION
            or self.consumed_test_data is not True
            or self.policy_update_authorized is not False
        ):
            raise ProtocolError("Terminal result coverage/claim boundary drifted.")
        expected_selection_keys = tuple(
            (method, action)
            for method in ("G_H", "R")
            for action in (BASELINE_ACTION_ID, *MIDOGPP_CENTERS)
        )
        if (
            tuple((row.method_id, row.action_id) for row in selections)
            != expected_selection_keys
            or any(row.total_case_count != self.total_case_count for row in selections)
            or any(
                sum(row.selection_count for row in selections if row.method_id == method)
                != self.total_case_count
                for method in ("G_H", "R")
            )
        ):
            raise ProtocolError("Terminal action-selection coverage drifted.")
        for row in selections:
            expected_count = sum(
                fold.case_count
                for fold in folds
                if (
                    fold.global_action_id if row.method_id == "G_H" else fold.routed_action_id
                )
                == row.action_id
            )
            if row.selection_count != expected_count:
                raise ProtocolError("Action-selection counts drifted from pooled fold rows.")
        for name in (
            "mean_center_baseline_bacc",
            "mean_center_global_bacc",
            "mean_center_routed_bacc",
            "mean_center_global_minus_baseline",
            "global_minus_baseline_ci95_lower",
            "global_minus_baseline_ci95_upper",
            "mean_center_routed_minus_global",
            "routed_minus_global_ci95_lower",
            "routed_minus_global_ci95_upper",
            "mean_center_routed_minus_baseline",
            "routed_minus_baseline_ci95_lower",
            "routed_minus_baseline_ci95_upper",
            "local_route_coverage",
            "mean_center_normalized_regret",
            "mean_center_top1_accuracy",
            "mean_center_tie_aware_top1_accuracy",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        expected_gmb = center_equal_ci95(tuple(value.global_minus_baseline for value in metrics))
        expected_rmg = center_equal_ci95(tuple(value.routed_minus_global for value in metrics))
        expected_rmb = center_equal_ci95(tuple(value.routed_minus_baseline for value in metrics))
        expected_scalars = {
            "mean_center_baseline_bacc": mean_values(value.baseline_bacc for value in metrics),
            "mean_center_global_bacc": mean_values(value.global_bacc for value in metrics),
            "mean_center_routed_bacc": mean_values(value.routed_bacc for value in metrics),
            "mean_center_global_minus_baseline": expected_gmb[0],
            "global_minus_baseline_ci95_lower": expected_gmb[1],
            "global_minus_baseline_ci95_upper": expected_gmb[2],
            "mean_center_routed_minus_global": expected_rmg[0],
            "routed_minus_global_ci95_lower": expected_rmg[1],
            "routed_minus_global_ci95_upper": expected_rmg[2],
            "mean_center_routed_minus_baseline": expected_rmb[0],
            "routed_minus_baseline_ci95_lower": expected_rmb[1],
            "routed_minus_baseline_ci95_upper": expected_rmb[2],
            "local_route_coverage": sum(
                value.local_route_coverage * value.case_count for value in metrics
            ) / sum(value.case_count for value in metrics),
            "mean_center_normalized_regret": mean_values(
                value.mean_fold_normalized_regret for value in metrics
            ),
            "mean_center_top1_accuracy": mean_values(
                value.mean_fold_top1_accuracy for value in metrics
            ),
            "mean_center_tie_aware_top1_accuracy": mean_values(
                value.mean_fold_tie_aware_top1_accuracy for value in metrics
            ),
        }
        if any(abs(getattr(self, name) - value) > 1.0e-12 for name, value in expected_scalars.items()):
            raise ProtocolError("Terminal aggregate scientific arithmetic drifted.")
        for center in MIDOGPP_CENTERS:
            center_folds = tuple(row for row in folds if row.target_center == center)
            center_metric = metrics[MIDOGPP_CENTERS.index(center)]
            if (
                sum(row.case_count for row in center_folds) != center_metric.case_count
                or sum(row.row_count for row in center_folds) != center_metric.row_count
                or abs(
                    mean_values(row.normalized_regret for row in center_folds)
                    - center_metric.mean_fold_normalized_regret
                ) > 1.0e-12
                or abs(
                    mean_values(row.top1_accuracy for row in center_folds)
                    - center_metric.mean_fold_top1_accuracy
                ) > 1.0e-12
                or abs(
                    mean_values(row.tie_aware_top1_accuracy for row in center_folds)
                    - center_metric.mean_fold_tie_aware_top1_accuracy
                ) > 1.0e-12
            ):
                raise ProtocolError("Fold-to-center descriptive aggregation drifted.")
        expected_inference = equal_center_rows(metrics)
        if tuple(row.to_payload() for row in inference) != tuple(
            row.to_payload() for row in expected_inference
        ):
            raise ProtocolError("Equal-center inference rows drifted from center metrics.")
        observed = permutations[0]
        if (
            abs(observed.observed_center_equal_value - self.mean_center_routed_minus_global)
            > 1.0e-12
            or abs(observed.observed_route_coverage - self.local_route_coverage) > 1.0e-12
        ):
            raise ProtocolError("Permutation observed statistics drifted from the result.")
        if canonical_hash(self._unhashed()) != self.scientific_result_hash:
            raise ProtocolError("Terminal scientific result hash drifted.")
        object.__setattr__(self, "fold_metric_rows", folds)
        object.__setattr__(self, "center_metrics", metrics)
        object.__setattr__(self, "equal_center_inference_rows", inference)
        object.__setattr__(self, "action_selection_rows", selections)
        object.__setattr__(self, "permutation_null_summary_rows", permutations)

    @property
    def center_metric_rows(self) -> tuple[CenterEvaluationMetric, ...]:
        return self.center_metrics

    @property
    def evaluation_exact_surface_hash(self) -> str:
        return self.evaluation_statistics_surface_hash

    def _unhashed(self) -> dict[str, object]:
        return result_payload(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "scientific_result_hash"
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "scientific_result_hash": self.scientific_result_hash}


CeilingEvaluationResult = PooledCeilingEvaluationResult


def center_equal_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != len(MIDOGPP_CENTERS):
        raise ProtocolError("Center-equal inference requires exactly nine centers.")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = STUDENT_T_975_DF8 * math.sqrt(max(0.0, variance) / len(values))
    return mean, mean - half_width, mean + half_width


def mean_values(values) -> float:
    data = tuple(float(value) for value in values)
    return sum(data) / len(data)


def equal_center_rows(
    metrics: Sequence[CenterEvaluationMetric],
) -> tuple[EqualCenterInferenceRow, ...]:
    endpoints = (
        ("B", tuple(value.baseline_bacc for value in metrics)),
        ("G_H", tuple(value.global_bacc for value in metrics)),
        ("R", tuple(value.routed_bacc for value in metrics)),
        ("G_H-B", tuple(value.global_minus_baseline for value in metrics)),
        ("R-G_H", tuple(value.routed_minus_global for value in metrics)),
        ("R-B", tuple(value.routed_minus_baseline for value in metrics)),
        (
            "normalized_regret",
            tuple(value.mean_fold_normalized_regret for value in metrics),
        ),
        ("top1_accuracy", tuple(value.mean_fold_top1_accuracy for value in metrics)),
        (
            "tie_aware_top1_accuracy",
            tuple(value.mean_fold_tie_aware_top1_accuracy for value in metrics),
        ),
    )
    rows: list[EqualCenterInferenceRow] = []
    for endpoint, values in endpoints:
        estimate, lower, upper = center_equal_ci95(values)
        rows.append(
            EqualCenterInferenceRow(
                endpoint=endpoint,
                center_count=len(MIDOGPP_CENTERS),
                estimate=estimate,
                ci95_lower=lower,
                ci95_upper=upper,
            )
        )
    return tuple(rows)


def result_payload(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "fixed_bank_pooled_bacc_case_oof_ceiling_result_v2",
        "fold_metric_rows": [value.to_payload() for value in values["fold_metric_rows"]],
        "center_metrics": [value.to_payload() for value in values["center_metrics"]],
        "equal_center_inference_rows": [
            value.to_payload() for value in values["equal_center_inference_rows"]
        ],
        "action_selection_rows": [
            value.to_payload() for value in values["action_selection_rows"]
        ],
        "permutation_null_summary_rows": [
            value.to_payload() for value in values["permutation_null_summary_rows"]
        ],
        **{
            key: value
            for key, value in values.items()
            if key
            not in {
                "center_metrics",
                "fold_metric_rows",
                "equal_center_inference_rows",
                "action_selection_rows",
                "permutation_null_summary_rows",
            }
        },
        "inference_unit": "target_center_equal_weight_n9",
        "primary_endpoint": "pooled_exact_bacc",
        "per_case_bacc_used": False,
        "evaluation_labels_opened_after_observed_and_null_seals": True,
        "terminal_consumed_test_diagnostic_only": True,
        "result_may_authorize_policy_or_action": False,
        "fresh_evidence": False,
        "diagnostic_only": True,
        "may_feed_later_stage": False,
        "model_update_authorized": False,
    }


__all__ = (
    "ActionSelectionMetricRow",
    "CeilingEvaluationResult",
    "CenterEvaluationMetric",
    "EqualCenterInferenceRow",
    "FoldEvaluationMetric",
    "PermutationNullSummaryRow",
    "PooledCeilingEvaluationResult",
    "center_equal_ci95",
    "equal_center_rows",
    "mean_values",
    "result_payload",
)
