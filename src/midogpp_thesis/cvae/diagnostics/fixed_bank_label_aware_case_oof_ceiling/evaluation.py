"""Terminal row-level OOF evaluation after the complete decision seal."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .core_contracts import BinaryLabelRow, SealedProbabilitySurface
from .core_hashing import canonical_hash, finite, require_sha256
from .decisions import DecisionSeal
from .partitions import CaseOOFPartition
from .permutation_plan import PermutationDecisionPlan
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, action_ids
from .utilities import (
    binary_balanced_accuracy,
    score_evaluation_utilities_after_decision_seal,
    soft_binary_balanced_accuracy,
)


STUDENT_T_975_DF8 = 2.306004135204166


@dataclass(frozen=True, order=True)
class CaseEvaluationMetric:
    target_center: str
    case_id: str
    fold_ordinal: int
    row_count: int
    baseline_action_id: str
    global_action_id: str
    routed_action_id: str
    oracle_action_ids: tuple[str, ...]
    baseline_exact_bacc: float
    global_exact_bacc: float
    routed_exact_bacc: float
    oracle_exact_bacc: float
    global_minus_baseline: float
    routed_minus_global: float
    routed_minus_baseline: float
    oracle_regret: float
    normalized_oracle_gap: float
    exact_top1: float
    tie_aware_top1: float
    route_tier: str
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS or not self.case_id:
            raise ProtocolError("Case metric uses an invalid target/case identity.")
        if self.row_count <= 0 or not self.oracle_action_ids:
            raise ProtocolError("Case metric lacks row/oracle coverage.")
        if any(action not in action_ids(self.target_center) for action in (
            self.baseline_action_id,
            self.global_action_id,
            self.routed_action_id,
            *self.oracle_action_ids,
        )):
            raise ProtocolError("Case metric contains an invalid action.")
        for name in (
            "baseline_exact_bacc",
            "global_exact_bacc",
            "routed_exact_bacc",
            "oracle_exact_bacc",
            "global_minus_baseline",
            "routed_minus_global",
            "routed_minus_baseline",
            "oracle_regret",
            "normalized_oracle_gap",
            "exact_top1",
            "tie_aware_top1",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "oracle_action_ids", tuple(self.oracle_action_ids))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_case_exact_metric_v1",
            **{
                name: (
                    list(self.oracle_action_ids)
                    if name == "oracle_action_ids"
                    else getattr(self, name)
                )
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "descriptive_unit": "whole_case_nested_within_target_center",
            "primary_inference_unit": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True, order=True)
class CenterEvaluationMetric:
    target_center: str
    row_count: int
    case_count: int
    baseline_bacc: float
    global_bacc: float
    routed_bacc: float
    oracle_case_action_bacc: float
    global_minus_baseline: float
    routed_minus_global: float
    routed_minus_baseline: float
    mean_case_oracle_regret: float
    mean_normalized_case_oracle_gap: float
    exact_top1_agreement: float
    tie_aware_top1_agreement: float
    local_route_coverage: float
    metric_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Center metric uses an unknown MIDOG++ target.")
        if self.row_count <= 0 or self.case_count <= 0:
            raise ProtocolError("Center metric lacks row/case coverage.")
        for name in (
            "baseline_bacc",
            "global_bacc",
            "routed_bacc",
            "oracle_case_action_bacc",
            "global_minus_baseline",
            "routed_minus_global",
            "routed_minus_baseline",
            "mean_case_oracle_regret",
            "mean_normalized_case_oracle_gap",
            "exact_top1_agreement",
            "tie_aware_top1_agreement",
            "local_route_coverage",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "metric_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_center_exact_metric_v1",
            "target_center": self.target_center,
            "row_count": self.row_count,
            "case_count": self.case_count,
            "baseline_bacc": self.baseline_bacc,
            "global_bacc": self.global_bacc,
            "routed_bacc": self.routed_bacc,
            "oracle_case_action_bacc": self.oracle_case_action_bacc,
            "global_minus_baseline": self.global_minus_baseline,
            "routed_minus_global": self.routed_minus_global,
            "routed_minus_baseline": self.routed_minus_baseline,
            "mean_case_oracle_regret": self.mean_case_oracle_regret,
            "mean_normalized_case_oracle_gap": self.mean_normalized_case_oracle_gap,
            "exact_top1_agreement": self.exact_top1_agreement,
            "tie_aware_top1_agreement": self.tie_aware_top1_agreement,
            "local_route_coverage": self.local_route_coverage,
            "primary_endpoint": "thresholded_row_level_bacc",
            "case_mean_bacc_is_primary": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metric_hash": self.metric_hash}


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
            "two_sided_p_value",
            "observed_route_coverage",
            "null_mean_route_coverage",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_permutation_null_summary_v1",
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "row_hash"
            },
            "blocking": "candidate_source_labels_within_H_fold_support_case",
            "center_equal_inference": True,
            "null_actions_sealed_before_evaluation_labels": True,
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
            "schema_version": "fixed_bank_label_aware_action_selection_metric_v1",
            "method_id": self.method_id,
            "action_id": self.action_id,
            "selection_count": self.selection_count,
            "total_case_count": self.total_case_count,
            "selection_share": self.selection_share,
            "selection_unit": "whole_oof_case",
            "target_centers_equal_weight_only_for_primary_utility_inference": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class CeilingEvaluationResult:
    case_metric_rows: tuple[CaseEvaluationMetric, ...]
    center_metrics: tuple[CenterEvaluationMetric, ...]
    action_selection_rows: tuple[ActionSelectionMetricRow, ...]
    permutation_null_summary_rows: tuple[PermutationNullSummaryRow, ...]
    decision_seal_hash: str
    permutation_plan_hash: str
    evaluation_exact_surface_hash: str
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
    mean_exact_top1_agreement: float
    mean_tie_aware_top1_agreement: float
    mean_normalized_case_oracle_gap: float
    local_route_coverage: float
    scientific_result_hash: str
    consumed_test_data: bool = True
    policy_update_authorized: bool = False

    def __post_init__(self) -> None:
        cases = tuple(self.case_metric_rows)
        metrics = tuple(self.center_metrics)
        selections = tuple(self.action_selection_rows)
        permutations = tuple(self.permutation_null_summary_rows)
        if tuple(value.target_center for value in metrics) != MIDOGPP_CENTERS:
            raise ProtocolError("Terminal result requires nine center inference units.")
        for name in (
            "decision_seal_hash",
            "permutation_plan_hash",
            "evaluation_exact_surface_hash",
            "scientific_result_hash",
        ):
            require_sha256(getattr(self, name), name)
        if (
            not cases
            or len(cases) != self.total_case_count
            or self.total_row_count != sum(value.row_count for value in metrics)
            or self.total_case_count != sum(value.case_count for value in metrics)
            or len(permutations) != 1
            or permutations[0].permutation_plan_hash != self.permutation_plan_hash
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
            "mean_exact_top1_agreement",
            "mean_tie_aware_top1_agreement",
            "mean_normalized_case_oracle_gap",
            "local_route_coverage",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if canonical_hash(self._unhashed()) != self.scientific_result_hash:
            raise ProtocolError("Terminal scientific result hash drifted.")
        object.__setattr__(self, "case_metric_rows", cases)
        object.__setattr__(self, "center_metrics", metrics)
        object.__setattr__(self, "action_selection_rows", selections)
        object.__setattr__(self, "permutation_null_summary_rows", permutations)

    @property
    def center_metric_rows(self) -> tuple[CenterEvaluationMetric, ...]:
        return self.center_metrics

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_label_aware_case_oof_ceiling_result_v1",
            "case_metric_rows": [value.to_payload() for value in self.case_metric_rows],
            "center_metrics": [value.to_payload() for value in self.center_metrics],
            "action_selection_rows": [
                value.to_payload() for value in self.action_selection_rows
            ],
            "permutation_null_summary_rows": [
                value.to_payload() for value in self.permutation_null_summary_rows
            ],
            "decision_seal_hash": self.decision_seal_hash,
            "permutation_plan_hash": self.permutation_plan_hash,
            "evaluation_exact_surface_hash": self.evaluation_exact_surface_hash,
            "total_row_count": self.total_row_count,
            "total_case_count": self.total_case_count,
            "mean_center_baseline_bacc": self.mean_center_baseline_bacc,
            "mean_center_global_bacc": self.mean_center_global_bacc,
            "mean_center_routed_bacc": self.mean_center_routed_bacc,
            "mean_center_global_minus_baseline": self.mean_center_global_minus_baseline,
            "global_minus_baseline_ci95_lower": self.global_minus_baseline_ci95_lower,
            "global_minus_baseline_ci95_upper": self.global_minus_baseline_ci95_upper,
            "mean_center_routed_minus_global": self.mean_center_routed_minus_global,
            "routed_minus_global_ci95_lower": self.routed_minus_global_ci95_lower,
            "routed_minus_global_ci95_upper": self.routed_minus_global_ci95_upper,
            "mean_center_routed_minus_baseline": self.mean_center_routed_minus_baseline,
            "routed_minus_baseline_ci95_lower": self.routed_minus_baseline_ci95_lower,
            "routed_minus_baseline_ci95_upper": self.routed_minus_baseline_ci95_upper,
            "mean_exact_top1_agreement": self.mean_exact_top1_agreement,
            "mean_tie_aware_top1_agreement": self.mean_tie_aware_top1_agreement,
            "mean_normalized_case_oracle_gap": self.mean_normalized_case_oracle_gap,
            "local_route_coverage": self.local_route_coverage,
            "inference_unit": "target_center_equal_weight_n9",
            "primary_endpoint": "thresholded_row_level_bacc",
            "smooth_metrics_in_scientific_identity": False,
            "evaluation_labels_opened_after_all_decisions": True,
            "terminal_consumed_test_diagnostic_only": True,
            "result_may_authorize_policy_or_action": False,
            "consumed_test_data": True,
            "fresh_evidence": False,
            "diagnostic_only": True,
            "policy_update_authorized": False,
            "may_feed_later_stage": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "scientific_result_hash": self.scientific_result_hash}


@dataclass(frozen=True, order=True)
class CenterSmoothMetric:
    target_center: str
    baseline_smooth_bacc: float
    global_smooth_bacc: float
    routed_smooth_bacc: float

    def __post_init__(self) -> None:
        if self.target_center not in MIDOGPP_CENTERS:
            raise ProtocolError("Smooth metric uses an unknown target.")
        for name in ("baseline_smooth_bacc", "global_smooth_bacc", "routed_smooth_bacc"):
            object.__setattr__(self, name, finite(getattr(self, name), name))


@dataclass(frozen=True)
class SmoothDescriptiveResult:
    center_metrics: tuple[CenterSmoothMetric, ...]
    decision_seal_hash: str
    descriptive_hash: str
    exact_scientific_result_hash: str
    decision_influence: bool = False
    publication_gate_eligible: bool = False

    def __post_init__(self) -> None:
        if tuple(value.target_center for value in self.center_metrics) != MIDOGPP_CENTERS:
            raise ProtocolError("Smooth descriptive result requires all centers.")
        for name in ("decision_seal_hash", "descriptive_hash", "exact_scientific_result_hash"):
            require_sha256(getattr(self, name), name)
        if self.decision_influence is not False or self.publication_gate_eligible is not False:
            raise ProtocolError("Smooth response cannot influence the route or gate.")


def evaluate_decision_seal(
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    permutation_plan: PermutationDecisionPlan,
    tie_tolerance: float = 1.0e-12,
    confidence_level: float = 0.95,
) -> CeilingEvaluationResult:
    """Evaluate B/G/R on pooled held rows per center, then average centers."""

    _validate_terminal_inputs(decision_seal, partition, probabilities)
    if (
        permutation_plan.partition_hash != partition.partition_hash
        or permutation_plan.probability_surface_hash != probabilities.surface_hash
        or permutation_plan.sealed_before_evaluation_labels is not True
    ):
        raise ProtocolError("Terminal evaluation requires the matching pre-eval null seal.")
    if abs(float(confidence_level) - 0.95) > 1.0e-12:
        raise ProtocolError("Only the predeclared 95% confidence level is allowed.")
    utility = score_evaluation_utilities_after_decision_seal(
        probabilities, labels, decision_seal=decision_seal
    )
    label_map = _terminal_label_map(probabilities, labels)
    probability_map = probabilities.probabilities()
    utility_map = utility.by_key()
    metrics: list[CenterEvaluationMetric] = []
    case_metrics: list[CaseEvaluationMetric] = []
    for center in MIDOGPP_CENTERS:
        center_identities = tuple(
            identity for identity in probabilities.identities if identity.target_center == center
        )
        truth: list[int] = []
        baseline_predictions: list[int] = []
        global_predictions: list[int] = []
        routed_predictions: list[int] = []
        oracle_predictions: list[int] = []
        exact_top1: list[float] = []
        tie_top1: list[float] = []
        regrets: list[float] = []
        normalized_gaps: list[float] = []
        local_routes = 0
        cases = sorted({identity.case_id for identity in center_identities})
        oracle_action_by_case: dict[str, str] = {}
        decision_by_case = {
            case_id: decision_seal.decision(
                center, partition.evaluation_fold_for_case(center, case_id).fold_ordinal
            )
            for case_id in cases
        }
        for case_id in cases:
            decision = decision_by_case[case_id]
            if decision.route_tier == "R":
                local_routes += 1
            action_scores = {
                action: utility_map[(center, case_id, action)].exact_bacc
                for action in action_ids(center)
            }
            oracle_value = max(action_scores.values())
            oracle_actions = tuple(
                action for action in action_ids(center)
                if abs(action_scores[action] - oracle_value) <= tie_tolerance
            )
            oracle_action = oracle_actions[0]
            oracle_action_by_case[case_id] = oracle_action
            selected_value = action_scores[decision.routed_action_id]
            baseline_value = action_scores[BASELINE_ACTION_ID]
            exact_top1.append(float(decision.routed_action_id == oracle_action))
            tie_top1.append(float(decision.routed_action_id in oracle_actions))
            regrets.append(oracle_value - selected_value)
            available_gain = oracle_value - baseline_value
            normalized_gaps.append(
                0.0
                if available_gain <= tie_tolerance
                else (oracle_value - selected_value) / available_gain
            )
            row_count = sum(
                identity.case_id == case_id for identity in center_identities
            )
            case_metrics.append(
                CaseEvaluationMetric(
                    target_center=center,
                    case_id=case_id,
                    fold_ordinal=decision.fold_ordinal,
                    row_count=row_count,
                    baseline_action_id=BASELINE_ACTION_ID,
                    global_action_id=decision.global_action_id,
                    routed_action_id=decision.routed_action_id,
                    oracle_action_ids=oracle_actions,
                    baseline_exact_bacc=baseline_value,
                    global_exact_bacc=action_scores[decision.global_action_id],
                    routed_exact_bacc=selected_value,
                    oracle_exact_bacc=oracle_value,
                    global_minus_baseline=(
                        action_scores[decision.global_action_id] - baseline_value
                    ),
                    routed_minus_global=(
                        selected_value - action_scores[decision.global_action_id]
                    ),
                    routed_minus_baseline=selected_value - baseline_value,
                    oracle_regret=oracle_value - selected_value,
                    normalized_oracle_gap=normalized_gaps[-1],
                    exact_top1=exact_top1[-1],
                    tie_aware_top1=tie_top1[-1],
                    route_tier=decision.route_tier,
                )
            )
        for identity in center_identities:
            decision = decision_by_case[identity.case_id]
            key = (center, identity.case_id, identity.sample_id)
            truth.append(label_map[key])
            baseline_predictions.append(int(probability_map[(*key, BASELINE_ACTION_ID)] >= 0.5))
            global_predictions.append(int(probability_map[(*key, decision.global_action_id)] >= 0.5))
            routed_predictions.append(int(probability_map[(*key, decision.routed_action_id)] >= 0.5))
            oracle_predictions.append(int(probability_map[(*key, oracle_action_by_case[identity.case_id])] >= 0.5))
        baseline_bacc = binary_balanced_accuracy(truth, baseline_predictions)
        global_bacc = binary_balanced_accuracy(truth, global_predictions)
        routed_bacc = binary_balanced_accuracy(truth, routed_predictions)
        metrics.append(
            CenterEvaluationMetric(
                target_center=center,
                row_count=len(center_identities),
                case_count=len(cases),
                baseline_bacc=baseline_bacc,
                global_bacc=global_bacc,
                routed_bacc=routed_bacc,
                oracle_case_action_bacc=binary_balanced_accuracy(truth, oracle_predictions),
                global_minus_baseline=global_bacc - baseline_bacc,
                routed_minus_global=routed_bacc - global_bacc,
                routed_minus_baseline=routed_bacc - baseline_bacc,
                mean_case_oracle_regret=sum(regrets) / len(regrets),
                mean_normalized_case_oracle_gap=sum(normalized_gaps) / len(normalized_gaps),
                exact_top1_agreement=sum(exact_top1) / len(exact_top1),
                tie_aware_top1_agreement=sum(tie_top1) / len(tie_top1),
                local_route_coverage=local_routes / len(cases),
            )
        )
    canonical = tuple(metrics)
    canonical_cases = tuple(case_metrics)
    action_universe = (BASELINE_ACTION_ID, *MIDOGPP_CENTERS)
    action_selection_rows = tuple(
        ActionSelectionMetricRow(
            method_id=method,
            action_id=action,
            selection_count=sum(
                (
                    row.global_action_id if method == "G_H" else row.routed_action_id
                )
                == action
                for row in canonical_cases
            ),
            total_case_count=len(canonical_cases),
            selection_share=(
                sum(
                    (
                        row.global_action_id
                        if method == "G_H"
                        else row.routed_action_id
                    )
                    == action
                    for row in canonical_cases
                )
                / len(canonical_cases)
            ),
        )
        for method in ("G_H", "R")
        for action in action_universe
    )
    gmb = tuple(value.global_minus_baseline for value in canonical)
    rmg = tuple(value.routed_minus_global for value in canonical)
    rmb = tuple(value.routed_minus_baseline for value in canonical)
    gmb_mean, gmb_lower, gmb_upper = _center_equal_ci95(gmb)
    rmg_mean, rmg_lower, rmg_upper = _center_equal_ci95(rmg)
    rmb_mean, rmb_lower, rmb_upper = _center_equal_ci95(rmb)
    permutation_summary = _score_permutation_null(
        permutation_plan=permutation_plan,
        decision_seal=decision_seal,
        partition=partition,
        probabilities=probabilities,
        label_map=label_map,
        utility_map=utility_map,
        observed_center_equal_r_minus_g=rmg_mean,
        observed_route_coverage=(
            sum(value.local_route_coverage * value.case_count for value in canonical)
            / sum(value.case_count for value in canonical)
        ),
    )
    values = {
        "case_metric_rows": canonical_cases,
        "center_metrics": canonical,
        "action_selection_rows": action_selection_rows,
        "permutation_null_summary_rows": (permutation_summary,),
        "decision_seal_hash": decision_seal.decision_seal_hash,
        "permutation_plan_hash": permutation_plan.plan_hash,
        "evaluation_exact_surface_hash": utility.exact_surface_hash,
        "total_row_count": sum(value.row_count for value in canonical),
        "total_case_count": sum(value.case_count for value in canonical),
        "mean_center_baseline_bacc": _mean(value.baseline_bacc for value in canonical),
        "mean_center_global_bacc": _mean(value.global_bacc for value in canonical),
        "mean_center_routed_bacc": _mean(value.routed_bacc for value in canonical),
        "mean_center_global_minus_baseline": gmb_mean,
        "global_minus_baseline_ci95_lower": gmb_lower,
        "global_minus_baseline_ci95_upper": gmb_upper,
        "mean_center_routed_minus_global": rmg_mean,
        "routed_minus_global_ci95_lower": rmg_lower,
        "routed_minus_global_ci95_upper": rmg_upper,
        "mean_center_routed_minus_baseline": rmb_mean,
        "routed_minus_baseline_ci95_lower": rmb_lower,
        "routed_minus_baseline_ci95_upper": rmb_upper,
        "mean_exact_top1_agreement": _mean(value.exact_top1_agreement for value in canonical),
        "mean_tie_aware_top1_agreement": _mean(value.tie_aware_top1_agreement for value in canonical),
        "mean_normalized_case_oracle_gap": _mean(value.mean_normalized_case_oracle_gap for value in canonical),
        "local_route_coverage": sum(
            value.local_route_coverage * value.case_count for value in canonical
        ) / sum(value.case_count for value in canonical),
    }
    unhashed = _result_unhashed(values)
    return CeilingEvaluationResult(
        **values,
        scientific_result_hash=canonical_hash(unhashed),
    )


def evaluate_smooth_descriptive(
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    exact_result: CeilingEvaluationResult,
) -> SmoothDescriptiveResult:
    """Report probability-continuous BACC without entering decision identity."""

    _validate_terminal_inputs(decision_seal, partition, probabilities)
    if exact_result.decision_seal_hash != decision_seal.decision_seal_hash:
        raise ProtocolError("Smooth report must bind to the exact scientific result.")
    label_map = _terminal_label_map(probabilities, labels)
    probability_map = probabilities.probabilities()
    metrics: list[CenterSmoothMetric] = []
    for center in MIDOGPP_CENTERS:
        identities = tuple(
            identity for identity in probabilities.identities if identity.target_center == center
        )
        truth: list[int] = []
        baseline: list[float] = []
        global_values: list[float] = []
        routed: list[float] = []
        for identity in identities:
            fold = partition.evaluation_fold_for_case(center, identity.case_id)
            decision = decision_seal.decision(center, fold.fold_ordinal)
            key = (center, identity.case_id, identity.sample_id)
            truth.append(label_map[key])
            baseline.append(probability_map[(*key, BASELINE_ACTION_ID)])
            global_values.append(probability_map[(*key, decision.global_action_id)])
            routed.append(probability_map[(*key, decision.routed_action_id)])
        metrics.append(
            CenterSmoothMetric(
                target_center=center,
                baseline_smooth_bacc=soft_binary_balanced_accuracy(truth, baseline),
                global_smooth_bacc=soft_binary_balanced_accuracy(truth, global_values),
                routed_smooth_bacc=soft_binary_balanced_accuracy(truth, routed),
            )
        )
    payload = {
        "schema_version": "fixed_bank_label_aware_smooth_descriptive_v1",
        "center_metrics": [value.__dict__ for value in metrics],
        "decision_seal_hash": decision_seal.decision_seal_hash,
        "exact_scientific_result_hash": exact_result.scientific_result_hash,
        "decision_influence": False,
        "publication_gate_eligible": False,
    }
    return SmoothDescriptiveResult(
        center_metrics=tuple(metrics),
        decision_seal_hash=decision_seal.decision_seal_hash,
        descriptive_hash=canonical_hash(payload),
        exact_scientific_result_hash=exact_result.scientific_result_hash,
    )


def _validate_terminal_inputs(
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
) -> None:
    if (
        decision_seal.partition_hash != partition.partition_hash
        or decision_seal.probability_surface_hash != probabilities.surface_hash
        or decision_seal.all_fold_decisions_sealed_before_evaluation_labels is not True
    ):
        raise ProtocolError("Terminal evaluation inputs are not bound to one sealed run.")


def _terminal_label_map(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
) -> Mapping[tuple[str, str, str], int]:
    expected = {
        (identity.target_center, identity.case_id, identity.sample_id)
        for identity in probabilities.identities
    }
    observed: dict[tuple[str, str, str], int] = {}
    for row in labels:
        key = (row.target_center, row.case_id, row.sample_id)
        if key not in expected:
            continue
        if key in observed:
            raise ProtocolError("Duplicate evaluation label row detected.")
        observed[key] = row.label
    if set(observed) != expected:
        raise ProtocolError("Evaluation labels do not cover the sealed probability rows.")
    return observed


def _score_permutation_null(
    *,
    permutation_plan: PermutationDecisionPlan,
    decision_seal: DecisionSeal,
    partition: CaseOOFPartition,
    probabilities: SealedProbabilitySurface,
    label_map: Mapping[tuple[str, str, str], int],
    utility_map: Mapping[tuple[str, str, str], object],
    observed_center_equal_r_minus_g: float,
    observed_route_coverage: float,
) -> PermutationNullSummaryRow:
    """Stream the compact presealed null actions through held-row scoring."""

    del utility_map  # Bound by evaluation exact hash; regret is reported on observed case rows.
    probability_map = probabilities.probabilities()
    null_center_contrasts = np.empty(
        (permutation_plan.permutation_count, len(MIDOGPP_CENTERS)),
        dtype=np.float64,
    )
    global_codes = np.empty(len(permutation_plan.fold_keys), dtype=np.uint8)
    fold_case_weights = np.empty(len(permutation_plan.fold_keys), dtype=np.float64)
    for column, (center, fold_ordinal) in enumerate(permutation_plan.fold_keys):
        decision = decision_seal.decision(center, fold_ordinal)
        global_codes[column] = action_ids(center).index(decision.global_action_id)
        fold_case_weights[column] = len(
            partition.fold(center, fold_ordinal).evaluation_case_ids
        )
    for center_index, center in enumerate(MIDOGPP_CENTERS):
        identities = tuple(
            identity
            for identity in probabilities.identities
            if identity.target_center == center
        )
        truth = np.asarray(
            [label_map[(center, row.case_id, row.sample_id)] for row in identities],
            dtype=np.uint8,
        )
        if not np.any(truth == 0) or not np.any(truth == 1):
            raise ProtocolError("Permutation BACC requires both classes per target center.")
        actions = action_ids(center)
        predicted = np.asarray(
            [
                [
                    probability_map[(center, row.case_id, row.sample_id, action)]
                    >= 0.5
                    for action in actions
                ]
                for row in identities
            ],
            dtype=np.uint8,
        )
        fold_columns = np.asarray(
            [
                center_index * 5
                + partition.evaluation_fold_for_case(center, row.case_id).fold_ordinal
                for row in identities
            ],
            dtype=np.int64,
        )
        selected_codes = permutation_plan.action_codes[:, fold_columns]
        row_indices = np.arange(len(identities), dtype=np.int64)[None, :]
        selected_predictions = predicted[row_indices, selected_codes]
        positive = truth == 1
        negative = truth == 0
        null_bacc = 0.5 * (
            selected_predictions[:, positive].mean(axis=1)
            + (1.0 - selected_predictions[:, negative]).mean(axis=1)
        )
        global_prediction = np.asarray(
            [
                predicted[row_index, global_codes[fold_columns[row_index]]]
                for row_index in range(len(identities))
            ],
            dtype=np.uint8,
        )
        global_bacc = 0.5 * (
            global_prediction[positive].mean()
            + (1.0 - global_prediction[negative]).mean()
        )
        null_center_contrasts[:, center_index] = null_bacc - global_bacc
    null_values = null_center_contrasts.mean(axis=1)
    weighted_switches = (
        permutation_plan.action_codes != global_codes[None, :]
    ) * fold_case_weights[None, :]
    null_coverage = weighted_switches.sum(axis=1) / fold_case_weights.sum()
    count = permutation_plan.permutation_count
    null_mean = float(null_values.mean())
    null_sd = float(null_values.std(ddof=1)) if count > 1 else 0.0
    one_sided = (1.0 + float(np.count_nonzero(null_values >= observed_center_equal_r_minus_g))) / (count + 1.0)
    two_sided = (1.0 + float(np.count_nonzero(np.abs(null_values) >= abs(observed_center_equal_r_minus_g)))) / (count + 1.0)
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
        two_sided_p_value=min(1.0, two_sided),
        observed_route_coverage=observed_route_coverage,
        null_mean_route_coverage=float(null_coverage.mean()),
        permutation_plan_hash=permutation_plan.plan_hash,
    )


def _center_equal_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != len(MIDOGPP_CENTERS):
        raise ProtocolError("Center-equal inference requires exactly nine centers.")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = STUDENT_T_975_DF8 * math.sqrt(max(0.0, variance) / len(values))
    return mean, mean - half_width, mean + half_width


def _mean(values) -> float:
    data = tuple(float(value) for value in values)
    return sum(data) / len(data)


def _result_unhashed(values: Mapping[str, object]) -> dict[str, object]:
    cases = values["case_metric_rows"]
    metrics = values["center_metrics"]
    selections = values["action_selection_rows"]
    permutations = values["permutation_null_summary_rows"]
    return {
        "schema_version": "fixed_bank_label_aware_case_oof_ceiling_result_v1",
        "case_metric_rows": [value.to_payload() for value in cases],
        "center_metrics": [value.to_payload() for value in metrics],
        "action_selection_rows": [value.to_payload() for value in selections],
        "permutation_null_summary_rows": [
            value.to_payload() for value in permutations
        ],
        **{
            key: value
            for key, value in values.items()
            if key
            not in {
                "case_metric_rows",
                "center_metrics",
                "action_selection_rows",
                "permutation_null_summary_rows",
            }
        },
        "inference_unit": "target_center_equal_weight_n9",
        "primary_endpoint": "thresholded_row_level_bacc",
        "smooth_metrics_in_scientific_identity": False,
        "evaluation_labels_opened_after_all_decisions": True,
        "terminal_consumed_test_diagnostic_only": True,
        "result_may_authorize_policy_or_action": False,
        "consumed_test_data": True,
        "fresh_evidence": False,
        "diagnostic_only": True,
        "policy_update_authorized": False,
        "may_feed_later_stage": False,
    }


__all__ = (
    "ActionSelectionMetricRow",
    "CaseEvaluationMetric",
    "CeilingEvaluationResult",
    "CenterEvaluationMetric",
    "CenterSmoothMetric",
    "PermutationNullSummaryRow",
    "SmoothDescriptiveResult",
    "evaluate_decision_seal",
    "evaluate_smooth_descriptive",
)
