"""Hash-verifying semantic reconstruction of persisted v2 scientific payloads."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .case_partitions import CaseFold, CaseOOFPartition
from .core_contracts import (
    AggregatedProbabilityRow,
    CaseActionSufficientStatistics,
    CaseIdentityRow,
    SealedProbabilitySurface,
    SufficientStatisticSurface,
)
from .decisions import DecisionSeal, FoldDecision
from .evaluation_contracts import (
    ActionSelectionMetricRow,
    CenterEvaluationMetric,
    EqualCenterInferenceRow,
    FoldEvaluationMetric,
    PermutationNullSummaryRow,
    PooledCeilingEvaluationResult,
)
from .permutation_plan import PermutationDecisionPlan
from .pooled_posterior import CandidatePosteriorEstimate, PooledFoldPosterior, PosteriorConfig
from .pooled_prior import (
    CandidateGlobalEstimate,
    PairwisePriorEstimate,
    PooledLocoPrior,
    PriorConfig,
)


def probability_surface_from_payload(
    payload: Mapping[str, object],
) -> SealedProbabilitySurface:
    rows = tuple(_aggregated_probability_row(value) for value in _sequence(payload, "rows"))
    result = SealedProbabilitySurface(
        rows=rows,
        probability_store_hash=str(payload.get("probability_store_hash", "")),
        surface_hash=str(payload.get("surface_hash", "")),
        predictions_globally_sealed_before_labels=_bool(
            payload, "predictions_globally_sealed_before_labels"
        ),
        labels_readable_during_materialization=_bool(
            payload, "labels_readable_during_materialization"
        ),
    )
    _assert_expected_fields(payload, result.to_payload(), "probability surface")
    return result


def statistics_surface_from_payload(
    payload: Mapping[str, object],
) -> SufficientStatisticSurface:
    rows = tuple(_statistic_row(value) for value in _sequence(payload, "rows"))
    allowed = tuple(
        (str(_sequence_value(value, 0)), str(_sequence_value(value, 1)))
        for value in _sequence(payload, "allowed_case_keys")
    )
    result = SufficientStatisticSurface(
        rows=rows,
        allowed_case_keys=allowed,
        label_scope=str(payload.get("label_scope", "")),
        prerequisite_seal_hash=str(payload.get("prerequisite_seal_hash", "")),
        statistics_surface_hash=str(payload.get("statistics_surface_hash", "")),
        hard_threshold=float(payload.get("hard_threshold", 0.5)),
    )
    _assert_expected_fields(payload, result.to_payload(), "statistics surface")
    return result


def partition_from_payload(payload: Mapping[str, object]) -> CaseOOFPartition:
    identities = tuple(
        CaseIdentityRow(
            target_center=str(value.get("target_center", "")),
            case_id=str(value.get("case_id", "")),
            sample_id=str(value.get("sample_id", "")),
        )
        for value in _mapping_sequence(payload, "identities")
    )
    folds = tuple(_case_fold(value) for value in _mapping_sequence(payload, "folds"))
    result = CaseOOFPartition(
        identities=identities,
        folds=folds,
        partition_seed=int(payload.get("partition_seed", -1)),
        partition_hash=str(payload.get("partition_hash", "")),
        fold_count=int(payload.get("fold_count", -1)),
    )
    _assert_expected_fields(payload, result.to_payload(), "case partition")
    return result


def pooled_loco_prior_from_payload(payload: Mapping[str, object]) -> PooledLocoPrior:
    config_payload = _mapping(payload.get("config"), "prior config")
    config = PriorConfig(
        variance_floor=float(config_payload.get("variance_floor")),
        confidence_multiplier=float(config_payload.get("confidence_multiplier")),
        minimum_gain=float(config_payload.get("minimum_gain")),
        tie_tolerance=float(config_payload.get("tie_tolerance")),
    )
    result = PooledLocoPrior(
        target_center=str(payload.get("target_center", "")),
        candidate_estimates=tuple(
            _candidate_global_estimate(value)
            for value in _mapping_sequence(payload, "candidate_estimates")
        ),
        pairwise_estimates=tuple(
            _pairwise_prior_estimate(value)
            for value in _mapping_sequence(payload, "pairwise_estimates")
        ),
        global_action_id=str(payload.get("global_action_id", "")),
        best_candidate_action_id=str(payload.get("best_candidate_action_id", "")),
        source_statistics_surface_hash=str(
            payload.get("source_statistics_surface_hash", "")
        ),
        probability_surface_hash=str(payload.get("probability_surface_hash", "")),
        config=config,
        prior_hash=str(payload.get("prior_hash", "")),
        sealed_before_h_support_access=bool(
            payload.get("G_H_sealed_before_H_support_access", False)
        ),
        h_labels_used_in_g_h=bool(payload.get("H_labels_used_in_G_H", True)),
    )
    _assert_expected_fields(payload, result.to_payload(), "pooled LOCO prior")
    return result


def pooled_fold_posterior_from_payload(
    payload: Mapping[str, object],
) -> PooledFoldPosterior:
    config_payload = _mapping(payload.get("config"), "posterior config")
    config = PosteriorConfig(
        variance_floor=float(config_payload.get("variance_floor")),
        confidence_multiplier=float(config_payload.get("confidence_multiplier")),
        minimum_gain=float(config_payload.get("minimum_gain")),
    )
    result = PooledFoldPosterior(
        target_center=str(payload.get("target_center", "")),
        fold_ordinal=int(payload.get("fold_ordinal", -1)),
        fold_hash=str(payload.get("fold_hash", "")),
        global_prior_hash=str(payload.get("global_prior_hash", "")),
        support_statistics_surface_hash=str(
            payload.get("support_statistics_surface_hash", "")
        ),
        support_case_ids=tuple(str(value) for value in _sequence(payload, "support_case_ids")),
        global_action_id=str(payload.get("global_action_id", "")),
        estimates=tuple(
            _posterior_estimate(value) for value in _mapping_sequence(payload, "estimates")
        ),
        config=config,
        posterior_hash=str(payload.get("posterior_hash", "")),
        evaluation_labels_used=bool(payload.get("evaluation_labels_used", True)),
    )
    _assert_expected_fields(payload, result.to_payload(), "pooled fold posterior")
    return result


def fold_decision_from_payload(payload: Mapping[str, object]) -> FoldDecision:
    result = FoldDecision(
        target_center=str(payload.get("target_center", "")),
        fold_ordinal=int(payload.get("fold_ordinal", -1)),
        fold_hash=str(payload.get("fold_hash", "")),
        evaluation_case_ids=tuple(
            str(value) for value in _sequence(payload, "evaluation_case_ids")
        ),
        baseline_action_id=str(payload.get("baseline_action_id", "")),
        global_action_id=str(payload.get("global_action_id", "")),
        selected_challenger_action_id=str(
            payload.get("selected_challenger_action_id", "")
        ),
        routed_action_id=str(payload.get("routed_action_id", "")),
        route_tier=str(payload.get("route_tier", "")),
        selected_posterior_mean_gain_vs_g=float(
            payload.get("selected_posterior_mean_gain_vs_g")
        ),
        selected_lower_confidence_bound=float(
            payload.get("selected_lower_confidence_bound")
        ),
        global_lower_confidence_bound_vs_b=float(
            payload.get("global_lower_confidence_bound_vs_b")
        ),
        global_prior_hash=str(payload.get("global_prior_hash", "")),
        posterior_hash=str(payload.get("posterior_hash", "")),
        evaluation_labels_used=bool(payload.get("evaluation_labels_used", True)),
    )
    _assert_expected_fields(payload, result.to_payload(), "fold decision")
    return result


def decision_seal_from_payload(payload: Mapping[str, object]) -> DecisionSeal:
    result = DecisionSeal(
        decisions=tuple(
            fold_decision_from_payload(value)
            for value in _mapping_sequence(payload, "decisions")
        ),
        partition_hash=str(payload.get("partition_hash", "")),
        probability_surface_hash=str(payload.get("probability_surface_hash", "")),
        decision_seal_hash=str(payload.get("decision_seal_hash", "")),
        all_fold_decisions_sealed_before_evaluation_labels=bool(
            payload.get("all_fold_decisions_sealed_before_evaluation_labels", False)
        ),
        fold_decision_count=int(payload.get("fold_decision_count", -1)),
    )
    _assert_expected_fields(payload, result.to_payload(), "decision seal")
    return result


def permutation_plan_from_payload(
    payload: Mapping[str, object], action_codes: np.ndarray
) -> PermutationDecisionPlan:
    result = PermutationDecisionPlan(
        action_codes=np.asarray(action_codes),
        permutation_seed=int(payload.get("permutation_seed", -1)),
        permutation_count=int(payload.get("permutation_count", -1)),
        fold_keys=tuple(
            (str(_sequence_value(value, 0)), int(_sequence_value(value, 1)))
            for value in _sequence(payload, "fold_keys")
        ),
        partition_hash=str(payload.get("partition_hash", "")),
        probability_surface_hash=str(payload.get("probability_surface_hash", "")),
        support_input_hash=str(payload.get("support_input_hash", "")),
        plan_hash=str(payload.get("plan_hash", "")),
        sealed_before_evaluation_labels=bool(
            payload.get("sealed_before_evaluation_labels", False)
        ),
        evaluation_labels_used_to_generate_actions=bool(
            payload.get("evaluation_labels_used_to_generate_actions", True)
        ),
    )
    _assert_expected_fields(payload, result.to_payload(), "permutation plan")
    return result


def evaluation_result_from_payload(
    payload: Mapping[str, object],
) -> PooledCeilingEvaluationResult:
    result = PooledCeilingEvaluationResult(
        fold_metric_rows=tuple(
            _fold_metric(value) for value in _mapping_sequence(payload, "fold_metric_rows")
        ),
        center_metrics=tuple(
            _center_metric(value) for value in _mapping_sequence(payload, "center_metrics")
        ),
        equal_center_inference_rows=tuple(
            _inference_row(value)
            for value in _mapping_sequence(payload, "equal_center_inference_rows")
        ),
        action_selection_rows=tuple(
            _action_selection_row(value)
            for value in _mapping_sequence(payload, "action_selection_rows")
        ),
        permutation_null_summary_rows=tuple(
            _permutation_summary(value)
            for value in _mapping_sequence(payload, "permutation_null_summary_rows")
        ),
        **{
            name: payload.get(name)
            for name in PooledCeilingEvaluationResult.__dataclass_fields__
            if name
            not in {
                "fold_metric_rows",
                "center_metrics",
                "equal_center_inference_rows",
                "action_selection_rows",
                "permutation_null_summary_rows",
            }
        },
    )
    _assert_expected_fields(payload, result.to_payload(), "evaluation result")
    return result


def _aggregated_probability_row(value: object) -> AggregatedProbabilityRow:
    payload = _mapping(value, "aggregated probability row")
    result = AggregatedProbabilityRow(
        target_center=str(payload.get("target_center", "")),
        case_id=str(payload.get("case_id", "")),
        sample_id=str(payload.get("sample_id", "")),
        action_id=str(payload.get("action_id", "")),
        probability_mean=float(payload.get("probability_mean")),
        probability_sd=float(payload.get("probability_sd")),
        seed_pair_count=int(payload.get("seed_pair_count", -1)),
        seed_probability_hash=str(payload.get("seed_probability_hash", "")),
    )
    _assert_expected_fields(payload, result.to_payload(), "aggregated probability row")
    return result


def _statistic_row(value: object) -> CaseActionSufficientStatistics:
    payload = _mapping(value, "case statistic row")
    result = CaseActionSufficientStatistics(
        target_center=str(payload.get("target_center", "")),
        case_id=str(payload.get("case_id", "")),
        action_id=str(payload.get("action_id", "")),
        n_positive=int(payload.get("n_positive", -1)),
        true_positive=int(payload.get("true_positive", -1)),
        n_negative=int(payload.get("n_negative", -1)),
        true_negative=int(payload.get("true_negative", -1)),
    )
    _assert_expected_fields(payload, result.to_payload(), "case statistic row")
    return result


def _case_fold(payload: Mapping[str, object]) -> CaseFold:
    result = CaseFold(
        target_center=str(payload.get("target_center", "")),
        fold_ordinal=int(payload.get("fold_ordinal", -1)),
        support_case_ids=tuple(str(value) for value in _sequence(payload, "support_case_ids")),
        evaluation_case_ids=tuple(
            str(value) for value in _sequence(payload, "evaluation_case_ids")
        ),
    )
    _assert_expected_fields(payload, result.to_payload(), "case fold")
    return result


def _candidate_global_estimate(payload: Mapping[str, object]) -> CandidateGlobalEstimate:
    result = CandidateGlobalEstimate(
        action_id=str(payload.get("action_id", "")),
        donor_center_effects=_effect_rows(payload),
        donor_center_case_count=int(payload.get("donor_center_case_count", -1)),
        mean_gain_vs_b=float(payload.get("mean_gain_vs_b")),
        variance_of_mean=float(payload.get("variance_of_mean")),
        standard_error=float(payload.get("standard_error")),
        lower_confidence_bound=float(payload.get("lower_confidence_bound")),
    )
    _assert_expected_fields(payload, result.to_payload(), "candidate global estimate")
    return result


def _pairwise_prior_estimate(payload: Mapping[str, object]) -> PairwisePriorEstimate:
    result = PairwisePriorEstimate(
        challenger_action_id=str(payload.get("challenger_action_id", "")),
        reference_action_id=str(payload.get("reference_action_id", "")),
        donor_center_effects=_effect_rows(payload),
        prior_mean=float(payload.get("prior_mean")),
        prior_variance=float(payload.get("prior_variance")),
    )
    _assert_expected_fields(payload, result.to_payload(), "pairwise prior estimate")
    return result


def _posterior_estimate(payload: Mapping[str, object]) -> CandidatePosteriorEstimate:
    kwargs = {
        name: payload.get(name)
        for name in CandidatePosteriorEstimate.__dataclass_fields__
        if name != "estimate_hash"
    }
    result = CandidatePosteriorEstimate(**kwargs)
    _assert_expected_fields(payload, result.to_payload(), "posterior estimate")
    return result


def _fold_metric(payload: Mapping[str, object]) -> FoldEvaluationMetric:
    kwargs = {
        name: payload.get(name)
        for name in FoldEvaluationMetric.__dataclass_fields__
        if name != "metric_hash"
    }
    kwargs["oracle_action_ids"] = tuple(str(v) for v in _sequence(payload, "oracle_action_ids"))
    result = FoldEvaluationMetric(**kwargs)
    _assert_expected_fields(payload, result.to_payload(), "fold metric")
    return result


def _center_metric(payload: Mapping[str, object]) -> CenterEvaluationMetric:
    kwargs = {
        name: payload.get(name)
        for name in CenterEvaluationMetric.__dataclass_fields__
        if name != "metric_hash"
    }
    kwargs["best_fixed_action_ids"] = tuple(
        str(v) for v in _sequence(payload, "best_fixed_action_ids")
    )
    result = CenterEvaluationMetric(**kwargs)
    _assert_expected_fields(payload, result.to_payload(), "center metric")
    return result


def _inference_row(payload: Mapping[str, object]) -> EqualCenterInferenceRow:
    result = EqualCenterInferenceRow(
        endpoint=str(payload.get("endpoint", "")),
        center_count=int(payload.get("center_count", -1)),
        estimate=float(payload.get("estimate")),
        ci95_lower=float(payload.get("ci95_lower")),
        ci95_upper=float(payload.get("ci95_upper")),
    )
    _assert_expected_fields(payload, result.to_payload(), "equal-center inference row")
    return result


def _action_selection_row(payload: Mapping[str, object]) -> ActionSelectionMetricRow:
    result = ActionSelectionMetricRow(
        method_id=str(payload.get("method_id", "")),
        action_id=str(payload.get("action_id", "")),
        selection_count=int(payload.get("selection_count", -1)),
        total_case_count=int(payload.get("total_case_count", -1)),
        selection_share=float(payload.get("selection_share")),
    )
    _assert_expected_fields(payload, result.to_payload(), "action selection row")
    return result


def _permutation_summary(payload: Mapping[str, object]) -> PermutationNullSummaryRow:
    kwargs = {
        name: payload.get(name)
        for name in PermutationNullSummaryRow.__dataclass_fields__
        if name != "row_hash"
    }
    result = PermutationNullSummaryRow(**kwargs)
    _assert_expected_fields(payload, result.to_payload(), "permutation summary row")
    return result


def _effect_rows(payload: Mapping[str, object]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (str(_sequence_value(value, 0)), float(_sequence_value(value, 1)))
        for value in _sequence(payload, "donor_center_effects")
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a mapping payload.")
    return value


def _sequence(payload: Mapping[str, object], name: str) -> Sequence[object]:
    value = payload.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"{name} must be a sequence payload.")
    return value


def _mapping_sequence(payload: Mapping[str, object], name: str) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(value, name) for value in _sequence(payload, name))


def _sequence_value(value: object, index: int) -> object:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) <= index:
        raise ProtocolError("Nested scientific payload row is malformed.")
    return value[index]


def _bool(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if type(value) is not bool:
        raise ProtocolError(f"{name} must be a boolean payload field.")
    return value


def _assert_expected_fields(
    observed: Mapping[str, object], expected: Mapping[str, object], name: str
) -> None:
    if any(key not in observed or observed[key] != value for key, value in expected.items()):
        raise ProtocolError(f"Persisted {name} failed semantic reconstruction.")


__all__ = (
    "decision_seal_from_payload",
    "evaluation_result_from_payload",
    "fold_decision_from_payload",
    "partition_from_payload",
    "permutation_plan_from_payload",
    "pooled_fold_posterior_from_payload",
    "pooled_loco_prior_from_payload",
    "probability_surface_from_payload",
    "statistics_surface_from_payload",
)
