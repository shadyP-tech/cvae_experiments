"""Leakage-safe global and target-interaction exact-tail utility models."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ..local_marginal_utility.ridge import (
    DEFAULT_RIDGE_ALPHAS,
    NestedLOQDOResult,
    fit_cluster_weighted_ridge,
    nested_loqdo_predictions,
    select_alpha_by_inner_loqdo,
)
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .metrics import _paired_bootstrap_lower_bound, _ranking_metrics
from .result_contracts import (
    CARDINALITY_CLAIM_ROLE,
    MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP,
    MODEL_SEMANTICS,
    SOURCE_INNER_TOP1_CHANCE,
    CardinalityTransferResult,
    CrossfitResult,
    FoldAudit,
    UtilityAlignedModels,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    INNER_ROLE,
    SEED_PAIR_COUNT,
    TARGET_CANDIDATE_COUNT,
    TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION,
    ExactTailUtilityRow,
)
from .serialization import _ridge_payload
from .surface_contracts import (
    ExactTailUtilitySurface,
    FeatureSurface,
    _immutable_array,
)
from .features import permute_interaction_features
from .utility_surface import validate_exact_tail_utility_rows


_GATE_TOLERANCE = 1.0e-12


def fit_utility_aligned_models(
    feature_surface: FeatureSurface,
    utility: ExactTailUtilitySurface | Sequence[ExactTailUtilityRow],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    permutation_seed: int | None = None,
) -> UtilityAlignedModels:
    """Fit global-only and target-interaction models with strict nested LOQDO.

    Hyperparameters, standardization, covariance, and fitted coefficients are
    learned only from source-inner rows.  Every nested validation domain is
    excluded from both its query and candidate-source roles.  With the frozen
    geometry this trains on six candidates per remaining query and evaluates
    on seven, making the cardinality transfer explicit.
    """

    if not isinstance(feature_surface, FeatureSurface) or feature_surface.role != INNER_ROLE:
        raise ProtocolError("Utility-aligned fitting requires a source-inner feature surface.")
    if (
        feature_surface.permutation_seed is not None
        or feature_surface.case_bootstrap_replicate is not None
    ):
        raise ProtocolError(
            "Pass an unpermuted point-estimate feature surface to model fitting."
        )
    utility_surface = (
        utility
        if isinstance(utility, ExactTailUtilitySurface)
        else validate_exact_tail_utility_rows(utility)
    )
    target_rows = utility_surface.rows_for_outer_target(feature_surface.outer_target_id)
    utility_by_key = {row.row_key: row for row in target_rows}
    if set(utility_by_key) != set(feature_surface.row_keys):
        raise ProtocolError("Feature and exact-tail utility row keys do not align exactly.")
    for feature_row in feature_surface.rows:
        utility_row = utility_by_key[feature_row.row_key]
        if feature_row.support_partition_hash != utility_row.support_partition_hash:
            raise ProtocolError("Feature/utility support partition hashes drifted.")
    fitted_surface = (
        permute_interaction_features(
            feature_surface, permutation_seed=permutation_seed
        )
        if permutation_seed is not None
        else feature_surface
    )
    response = _immutable_array(
        np.asarray(
            [utility_by_key[key].utility_delta for key in fitted_surface.row_keys],
            dtype=np.float64,
        )
    )
    query_clusters = fitted_surface.query_clusters
    source_clusters = fitted_surface.source_clusters
    global_nested = nested_loqdo_predictions(
        fitted_surface.global_values,
        response,
        query_clusters,
        alphas=alphas,
        feature_names=fitted_surface.global_feature_names,
        include_residual_variance=True,
        source_clusters=source_clusters,
    )
    interaction_nested = nested_loqdo_predictions(
        fitted_surface.interaction_values,
        response,
        query_clusters,
        alphas=alphas,
        feature_names=fitted_surface.interaction_feature_names,
        include_residual_variance=True,
        source_clusters=source_clusters,
    )
    global_crossfit = _crossfit_contract("global_only", global_nested)
    interaction_crossfit = _crossfit_contract(
        "permuted_target_interaction"
        if permutation_seed is not None
        else "target_interaction",
        interaction_nested,
    )
    global_alpha, _global_losses = select_alpha_by_inner_loqdo(
        fitted_surface.global_values,
        response,
        query_clusters,
        alphas=alphas,
        feature_names=fitted_surface.global_feature_names,
        source_clusters=source_clusters,
    )
    interaction_alpha, _interaction_losses = select_alpha_by_inner_loqdo(
        fitted_surface.interaction_values,
        response,
        query_clusters,
        alphas=alphas,
        feature_names=fitted_surface.interaction_feature_names,
        source_clusters=source_clusters,
    )
    global_model = fit_cluster_weighted_ridge(
        fitted_surface.global_values,
        response,
        query_clusters,
        alpha=global_alpha,
        feature_names=fitted_surface.global_feature_names,
    )
    interaction_model = fit_cluster_weighted_ridge(
        fitted_surface.interaction_values,
        response,
        query_clusters,
        alpha=interaction_alpha,
        feature_names=fitted_surface.interaction_feature_names,
    )
    utility_hash = canonical_sha256(
        {
            "parent_surface_hash": utility_surface.surface_hash,
            "outer_target_id": fitted_surface.outer_target_id,
            "row_hashes": [utility_by_key[key].row_hash for key in fitted_surface.row_keys],
        }
    )
    model_payload = {
        "schema_version": "midogpp_utility_aligned_model_bundle_v1",
        "outer_target_id": fitted_surface.outer_target_id,
        "candidate_sources": list(fitted_surface.candidate_sources),
        "feature_surface_hash": fitted_surface.surface_hash,
        "utility_surface_hash": utility_hash,
        "global_selected_alpha": global_alpha,
        "interaction_selected_alpha": interaction_alpha,
        "global_model": _ridge_payload(global_model),
        "interaction_model": _ridge_payload(interaction_model),
        "global_crossfit_hash": global_crossfit.crossfit_hash,
        "interaction_crossfit_hash": interaction_crossfit.crossfit_hash,
        "permutation_seed": permutation_seed,
        "model_semantics": MODEL_SEMANTICS,
        "strict_nested_query_source_exclusion": True,
        "target_or_query_identity_features_used": False,
        "target_labels_used": False,
        "seed_selection_performed": False,
    }
    return UtilityAlignedModels(
        outer_target_id=fitted_surface.outer_target_id,
        candidate_sources=fitted_surface.candidate_sources,
        global_model=global_model,
        interaction_model=interaction_model,
        global_crossfit=global_crossfit,
        interaction_crossfit=interaction_crossfit,
        global_selected_alpha=global_alpha,
        interaction_selected_alpha=interaction_alpha,
        feature_surface_hash=fitted_surface.surface_hash,
        utility_surface_hash=utility_hash,
        permutation_seed=permutation_seed,
        model_semantics=MODEL_SEMANTICS,
        model_hash=canonical_sha256(model_payload),
    )


def nested_cardinality_transfer_evaluation(
    models: UtilityAlignedModels,
    feature_surface: FeatureSurface,
    utility: ExactTailUtilitySurface | Sequence[ExactTailUtilityRow],
) -> CardinalityTransferResult:
    """Evaluate strict nested 6->7 ranking for 7->8 *eligibility* only.

    Source-inner query domains, not seed cells, are the independent units.
    Predictions and true deltas are first averaged over all nine paired seed
    cells for each query/candidate.  No seed may be selected.  Passing this
    gate is not evidence that the router works at target cardinality and does
    not authorize a downstream-utility claim; only fresh Stage-70 evaluation
    can establish that claim.
    """

    if not isinstance(models, UtilityAlignedModels):
        raise ProtocolError("Cardinality transfer requires fitted utility models.")
    if feature_surface.role != INNER_ROLE or models.outer_target_id != feature_surface.outer_target_id:
        raise ProtocolError("Cardinality transfer feature surface drifted.")
    utility_surface = (
        utility
        if isinstance(utility, ExactTailUtilitySurface)
        else validate_exact_tail_utility_rows(utility)
    )
    target_rows = utility_surface.rows_for_outer_target(models.outer_target_id)
    utility_by_key = {row.row_key: row for row in target_rows}
    if set(utility_by_key) != set(feature_surface.row_keys):
        raise ProtocolError("Cardinality transfer row keys do not align.")
    response = np.asarray(
        [utility_by_key[key].utility_delta for key in feature_surface.row_keys],
        dtype=np.float64,
    )
    global_metrics, global_query_metrics = _ranking_metrics(
        feature_surface,
        response,
        models.global_crossfit.predictions,
        model_role="global_only",
    )
    interaction_metrics, interaction_query_metrics = _ranking_metrics(
        feature_surface,
        response,
        models.interaction_crossfit.predictions,
        model_role="target_interaction",
    )
    top1_delta = (
        interaction_metrics.top1_oracle_agreement
        - global_metrics.top1_oracle_agreement
    )
    spearman_delta = interaction_metrics.mean_spearman - global_metrics.mean_spearman
    gap_reduction = (
        global_metrics.mean_normalized_oracle_gap
        - interaction_metrics.mean_normalized_oracle_gap
    )
    pairwise_delta = interaction_metrics.pairwise_accuracy - global_metrics.pairwise_accuracy
    selected_utility_delta = (
        interaction_metrics.mean_selected_utility_delta
        - global_metrics.mean_selected_utility_delta
    )
    gap_reduction_lcb = _paired_bootstrap_lower_bound(
        global_query_metrics["normalized_gap"]
        - interaction_query_metrics["normalized_gap"]
    )
    selected_utility_delta_lcb = _paired_bootstrap_lower_bound(
        interaction_query_metrics["selected_utility"]
        - global_query_metrics["selected_utility"]
    )
    global_gate_passed = global_metrics.selected_utility_lower_bound > _GATE_TOLERANCE
    global_gate_reason = (
        "pass_global_source_quality_positive_gain_lcb"
        if global_gate_passed
        else "global_selected_additive_tail_gain_lcb_nonpositive"
    )
    failures: list[str] = []
    if not top1_delta > _GATE_TOLERANCE:
        failures.append("top1_not_better_than_global")
    if not interaction_metrics.top1_lower_bound > SOURCE_INNER_TOP1_CHANCE:
        failures.append("top1_lcb_not_above_one_in_seven_chance")
    if not interaction_metrics.spearman_lower_bound > _GATE_TOLERANCE:
        failures.append("spearman_lcb_nonpositive")
    if not interaction_metrics.mean_spearman > max(
        0.0, global_metrics.mean_spearman
    ) + _GATE_TOLERANCE:
        failures.append("spearman_not_better_than_global")
    if not gap_reduction_lcb > _GATE_TOLERANCE:
        failures.append("normalized_gap_reduction_lcb_nonpositive")
    if not (
        interaction_metrics.normalized_oracle_gap_upper_bound
        < MAX_AUTHORIZED_NORMALIZED_ORACLE_GAP
    ):
        failures.append("normalized_gap_ucb_not_below_point_four_six")
    if not selected_utility_delta_lcb > _GATE_TOLERANCE:
        failures.append("selected_utility_delta_lcb_nonpositive")
    if not interaction_metrics.selected_utility_lower_bound > _GATE_TOLERANCE:
        failures.append("selected_additive_tail_gain_lcb_nonpositive")
    if pairwise_delta < -_GATE_TOLERANCE:
        failures.append("pairwise_accuracy_worse_than_global")
    eligibility_passed = not failures
    eligibility_reason = (
        "eligible_for_fresh_7_to_8_evaluation_not_evidence"
        if eligibility_passed
        else "|".join(failures)
    )
    payload = {
        "schema_version": "midogpp_utility_aligned_cardinality_transfer_v1",
        "outer_target_id": models.outer_target_id,
        "candidate_sources": list(models.candidate_sources),
        "training_candidate_count": TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION,
        "evaluation_candidate_count": INNER_CANDIDATE_COUNT,
        "deployment_candidate_count": TARGET_CANDIDATE_COUNT,
        "global_metrics_hash": global_metrics.metrics_hash,
        "interaction_metrics_hash": interaction_metrics.metrics_hash,
        "top1_delta": top1_delta,
        "spearman_delta": spearman_delta,
        "normalized_gap_reduction": gap_reduction,
        "normalized_gap_reduction_lower_bound": gap_reduction_lcb,
        "pairwise_accuracy_delta": pairwise_delta,
        "selected_utility_delta": selected_utility_delta,
        "selected_utility_delta_lower_bound": selected_utility_delta_lcb,
        "global_gate_passed": global_gate_passed,
        "global_gate_reason": global_gate_reason,
        "eligibility_passed": eligibility_passed,
        "eligibility_reason": eligibility_reason,
        "claim_role": CARDINALITY_CLAIM_ROLE,
        "model_hash": models.model_hash,
        "query_domains_are_independent_units": True,
        "seed_selection_performed": False,
    }
    return CardinalityTransferResult(
        outer_target_id=models.outer_target_id,
        candidate_sources=models.candidate_sources,
        training_candidate_count=TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION,
        evaluation_candidate_count=INNER_CANDIDATE_COUNT,
        deployment_candidate_count=TARGET_CANDIDATE_COUNT,
        global_metrics=global_metrics,
        interaction_metrics=interaction_metrics,
        top1_delta=top1_delta,
        spearman_delta=spearman_delta,
        normalized_gap_reduction=gap_reduction,
        normalized_gap_reduction_lower_bound=gap_reduction_lcb,
        pairwise_accuracy_delta=pairwise_delta,
        selected_utility_delta=selected_utility_delta,
        selected_utility_delta_lower_bound=selected_utility_delta_lcb,
        global_gate_passed=global_gate_passed,
        global_gate_reason=global_gate_reason,
        eligibility_passed=eligibility_passed,
        eligibility_reason=eligibility_reason,
        claim_role=CARDINALITY_CLAIM_ROLE,
        model_hash=models.model_hash,
        result_hash=canonical_sha256(payload),
    )


def _crossfit_contract(model_role: str, result: NestedLOQDOResult) -> CrossfitResult:
    folds: list[FoldAudit] = []
    for fold in result.folds:
        training_queries = fold.model.training_query_clusters
        if fold.heldout_query_cluster in training_queries:
            raise ProtocolError("Nested fold retained its held-out query.")
        if fold.heldout_query_cluster in fold.training_source_clusters:
            raise ProtocolError("Nested fold retained its held-out source domain.")
        denominator = len(training_queries) * SEED_PAIR_COUNT
        if denominator <= 0 or fold.model.observation_count % denominator != 0:
            raise ProtocolError("Nested fold observation geometry is not query-balanced.")
        training_candidate_count = fold.model.observation_count // denominator
        if training_candidate_count != TRAIN_CANDIDATE_COUNT_AFTER_STRICT_EXCLUSION:
            raise ProtocolError("Nested fold is not the required strict 6->7 transfer.")
        if len(fold.heldout_row_indices) != INNER_CANDIDATE_COUNT * SEED_PAIR_COUNT:
            raise ProtocolError("Nested held-out query is not a complete seven-source list.")
        folds.append(
            FoldAudit(
                heldout_query_id=fold.heldout_query_cluster,
                heldout_row_indices=fold.heldout_row_indices,
                training_query_ids=training_queries,
                training_source_ids=fold.training_source_clusters,
                training_candidate_count_per_query=training_candidate_count,
                selected_alpha=fold.selected_alpha,
                inner_loss_by_alpha=fold.inner_mse_by_alpha,
                observation_count=fold.model.observation_count,
            )
        )
    predictions = _immutable_array(result.predictions)
    standard_errors = _immutable_array(result.standard_errors)
    payload = {
        "schema_version": "midogpp_utility_aligned_crossfit_v1",
        "model_role": model_role,
        "prediction_sha256": array_sha256(predictions),
        "standard_error_sha256": array_sha256(standard_errors),
        "folds": [
            {
                "heldout_query_id": fold.heldout_query_id,
                "heldout_row_indices": list(fold.heldout_row_indices),
                "training_query_ids": list(fold.training_query_ids),
                "training_source_ids": list(fold.training_source_ids),
                "training_candidate_count_per_query": fold.training_candidate_count_per_query,
                "selected_alpha": fold.selected_alpha,
                "inner_loss_by_alpha": dict(fold.inner_loss_by_alpha),
                "observation_count": fold.observation_count,
            }
            for fold in folds
        ],
    }
    return CrossfitResult(
        model_role=model_role,
        predictions=predictions,
        standard_errors=standard_errors,
        folds=tuple(folds),
        crossfit_hash=canonical_sha256(payload),
    )

__all__ = (
    "fit_utility_aligned_models",
    "nested_cardinality_transfer_evaluation",
)
