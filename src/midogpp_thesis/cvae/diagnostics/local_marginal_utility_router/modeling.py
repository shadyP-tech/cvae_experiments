"""Leakage-safe utility learnability and unscored target-plan construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import build_hamilton_allocation
from ...routing.local_marginal_utility import (
    build_energy_feature_matrix,
    fit_cluster_weighted_ridge,
    nested_loqdo_predictions,
    robust_local_utility_weights,
    select_alpha_by_inner_loqdo,
)
from .contracts import (
    CENTERS,
    MODEL_ALPHA_GRID,
    OPTIMIZER_KAPPA,
    OPTIMIZER_L2_PENALTY,
    TOTAL_PER_CLASS,
    legal_sources,
    target_sources,
)
from .utility_surface import summarize_loqdo_learnability


MODEL_FIT_COLUMNS = (
    "schema_version",
    "row_role",
    "claim_role",
    "outer_target",
    "selected_alpha",
    "inner_query_equal_mse_by_alpha_json",
    "feature_names_json",
    "feature_mean_json",
    "feature_scale_json",
    "intercept",
    "coefficients_json",
    "coefficient_covariance_json",
    "residual_variance",
    "training_query_centers_json",
    "training_row_count",
    "effective_rank",
    "outer_target_query_excluded_from_fit",
    "outer_target_source_excluded_from_fit",
    "normalization_fit_on_q_not_H_only",
    "support_labels_used",
    "seed_selection_performed",
    "model_hash",
)

TARGET_PLAN_COLUMNS = (
    "schema_version",
    "row_role",
    "claim_role",
    "target_center",
    "candidate_sources_json",
    "predicted_marginal_utility_json",
    "prediction_standard_error_json",
    "prediction_covariance_json",
    "uniform_weights_json",
    "delta_json",
    "weights_json",
    "allocations_per_class_json",
    "total_per_class",
    "robust_objective_value",
    "expected_first_order_gain",
    "uncertainty_penalty",
    "l2_penalty_value",
    "kappa",
    "l2_penalty",
    "maximum_source_weight",
    "effective_source_count",
    "used_uniform_fallback",
    "fallback_reason",
    "solver_success",
    "solver_message",
    "solver_iterations",
    "selected_alpha",
    "geometry_transfer_status",
    "selection_source",
    "target_labels_used",
    "support_labels_used",
    "seed_selection_performed",
    "target_performance_scored",
    "oracle_eligible",
    "may_feed_stage60",
    "may_feed_stage70",
    "plan_hash",
)


@dataclass(frozen=True)
class LearnedUtilitySurface:
    learnability_prediction_rows: tuple[Mapping[str, object], ...]
    learnability_summary_rows: tuple[Mapping[str, object], ...]
    model_fit_rows: tuple[Mapping[str, object], ...]
    target_plan_rows: tuple[Mapping[str, object], ...]


def fit_models_and_build_unscored_target_plans(
    *,
    calibrated_energy_by_query: Mapping[str, Mapping[str, float]],
    marginal_utility_rows: Sequence[Mapping[str, object]],
    alpha_grid: Sequence[float] = MODEL_ALPHA_GRID,
    kappa: float = OPTIMIZER_KAPPA,
    l2_penalty: float = OPTIMIZER_L2_PENALTY,
) -> LearnedUtilitySurface:
    """Fit per-H models and emit plans whose target outcomes remain unopened.

    All features are label-free.  For a held-out inner query ``q``, both rows
    queried at ``q`` and rows using expert ``q`` are excluded from fitting.
    This mirrors the outer target firewall rather than treating an expert from
    the held-out domain as ordinary training information.
    """

    utility_rows = tuple(marginal_utility_rows)
    prediction_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for outer_target in CENTERS:
        fold_rows = tuple(
            row for row in utility_rows if str(row.get("outer_target")) == outer_target
        )
        _validate_outer_surface_rows(fold_rows, outer_target=outer_target)

        candidates_by_query = {
            query: (
                target_sources(outer_target)
                if query == outer_target
                else legal_sources(
                    outer_target=outer_target,
                    query_center=query,
                )
            )
            for query in CENTERS
        }
        energy_subset = {
            query: {
                source: _energy(calibrated_energy_by_query, query, source)
                for source in candidates_by_query[query]
            }
            for query in CENTERS
        }
        feature_surface = build_energy_feature_matrix(
            energy_subset,
            candidate_sources_by_query=candidates_by_query,
            include_source_indicators=True,
        )
        feature_by_key = {
            key: feature_surface.values[index]
            for index, key in enumerate(feature_surface.row_keys)
        }
        matrix = np.asarray(
            [
                feature_by_key[
                    (str(row["query_center"]), str(row["source_center"]))
                ]
                for row in fold_rows
            ],
            dtype=np.float64,
        )
        utility = np.asarray(
            [float(row["marginal_bacc_utility"]) for row in fold_rows],
            dtype=np.float64,
        )
        query_domains = tuple(str(row["query_center"]) for row in fold_rows)
        source_domains = tuple(str(row["source_center"]) for row in fold_rows)
        if not np.isfinite(utility).all():
            raise ProtocolError("Local marginal utility response is non-finite.")

        nested = nested_loqdo_predictions(
            matrix,
            utility,
            query_domains,
            source_clusters=source_domains,
            alphas=alpha_grid,
            feature_names=feature_surface.feature_names,
            include_residual_variance=True,
        )
        fold_by_query = {
            fold.heldout_query_cluster: fold for fold in nested.folds
        }
        for index, row in enumerate(fold_rows):
            heldout = str(row["query_center"])
            fold = fold_by_query[heldout]
            train_queries = tuple(
                query
                for query in sorted(set(query_domains))
                if query != heldout
            )
            prediction_rows.append(
                {
                    "schema_version": "midogpp_local_marginal_utility_loqdo_prediction_v1",
                    "row_role": "inner_loqdo_heldout_query_prediction",
                    "claim_role": "utility_learnability_not_routing_performance",
                    "outer_target": outer_target,
                    "heldout_query_center": heldout,
                    "source_center": str(row["source_center"]),
                    "training_seed": int(row["training_seed"]),
                    "generation_seed": int(row["generation_seed"]),
                    "observed_marginal_utility": float(utility[index]),
                    "predicted_marginal_utility": float(nested.predictions[index]),
                    "prediction_standard_error": float(nested.standard_errors[index]),
                    "alpha": float(fold.selected_alpha),
                    "train_query_centers_json": _json_compact(train_queries),
                    "heldout_query_excluded_from_fit": True,
                    "heldout_query_excluded_from_source_role": True,
                    "outer_target_excluded_from_fit": True,
                    "seed_selection_performed": False,
                }
            )

        selected_alpha, inner_losses = select_alpha_by_inner_loqdo(
            matrix,
            utility,
            query_domains,
            source_clusters=source_domains,
            alphas=alpha_grid,
            feature_names=feature_surface.feature_names,
        )
        model = fit_cluster_weighted_ridge(
            matrix,
            utility,
            query_domains,
            alpha=selected_alpha,
            feature_names=feature_surface.feature_names,
        )
        model_payload = {
            "schema_version": "midogpp_local_marginal_utility_model_fit_v1",
            "row_role": "outer_target_development_model",
            "claim_role": "consumed_validation_diagnostic_only",
            "outer_target": outer_target,
            "selected_alpha": selected_alpha,
            "inner_query_equal_mse_by_alpha_json": _json_compact(
                {str(alpha): loss for alpha, loss in sorted(inner_losses.items())}
            ),
            "feature_names_json": _json_compact(model.feature_names),
            "feature_mean_json": _json_compact(model.feature_mean.tolist()),
            "feature_scale_json": _json_compact(model.feature_scale.tolist()),
            "intercept": model.intercept,
            "coefficients_json": _json_compact(model.coefficients.tolist()),
            "coefficient_covariance_json": _json_compact(
                model.coefficient_covariance.tolist()
            ),
            "residual_variance": model.residual_variance,
            "training_query_centers_json": _json_compact(
                sorted(set(query_domains))
            ),
            "training_row_count": model.observation_count,
            "effective_rank": model.effective_rank,
            "outer_target_query_excluded_from_fit": True,
            "outer_target_source_excluded_from_fit": True,
            "normalization_fit_on_q_not_H_only": True,
            "support_labels_used": False,
            "seed_selection_performed": False,
        }
        model_payload["model_hash"] = stable_hash(model_payload)
        model_rows.append(model_payload)

        target_candidates = target_sources(outer_target)
        target_matrix = np.asarray(
            [feature_by_key[(outer_target, source)] for source in target_candidates],
            dtype=np.float64,
        )
        target_prediction = model.predict_with_uncertainty(
            target_matrix,
            include_residual_variance=True,
        )
        marginal_by_source = {
            source: float(value)
            for source, value in zip(
                target_candidates, target_prediction.mean, strict=True
            )
        }
        solution = robust_local_utility_weights(
            marginal_by_source,
            target_prediction.covariance,
            covariance_source_order=target_candidates,
            kappa=kappa,
            l2_penalty=l2_penalty,
            max_source_weight=0.25,
            min_effective_sources=6.0,
        )
        allocation = build_hamilton_allocation(
            solution.weights,
            total=TOTAL_PER_CLASS,
            minimum_per_source=1,
        )
        standard_error_by_source = {
            source: float(value)
            for source, value in zip(
                target_candidates, target_prediction.standard_error, strict=True
            )
        }
        target_payload = {
            "schema_version": "midogpp_local_marginal_utility_target_plan_v1",
            "row_role": "unscored_target_plan_recommendation",
            "claim_role": "mechanism_diagnostic_not_routing_evidence",
            "target_center": outer_target,
            "candidate_sources_json": _json_compact(target_candidates),
            "predicted_marginal_utility_json": _json_compact(
                solution.predicted_marginal_utility
            ),
            "prediction_standard_error_json": _json_compact(
                standard_error_by_source
            ),
            "prediction_covariance_json": _json_compact(
                target_prediction.covariance.tolist()
            ),
            "uniform_weights_json": _json_compact(solution.uniform_weights),
            "delta_json": _json_compact(solution.delta),
            "weights_json": _json_compact(solution.weights),
            "allocations_per_class_json": _json_compact(allocation.allocations),
            "total_per_class": TOTAL_PER_CLASS,
            "robust_objective_value": solution.objective_value,
            "expected_first_order_gain": solution.expected_gain,
            "uncertainty_penalty": solution.uncertainty_penalty,
            "l2_penalty_value": solution.l2_penalty_value,
            "kappa": float(kappa),
            "l2_penalty": float(l2_penalty),
            "maximum_source_weight": solution.maximum_source_weight,
            "effective_source_count": solution.effective_source_count,
            "used_uniform_fallback": solution.used_uniform_fallback,
            "fallback_reason": solution.fallback_reason or "",
            "solver_success": solution.solver_success,
            "solver_message": solution.solver_message,
            "solver_iterations": solution.solver_iterations,
            "selected_alpha": selected_alpha,
            "geometry_transfer_status": "extrapolative_unscored_diagnostic_only",
            "selection_source": "q_not_H_consumed_validation_utility_surface",
            "target_labels_used": False,
            "support_labels_used": False,
            "seed_selection_performed": False,
            "target_performance_scored": False,
            "oracle_eligible": False,
            "may_feed_stage60": False,
            "may_feed_stage70": False,
        }
        target_payload["plan_hash"] = stable_hash(target_payload)
        target_rows.append(target_payload)

    summaries = summarize_loqdo_learnability(prediction_rows)
    return LearnedUtilitySurface(
        learnability_prediction_rows=tuple(prediction_rows),
        learnability_summary_rows=summaries,
        model_fit_rows=tuple(model_rows),
        target_plan_rows=tuple(target_rows),
    )


def _validate_outer_surface_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    outer_target: str,
) -> None:
    expected = {
        (query, source, training_seed, generation_seed)
        for query in CENTERS
        if query != outer_target
        for source in legal_sources(
            outer_target=outer_target,
            query_center=query,
        )
        for training_seed in (17, 42, 101)
        for generation_seed in (17, 42, 101)
    }
    observed = {
        (
            str(row.get("query_center")),
            str(row.get("source_center")),
            int(row.get("training_seed")),
            int(row.get("generation_seed")),
        )
        for row in rows
    }
    if len(rows) != len(observed) or observed != expected:
        raise ProtocolError("Local marginal-utility outer surface coverage drifted.")
    if any(
        row.get("support_labels_used") is not False
        or row.get("target_H_labels_used") is not False
        or row.get("seed_selection_performed") is not False
        for row in rows
    ):
        raise ProtocolError("Local marginal-utility row violates the claim firewall.")


def _energy(
    energies: Mapping[str, Mapping[str, float]],
    query: str,
    source: str,
) -> float:
    try:
        value = float(energies[query][source])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Compatibility features do not cover a legal q/e pair.") from exc
    if not math.isfinite(value):
        raise ProtocolError("Compatibility feature is non-finite.")
    return value


def _json_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = (
    "MODEL_FIT_COLUMNS",
    "TARGET_PLAN_COLUMNS",
    "LearnedUtilitySurface",
    "fit_models_and_build_unscored_target_plans",
)
