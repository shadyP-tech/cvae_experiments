"""Low-capacity candidate-level ensemble utility modeling."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..local_marginal_utility.ridge import (
    DEFAULT_RIDGE_ALPHAS,
    ClusterWeightedRidgeModel,
    fit_cluster_weighted_ridge,
)
from ..residual_topup.hashing import array_sha256, canonical_sha256
from .ensemble_utility_contracts import (
    EnsembleUtilityResponse,
    EnsembleUtilitySurface,
    ScoredEnsembleUtilityResponse,
)
from .ensemble_endpoint import validate_ensemble_utility_responses
from .ensemble_feature_contracts import (
    GLOBAL_SOURCE_CONTROL_NAME,
    EnsembleFeatureSurface,
)
from .ensemble_model_contracts import (
    ROUTING_TUNING_ENDPOINT,
    EnsembleCapacityReport,
    EnsembleFoldAudit,
    EnsembleUtilityModel,
)
from .row_contracts import INNER_ROLE
from .serialization import _ridge_payload
from .surface_contracts import _immutable_array


_CONSTANT_TOLERANCE = float(np.sqrt(np.finfo(np.float64).eps))


def fit_ensemble_utility_model(
    feature_surface: EnsembleFeatureSurface,
    utility: EnsembleUtilitySurface
    | Sequence[
        EnsembleUtilityResponse | ScoredEnsembleUtilityResponse | Mapping[str, object]
    ],
    *,
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
) -> EnsembleUtilityModel:
    """Fit M0/M1 with strict H/q/e exclusion and routing-endpoint tuning.

    The response has one row per candidate ``(H,q,e)``.  For every crossfit
    prediction, all rows using ``H``, ``q``, or ``e`` in either query or source
    role are excluded.  Alpha is selected by query-level normalized oracle
    regret, never by seed-row or candidate-row MSE.  Deployment receives one
    candidate-specific model per source, fitted without that source in either
    source-inner role.
    """

    _validate_model_surface(feature_surface)
    if feature_surface.role != INNER_ROLE:
        raise ProtocolError("Ensemble utility fitting requires source-inner features.")
    utility_surface = (
        utility
        if isinstance(utility, EnsembleUtilitySurface)
        else validate_ensemble_utility_responses(utility)
    )
    utility_rows = utility_surface.rows_for_outer_target(feature_surface.outer_target_id)
    utility_by_key = {row.row_key: row for row in utility_rows}
    if set(utility_by_key) != set(feature_surface.row_keys):
        raise ProtocolError("Ensemble feature/utility H/q/e keys do not align exactly.")
    for feature_row in feature_surface.rows:
        if (
            feature_row.support_partition_hash
            != utility_by_key[feature_row.row_key].support_partition_hash
        ):
            raise ProtocolError("Ensemble feature/utility support partitions drifted.")
    response = np.asarray(
        [utility_by_key[key].utility_delta for key in feature_surface.row_keys],
        dtype=np.float64,
    )
    candidates = _validated_alphas(alphas)
    queries = tuple(sorted(set(feature_surface.query_clusters)))
    row_index = {key: index for index, key in enumerate(feature_surface.row_keys)}

    crossfit = np.empty(len(feature_surface.rows), dtype=np.float64)
    selected_by_query: dict[str, float] = {}
    fold_audits: list[EnsembleFoldAudit] = []
    for heldout_query in queries:
        inner_losses = _routing_loss_by_alpha(
            feature_surface,
            response,
            candidates,
            permanently_excluded=(heldout_query,),
        )
        selected = min(candidates, key=lambda alpha: (inner_losses[alpha], alpha))
        selected_by_query[heldout_query] = selected
        for row in feature_surface.rows:
            if row.query_id != heldout_query:
                continue
            excluded = (
                feature_surface.outer_target_id,
                heldout_query,
                row.candidate_source,
            )
            model, report, training_indices = _fit_with_exclusion(
                feature_surface,
                response,
                excluded_domain_ids=excluded,
                alpha=selected,
                context=(
                    f"crossfit::{feature_surface.outer_target_id}::"
                    f"{heldout_query}::{row.candidate_source}"
                ),
            )
            index = row_index[row.row_key]
            crossfit[index] = float(model.predict(feature_surface.values[index : index + 1])[0])
            training_rows = tuple(feature_surface.rows[index] for index in training_indices)
            training_queries = tuple(sorted({item.query_id for item in training_rows}))
            training_sources = tuple(
                sorted({item.candidate_source for item in training_rows})
            )
            excluded_set = set(excluded)
            if excluded_set & set(training_queries) or excluded_set & set(training_sources):
                raise ProtocolError("Strict H/q/e exclusion audit failed.")
            fold_audits.append(
                EnsembleFoldAudit(
                    predicted_row_key=row.row_key,
                    excluded_domain_ids=tuple(sorted(excluded_set)),
                    training_query_ids=training_queries,
                    training_source_ids=training_sources,
                    selected_alpha=selected,
                    capacity_report_hash=report.report_hash,
                )
            )
    if not np.isfinite(crossfit).all():
        raise ProtocolError("Ensemble crossfit produced non-finite predictions.")
    crossfit = _immutable_array(crossfit)

    routing_losses = _routing_loss_by_alpha(
        feature_surface, response, candidates, permanently_excluded=()
    )
    selected_alpha = min(candidates, key=lambda alpha: (routing_losses[alpha], alpha))
    candidate_models: dict[str, ClusterWeightedRidgeModel] = {}
    candidate_reports: dict[str, EnsembleCapacityReport] = {}
    for source in feature_surface.candidate_sources:
        model, report, _indices = _fit_with_exclusion(
            feature_surface,
            response,
            excluded_domain_ids=(feature_surface.outer_target_id, source),
            alpha=selected_alpha,
            context=f"deployment::{feature_surface.outer_target_id}::{source}",
        )
        candidate_models[source] = model
        candidate_reports[source] = report
    utility_hash = canonical_sha256(
        {
            "schema_version": "midogpp_utility_aligned_ensemble_utility_binding_v1",
            "parent_surface_hash": utility_surface.surface_hash,
            "outer_target_id": feature_surface.outer_target_id,
            "row_hashes": [utility_by_key[key].row_hash for key in feature_surface.row_keys],
        }
    )
    payload = {
        "schema_version": "midogpp_utility_aligned_ensemble_model_v1",
        "outer_target_id": feature_surface.outer_target_id,
        "feature_names": list(feature_surface.feature_names),
        "selected_alpha": selected_alpha,
        "routing_tuning_endpoint": ROUTING_TUNING_ENDPOINT,
        "routing_loss_by_alpha": {str(key): value for key, value in routing_losses.items()},
        "selected_alpha_by_heldout_query": selected_by_query,
        "candidate_models": {
            source: _ridge_payload(model)
            for source, model in sorted(candidate_models.items())
        },
        "candidate_capacity_report_hashes": {
            source: report.report_hash
            for source, report in sorted(candidate_reports.items())
        },
        "crossfit_prediction_sha256": array_sha256(crossfit),
        "crossfit_row_keys": [list(key) for key in feature_surface.row_keys],
        "fold_audits": [
            {
                "predicted_row_key": list(audit.predicted_row_key),
                "excluded_domain_ids": list(audit.excluded_domain_ids),
                "training_query_ids": list(audit.training_query_ids),
                "training_source_ids": list(audit.training_source_ids),
                "selected_alpha": audit.selected_alpha,
                "capacity_report_hash": audit.capacity_report_hash,
                "strict_h_q_e_exclusion": audit.strict_h_q_e_exclusion,
            }
            for audit in fold_audits
        ],
        "feature_surface_hash": feature_surface.surface_hash,
        "utility_surface_hash": utility_hash,
        "permutation_seed": feature_surface.permutation_seed,
        "candidate_response_unit": "H_q_e_after_exact_nine_probability_ensemble",
        "seed_rows_are_independent_observations": False,
        "target_or_query_identity_features_used": False,
        "tuning_uses_row_mse": False,
        "strict_h_q_e_exclusion": True,
    }
    return EnsembleUtilityModel(
        outer_target_id=feature_surface.outer_target_id,
        feature_names=feature_surface.feature_names,
        selected_alpha=selected_alpha,
        routing_tuning_endpoint=ROUTING_TUNING_ENDPOINT,
        routing_loss_by_alpha=routing_losses,
        selected_alpha_by_heldout_query=selected_by_query,
        candidate_models=candidate_models,
        candidate_capacity_reports=candidate_reports,
        crossfit_predictions=crossfit,
        crossfit_row_keys=feature_surface.row_keys,
        fold_audits=tuple(fold_audits),
        feature_surface_hash=feature_surface.surface_hash,
        utility_surface_hash=utility_hash,
        permutation_seed=feature_surface.permutation_seed,
        model_hash=canonical_sha256(payload),
    )


def _routing_loss_by_alpha(
    surface: EnsembleFeatureSurface,
    response: np.ndarray,
    alphas: tuple[float, ...],
    *,
    permanently_excluded: tuple[str, ...],
) -> dict[float, float]:
    excluded_permanent = set(permanently_excluded)
    query_ids = tuple(
        query for query in sorted(set(surface.query_clusters)) if query not in excluded_permanent
    )
    if len(query_ids) < 3:
        raise ProtocolError("Routing-endpoint tuning requires at least three query groups.")
    losses: dict[float, float] = {}
    for alpha in alphas:
        query_regrets: list[float] = []
        for query in query_ids:
            query_indices = [
                index
                for index, row in enumerate(surface.rows)
                if row.query_id == query
                and row.candidate_source not in excluded_permanent
            ]
            if len(query_indices) < 2:
                raise ProtocolError("Routing tuning candidate list is incomplete.")
            predicted: dict[str, float] = {}
            truth: dict[str, float] = {}
            for index in query_indices:
                row = surface.rows[index]
                excluded = tuple(
                    sorted(
                        {
                            surface.outer_target_id,
                            *excluded_permanent,
                            query,
                            row.candidate_source,
                        }
                    )
                )
                model, _report, _training_indices = _fit_with_exclusion(
                    surface,
                    response,
                    excluded_domain_ids=excluded,
                    alpha=alpha,
                    context=(
                        f"routing_tune::{surface.outer_target_id}::"
                        f"{query}::{row.candidate_source}"
                    ),
                )
                predicted[row.candidate_source] = float(
                    model.predict(surface.values[index : index + 1])[0]
                )
                truth[row.candidate_source] = float(response[index])
            selected = min(predicted, key=lambda source: (-predicted[source], source))
            maximum = max(truth.values())
            minimum = min(truth.values())
            denominator = maximum - minimum
            regret = 0.0 if denominator <= 0.0 else (maximum - truth[selected]) / denominator
            query_regrets.append(float(regret))
        losses[alpha] = float(np.mean(query_regrets, dtype=np.float64))
    return losses


def _fit_with_exclusion(
    surface: EnsembleFeatureSurface,
    response: np.ndarray,
    *,
    excluded_domain_ids: Iterable[str],
    alpha: float,
    context: str,
) -> tuple[ClusterWeightedRidgeModel, EnsembleCapacityReport, tuple[int, ...]]:
    excluded = {str(value) for value in excluded_domain_ids}
    indices = tuple(
        index
        for index, row in enumerate(surface.rows)
        if row.query_id not in excluded and row.candidate_source not in excluded
    )
    if not indices:
        raise ProtocolError("Strict H/q/e exclusion left no training candidates.")
    matrix = np.asarray(surface.values[list(indices)], dtype=np.float64)
    target = np.asarray(response[list(indices)], dtype=np.float64)
    clusters = tuple(surface.rows[index].query_id for index in indices)
    preliminary = _capacity_report(
        matrix,
        clusters,
        surface.feature_names,
        context=context,
        model=None,
        response=None,
    )
    if not preliminary.gate_passed:
        raise ProtocolError(
            "Ensemble model capacity gate failed: " + "; ".join(preliminary.failures)
        )
    model = fit_cluster_weighted_ridge(
        matrix,
        target,
        clusters,
        alpha=alpha,
        feature_names=surface.feature_names,
    )
    report = _capacity_report(
        matrix,
        clusters,
        surface.feature_names,
        context=context,
        model=model,
        response=target,
    )
    return model, report, indices


def _capacity_report(
    matrix: np.ndarray,
    clusters: tuple[str, ...],
    feature_names: tuple[str, ...],
    *,
    context: str,
    model: ClusterWeightedRidgeModel | None,
    response: np.ndarray | None,
) -> EnsembleCapacityReport:
    values = np.asarray(matrix, dtype=np.float64)
    unique_queries = tuple(sorted(set(clusters)))
    constant_columns = tuple(
        feature_names[index]
        for index in range(values.shape[1])
        if float(np.ptp(values[:, index])) <= _CONSTANT_TOLERANCE
    )
    design = np.column_stack((np.ones(len(values), dtype=np.float64), values))
    design_rank = int(np.linalg.matrix_rank(design))
    design_columns = int(design.shape[1])
    sandwich_ceiling = min(max(len(unique_queries) - 1, 0), design_rank)
    failures: list[str] = []
    if values.shape[1] > 2:
        failures.append("more_than_two_predictor_columns")
    if not feature_names or feature_names[0] != GLOBAL_SOURCE_CONTROL_NAME:
        failures.append("global_source_control_missing_or_not_first")
    if constant_columns:
        failures.append("constant_predictor_columns")
    if design_rank != design_columns:
        failures.append("rank_deficient_design")
    if sandwich_ceiling < design_rank:
        failures.append("cluster_sandwich_rank_ceiling_below_design_rank")
    observed_sandwich_rank: int | None = None
    if model is not None and response is not None:
        standardized = (values - model.feature_mean) / model.feature_scale
        standardized_design = np.column_stack(
            (np.ones(len(values), dtype=np.float64), standardized)
        )
        residual = response - model.predict(values)
        count_by_cluster = {
            cluster: sum(item == cluster for item in clusters)
            for cluster in unique_queries
        }
        weights = np.asarray(
            [
                len(values) / (len(unique_queries) * count_by_cluster[cluster])
                for cluster in clusters
            ],
            dtype=np.float64,
        )
        scores: list[np.ndarray] = []
        cluster_array = np.asarray(clusters, dtype=object)
        for cluster in unique_queries:
            mask = cluster_array == cluster
            scores.append(
                standardized_design[mask].T @ (weights[mask] * residual[mask])
            )
        observed_sandwich_rank = int(np.linalg.matrix_rank(np.stack(scores)))
    unhashed = {
        "schema_version": "midogpp_utility_aligned_ensemble_capacity_report_v1",
        "context": context,
        "independent_query_count": len(unique_queries),
        "observation_count": len(values),
        "predictor_column_count": values.shape[1],
        "design_column_count": design_columns,
        "design_rank": design_rank,
        "constant_columns": list(constant_columns),
        "sandwich_rank_ceiling": sandwich_ceiling,
        "observed_sandwich_rank": observed_sandwich_rank,
        "gate_passed": not failures,
        "failures": failures,
    }
    return EnsembleCapacityReport(
        context=context,
        independent_query_count=len(unique_queries),
        observation_count=len(values),
        predictor_column_count=values.shape[1],
        design_column_count=design_columns,
        design_rank=design_rank,
        constant_columns=constant_columns,
        sandwich_rank_ceiling=sandwich_ceiling,
        observed_sandwich_rank=observed_sandwich_rank,
        gate_passed=not failures,
        failures=tuple(failures),
        report_hash=canonical_sha256(unhashed),
    )


def _validate_model_surface(surface: EnsembleFeatureSurface) -> None:
    if not isinstance(surface, EnsembleFeatureSurface):
        raise ProtocolError("Ensemble fitting requires its typed feature surface.")
    matrix = np.asarray(surface.values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape != (len(surface.rows), len(surface.feature_names))
        or not np.isfinite(matrix).all()
        or not 1 <= len(surface.feature_names) <= 2
        or surface.feature_names[0] != GLOBAL_SOURCE_CONTROL_NAME
        or len(set(surface.feature_names)) != len(surface.feature_names)
    ):
        raise ProtocolError("Ensemble model surface exceeds the locked M0/M1 capacity.")
    if len(surface.feature_names) == 1 and surface.target_local_scalar_name is not None:
        raise ProtocolError("M0 surface unexpectedly carries a target-local scalar.")
    if len(surface.feature_names) == 2 and surface.target_local_scalar_name is None:
        raise ProtocolError("M1 surface is missing its target-local scalar contract.")


def _validated_alphas(values: Sequence[float]) -> tuple[float, ...]:
    try:
        candidates = tuple(sorted(set(float(value) for value in values)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Ensemble ridge alphas are invalid.") from exc
    if not candidates or any(not np.isfinite(value) or value <= 0.0 for value in candidates):
        raise ProtocolError("Ensemble ridge alphas must be finite and positive.")
    return candidates


__all__ = ("fit_ensemble_utility_model",)
