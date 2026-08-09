"""Fixed-alpha candidate cross-fitting with strict domain-role exclusion."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.local_marginal_utility.ridge import fit_cluster_weighted_ridge
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    CENTERS,
    EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
    EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT,
    FAMILY_IDS,
    RIDGE_ALPHA,
    CrossfitFoldAudit,
    CrossfitFoldLock,
    CrossfitPredictionRow,
    ProxyCrossfitResult,
    ProxyFeatureSurface,
    ProxyUtilitySurface,
)
from .proxy_features import build_proxy_family_designs, build_proxy_utility_surface


def crossfit_proxy_families(
    feature_surface: ProxyFeatureSurface,
    utility_rows: ProxyUtilitySurface | Sequence[object],
) -> ProxyCrossfitResult:
    """Cross-fit all predeclared families with fixed ``ridge_alpha=1``.

    For a prediction of candidate ``(H,q,e)``, every training row using any of
    ``H,q,e`` in any outer-target, query, or candidate-source role is removed.
    Centering, scaling, and ridge fitting see only that cross-H training fold.
    The already
    computed within-``(H,q)`` z-primitives are label-free candidate-list
    transforms; the audit records that distinct transductive boundary instead
    of falsely describing them as learned fold transforms.
    """

    if not isinstance(feature_surface, ProxyFeatureSurface):
        raise ProtocolError("Proxy crossfit requires a typed feature surface.")
    utility = (
        utility_rows
        if isinstance(utility_rows, ProxyUtilitySurface)
        else build_proxy_utility_surface(utility_rows)
    )
    if feature_surface.row_keys != utility.row_keys:
        raise ProtocolError("Proxy feature and utility H/q/e keys do not align.")
    for feature, response in zip(feature_surface.rows, utility.rows, strict=True):
        if feature.support_partition_hash != response.support_partition_hash:
            raise ProtocolError("Proxy feature/utility support partitions drifted.")

    designs = build_proxy_family_designs(feature_surface)
    response = np.asarray(
        [row.utility_delta for row in utility.rows], dtype=np.float64
    )
    prediction_rows: list[CrossfitPredictionRow] = []
    fold_audits: list[CrossfitFoldAudit] = []
    rows = feature_surface.rows

    for family_id in FAMILY_IDS:
        design = designs[family_id]
        for prediction_index, predicted_row in enumerate(rows):
            excluded_set = {
                predicted_row.outer_target_id,
                predicted_row.query_id,
                predicted_row.candidate_source,
            }
            training_indices = tuple(
                index
                for index, row in enumerate(rows)
                if not excluded_set.intersection(row.row_key)
            )
            if len(training_indices) != EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT:
                raise ProtocolError(
                    "Strict H/q/e crossfit must leave exactly 120 training rows."
                )
            training_rows = tuple(rows[index] for index in training_indices)
            training_outers = tuple(
                sorted({row.outer_target_id for row in training_rows})
            )
            training_queries = tuple(
                sorted({row.query_id for row in training_rows})
            )
            training_sources = tuple(
                sorted({row.candidate_source for row in training_rows})
            )
            if (
                excluded_set.intersection(training_outers)
                or excluded_set.intersection(training_queries)
                or excluded_set.intersection(training_sources)
                or len(training_outers) != 6
                or len(training_queries) != 6
                or len(training_sources) != 6
            ):
                raise ProtocolError("Strict H/q/e learned-role exclusion failed.")
            training_index_array = np.asarray(training_indices, dtype=np.int64)
            training_clusters = tuple(
                f"{row.outer_target_id}::{row.query_id}" for row in training_rows
            )
            model = fit_cluster_weighted_ridge(
                design.values[training_index_array],
                response[training_index_array],
                training_clusters,
                alpha=RIDGE_ALPHA,
                feature_names=design.spec.predictor_names,
            )
            if model.training_query_clusters != tuple(sorted(set(training_clusters))):
                raise ProtocolError("Crossfit ridge outer-query cluster audit drifted.")
            prediction = float(
                model.predict(design.values[prediction_index : prediction_index + 1])[0]
            )
            if not np.isfinite(prediction):
                raise ProtocolError("Proxy crossfit prediction is non-finite.")
            excluded = tuple(center for center in CENTERS if center in excluded_set)
            fold_unhashed = {
                "schema_version": "midogpp_stage90_proxy_information_fold_audit_v1",
                "family_id": family_id,
                "predicted_row_key": list(predicted_row.row_key),
                "excluded_domain_ids": list(excluded),
                "training_row_keys": [row.row_key for row in training_rows],
                "training_outer_target_ids": list(training_outers),
                "training_query_ids": list(training_queries),
                "training_source_ids": list(training_sources),
                "training_row_count": len(training_rows),
                "ridge_alpha": RIDGE_ALPHA,
                "ridge_cluster_unit": "outer_target_query",
                "hyperparameter_selection": "none_fixed_predeclared",
                "feature_mean": model.feature_mean.tolist(),
                "feature_scale": model.feature_scale.tolist(),
                "intercept": model.intercept,
                "coefficients": model.coefficients.tolist(),
                "learned_scaling_fit_on_training_fold_only": True,
                "precomputed_candidate_list_transforms_are_label_free": True,
                "strict_H_q_e_exclusion_from_all_training_roles": True,
                "family_design_hash": design.design_hash,
            }
            fold_hash = canonical_sha256(fold_unhashed)
            audit = CrossfitFoldAudit(
                family_id=family_id,
                predicted_row_key=predicted_row.row_key,
                excluded_domain_ids=excluded,
                training_row_keys=tuple(row.row_key for row in training_rows),
                training_outer_target_ids=training_outers,
                training_query_ids=training_queries,
                training_source_ids=training_sources,
                training_row_count=len(training_rows),
                ridge_alpha=RIDGE_ALPHA,
                learned_scaling_fit_on_training_fold_only=True,
                precomputed_candidate_list_transforms_are_label_free=True,
                fold_hash=fold_hash,
            )
            prediction_unhashed = {
                "schema_version": (
                    "midogpp_stage90_proxy_information_crossfit_prediction_v1"
                ),
                "family_id": family_id,
                "outer_target_id": predicted_row.outer_target_id,
                "query_id": predicted_row.query_id,
                "candidate_source": predicted_row.candidate_source,
                "predicted_utility_delta": prediction,
                "observed_utility_delta": float(response[prediction_index]),
                "predictor_count": design.spec.predictor_count,
                "training_row_count": len(training_rows),
                "fold_hash": fold_hash,
                "response_unit": (
                    "candidate_H_q_e_exact_nine_probability_ensemble"
                ),
                "technical_seed_rows_are_independent_observations": False,
            }
            prediction_rows.append(
                CrossfitPredictionRow(
                    family_id=family_id,
                    outer_target_id=predicted_row.outer_target_id,
                    query_id=predicted_row.query_id,
                    candidate_source=predicted_row.candidate_source,
                    predicted_utility_delta=prediction,
                    observed_utility_delta=float(response[prediction_index]),
                    predictor_count=design.spec.predictor_count,
                    training_row_count=len(training_rows),
                    fold_hash=fold_hash,
                    row_hash=canonical_sha256(prediction_unhashed),
                )
            )
            fold_audits.append(audit)

    expected_fold_count = len(FAMILY_IDS) * EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT
    if len(prediction_rows) != expected_fold_count or len(fold_audits) != expected_fold_count:
        raise ProtocolError("Proxy crossfit family/fold coverage drifted.")
    fold_hashes = tuple(audit.fold_hash for audit in fold_audits)
    lock_unhashed = {
        "schema_version": "midogpp_stage90_proxy_information_crossfit_fold_lock_v1",
        "family_ids": list(FAMILY_IDS),
        "feature_surface_hash": feature_surface.surface_hash,
        "utility_surface_hash": utility.surface_hash,
        "fold_count": len(fold_audits),
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_cluster_unit": "outer_target_query",
        "hyperparameter_selection": "none_fixed_predeclared",
        "ordered_fold_hashes": list(fold_hashes),
        "strict_H_q_e_exclusion_from_all_training_roles": True,
        "scaling_fit_on_training_fold_only": True,
        "response_row_count": EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT,
        "descriptive_seed_row_count": 4_536,
        "descriptive_seed_rows_may_feed_model": False,
    }
    fold_lock = CrossfitFoldLock(
        family_ids=FAMILY_IDS,
        feature_surface_hash=feature_surface.surface_hash,
        utility_surface_hash=utility.surface_hash,
        fold_count=len(fold_audits),
        ridge_alpha=RIDGE_ALPHA,
        ordered_fold_hashes=fold_hashes,
        lock_hash=canonical_sha256(lock_unhashed),
    )
    result_unhashed = {
        "schema_version": "midogpp_stage90_proxy_information_crossfit_result_v1",
        "feature_surface_hash": feature_surface.surface_hash,
        "utility_surface_hash": utility.surface_hash,
        "crossfit_fold_lock_hash": fold_lock.lock_hash,
        "ordered_prediction_row_hashes": [row.row_hash for row in prediction_rows],
        "prediction_row_count": len(prediction_rows),
        "response_rows_are_exact_nine_candidate_endpoints": True,
        "seed_rows_are_independent_observations": False,
    }
    return ProxyCrossfitResult(
        predictions=tuple(prediction_rows),
        fold_audits=tuple(fold_audits),
        fold_lock=fold_lock,
        feature_surface_hash=feature_surface.surface_hash,
        utility_surface_hash=utility.surface_hash,
        result_hash=canonical_sha256(result_unhashed),
    )


__all__ = ("crossfit_proxy_families",)
