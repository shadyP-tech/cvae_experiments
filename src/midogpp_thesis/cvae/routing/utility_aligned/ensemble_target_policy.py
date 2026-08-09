"""Fresh-target G/R/P policy construction with exact-B fallback."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ..residual_topup.hashing import canonical_sha256
from .ensemble_features import cyclically_permute_target_scalar
from .ensemble_feature_contracts import EnsembleFeatureSurface
from .ensemble_model_contracts import (
    EnsembleCardinalityTransferResult,
    EnsembleUtilityModel,
)
from .ensemble_policy_contracts import (
    ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS,
    ENSEMBLE_BASE_ROLE,
    ENSEMBLE_GAIN_LCB_MULTIPLIER,
    ENSEMBLE_GLOBAL_ROLE,
    ENSEMBLE_PERMUTATION_ROLE,
    ENSEMBLE_ROUTED_ROLE,
    ENSEMBLE_TARGET_SEED_SPREAD_ROLE,
    EnsembleUtilityPolicy,
)
from .ensemble_prediction import (
    permuted_target_seed_spread,
    predict_target_candidate_distributions,
    predict_target_candidates,
)
from .row_contracts import (
    INNER_CANDIDATE_COUNT,
    MIN_SUPPORT_BOOTSTRAP_REPLICATES,
    TARGET_CANDIDATE_COUNT,
    TARGET_ROLE,
)


def build_ensemble_utility_policy(
    global_model: EnsembleUtilityModel,
    routed_model: EnsembleUtilityModel,
    permutation_model: EnsembleUtilityModel,
    point_features: EnsembleFeatureSurface,
    bootstrap_features: Sequence[EnsembleFeatureSurface],
    cardinality_transfer: EnsembleCardinalityTransferResult,
) -> EnsembleUtilityPolicy:
    """Build target G/R/P diagnostics and an R-or-exact-B decision.

    Target labels and utility are absent from this API.  Failure of the
    source-inner transfer/capacity gate returns exact B with no selected source.
    """

    models = {
        ENSEMBLE_GLOBAL_ROLE: global_model,
        ENSEMBLE_ROUTED_ROLE: routed_model,
        ENSEMBLE_PERMUTATION_ROLE: permutation_model,
    }
    if (
        not isinstance(point_features, EnsembleFeatureSurface)
        or point_features.role != TARGET_ROLE
        or len(point_features.rows) != TARGET_CANDIDATE_COUNT
        or len(point_features.feature_names) != 2
        or point_features.permutation_seed is not None
    ):
        raise ProtocolError("Ensemble policy requires one unpermuted target M1 point surface.")
    target = point_features.outer_target_id
    if (
        cardinality_transfer.outer_target_id != target
        or any(model.outer_target_id != target for model in models.values())
        or cardinality_transfer.source_inner_candidate_count != INNER_CANDIDATE_COUNT
        or cardinality_transfer.deployment_candidate_count != TARGET_CANDIDATE_COUNT
    ):
        raise ProtocolError("Ensemble policy target/cardinality binding drifted.")
    bootstrap = tuple(bootstrap_features)
    if len(bootstrap) < MIN_SUPPORT_BOOTSTRAP_REPLICATES:
        raise ProtocolError("Ensemble policy requires at least 32 case-bootstrap surfaces.")
    expected_sources = point_features.candidate_sources
    for surface in bootstrap:
        if (
            not isinstance(surface, EnsembleFeatureSurface)
            or surface.role != TARGET_ROLE
            or surface.outer_target_id != target
            or surface.candidate_sources != expected_sources
            or surface.feature_names != point_features.feature_names
            or surface.permutation_seed is not None
            or len(surface.rows) != TARGET_CANDIDATE_COUNT
        ):
            raise ProtocolError("Ensemble bootstrap target feature surface drifted.")
    if len({surface.surface_hash for surface in bootstrap}) != len(bootstrap):
        raise ProtocolError("Ensemble bootstrap surfaces must carry unique provenance.")
    if permutation_model.permutation_seed is None:
        raise ProtocolError("Permutation control model has no cyclic permutation seed.")
    permuted_point = cyclically_permute_target_scalar(
        point_features, permutation_seed=permutation_model.permutation_seed
    )
    point_distributions = {
        ENSEMBLE_GLOBAL_ROLE: predict_target_candidate_distributions(
            global_model, point_features
        ),
        ENSEMBLE_ROUTED_ROLE: predict_target_candidate_distributions(
            routed_model, point_features
        ),
        ENSEMBLE_PERMUTATION_ROLE: predict_target_candidate_distributions(
            permutation_model, permuted_point
        ),
    }
    prediction_by_role = {
        role: {source: values[0] for source, values in by_source.items()}
        for role, by_source in point_distributions.items()
    }
    model_standard_error_by_role = {
        role: {source: values[1] for source, values in by_source.items()}
        for role, by_source in point_distributions.items()
    }
    candidate_argmax_by_role = {
        role: min(values, key=lambda source: (-values[source], source))
        for role, values in prediction_by_role.items()
    }
    bootstrap_predictions = {
        role: {source: [] for source in expected_sources}
        for role in (
            ENSEMBLE_GLOBAL_ROLE,
            ENSEMBLE_ROUTED_ROLE,
            ENSEMBLE_PERMUTATION_ROLE,
        )
    }
    for surface in bootstrap:
        permuted_surface = cyclically_permute_target_scalar(
            surface, permutation_seed=permutation_model.permutation_seed
        )
        predicted_by_role = {
            ENSEMBLE_GLOBAL_ROLE: predict_target_candidates(global_model, surface),
            ENSEMBLE_ROUTED_ROLE: predict_target_candidates(routed_model, surface),
            ENSEMBLE_PERMUTATION_ROLE: predict_target_candidates(
                permutation_model, permuted_surface
            ),
        }
        for role, predicted in predicted_by_role.items():
            for source, value in predicted.items():
                bootstrap_predictions[role][source].append(value)
    bootstrap_mean_by_role = {
        role: {
            source: float(np.mean(values, dtype=np.float64))
            for source, values in by_source.items()
        }
        for role, by_source in bootstrap_predictions.items()
    }
    bootstrap_sd_by_role = {
        role: {
            source: float(np.std(values, ddof=0, dtype=np.float64))
            for source, values in by_source.items()
        }
        for role, by_source in bootstrap_predictions.items()
    }
    routed_seed_spread = {
        row.candidate_source: float(row.target_local_scalar_seed_standard_deviation)
        for row in point_features.rows
        if row.target_local_scalar_seed_standard_deviation is not None
    }
    if set(routed_seed_spread) != set(expected_sources):
        raise ProtocolError("Target action gate is missing full nine-seed scalar spread.")
    seed_spread_by_role = {
        ENSEMBLE_GLOBAL_ROLE: {source: 0.0 for source in expected_sources},
        ENSEMBLE_ROUTED_ROLE: routed_seed_spread,
        ENSEMBLE_PERMUTATION_ROLE: permuted_target_seed_spread(
            point_features, permutation_seed=permutation_model.permutation_seed
        ),
    }
    combined_se_by_role = {
        role: {
            source: float(
                np.sqrt(
                    model_standard_error_by_role[role][source] ** 2
                    + bootstrap_sd_by_role[role][source] ** 2
                )
            )
            for source in expected_sources
        }
        for role in prediction_by_role
    }
    lower_bound_by_role = {
        role: {
            source: prediction_by_role[role][source]
            - ENSEMBLE_GAIN_LCB_MULTIPLIER * combined_se_by_role[role][source]
            for source in expected_sources
        }
        for role in prediction_by_role
    }
    role_selected_source: dict[str, str | None] = {}
    role_selected_action: dict[str, str] = {}
    for role, candidate in candidate_argmax_by_role.items():
        positive_lcb = lower_bound_by_role[role][candidate] > 0.0
        allowed = positive_lcb and (
            role != ENSEMBLE_ROUTED_ROLE
            or cardinality_transfer.authorized_for_target_policy
        )
        role_selected_action[role] = role if allowed else ENSEMBLE_BASE_ROLE
        role_selected_source[role] = candidate if allowed else None
    selected_role = role_selected_action[ENSEMBLE_ROUTED_ROLE]
    selected_source = role_selected_source[ENSEMBLE_ROUTED_ROLE]
    fallback = selected_role == ENSEMBLE_BASE_ROLE
    routed_candidate = candidate_argmax_by_role[ENSEMBLE_ROUTED_ROLE]
    if not cardinality_transfer.authorized_for_target_policy:
        fallback_reason = "source_inner_cardinality_or_capacity_gate_failed"
    elif lower_bound_by_role[ENSEMBLE_ROUTED_ROLE][routed_candidate] <= 0.0:
        fallback_reason = "routed_selected_gain_lcb_not_positive"
    else:
        fallback_reason = None
    unhashed = {
        "schema_version": "midogpp_utility_aligned_ensemble_policy_v1",
        "target_id": target,
        "selected_action_role": selected_role,
        "selected_source": selected_source,
        "exact_b_fallback": fallback,
        "fallback_reason": fallback_reason,
        "role_selected_source": role_selected_source,
        "role_selected_action": role_selected_action,
        "role_prediction_by_source": prediction_by_role,
        "role_model_standard_error_by_source": model_standard_error_by_role,
        "role_bootstrap_standard_deviation_by_source": bootstrap_sd_by_role,
        "role_target_scalar_seed_standard_deviation_by_source": (
            seed_spread_by_role
        ),
        "role_combined_standard_error_by_source": combined_se_by_role,
        "role_lower_confidence_bound_by_source": lower_bound_by_role,
        "gain_lcb_multiplier": ENSEMBLE_GAIN_LCB_MULTIPLIER,
        "bootstrap_dispersion_divided_by_seed_repeat_sqrt": False,
        "authorization_uncertainty_components": list(
            ENSEMBLE_AUTHORIZATION_UNCERTAINTY_COMPONENTS
        ),
        "target_scalar_seed_spread_role": ENSEMBLE_TARGET_SEED_SPREAD_ROLE,
        "target_scalar_seed_spread_enters_combined_standard_error": False,
        "routed_bootstrap_mean_by_source": bootstrap_mean_by_role[
            ENSEMBLE_ROUTED_ROLE
        ],
        "routed_bootstrap_standard_deviation_by_source": bootstrap_sd_by_role[
            ENSEMBLE_ROUTED_ROLE
        ],
        "point_feature_surface_hash": point_features.surface_hash,
        "bootstrap_feature_surface_hashes": [
            surface.surface_hash for surface in bootstrap
        ],
        "cardinality_transfer_hash": cardinality_transfer.transfer_hash,
        "target_labels_used": False,
        "target_utility_used": False,
        "seed_rows_are_independent_observations": False,
    }
    return EnsembleUtilityPolicy(
        target_id=target,
        selected_action_role=selected_role,
        selected_source=selected_source,
        exact_b_fallback=fallback,
        fallback_reason=fallback_reason,
        role_selected_source=role_selected_source,
        role_selected_action=role_selected_action,
        role_prediction_by_source=prediction_by_role,
        role_model_standard_error_by_source=model_standard_error_by_role,
        role_bootstrap_standard_deviation_by_source=bootstrap_sd_by_role,
        role_target_scalar_seed_standard_deviation_by_source=seed_spread_by_role,
        role_combined_standard_error_by_source=combined_se_by_role,
        role_lower_confidence_bound_by_source=lower_bound_by_role,
        routed_bootstrap_mean_by_source=bootstrap_mean_by_role[
            ENSEMBLE_ROUTED_ROLE
        ],
        routed_bootstrap_standard_deviation_by_source=bootstrap_sd_by_role[
            ENSEMBLE_ROUTED_ROLE
        ],
        point_feature_surface_hash=point_features.surface_hash,
        bootstrap_feature_surface_hashes=tuple(
            surface.surface_hash for surface in bootstrap
        ),
        cardinality_transfer_hash=cardinality_transfer.transfer_hash,
        policy_hash=canonical_sha256(unhashed),
    )




__all__ = ("build_ensemble_utility_policy",)
