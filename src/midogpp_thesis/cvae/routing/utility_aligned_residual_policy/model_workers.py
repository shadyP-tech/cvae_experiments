"""Spawned, BLAS-bounded outer-target model fitting."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .ensemble_model_adapter import (
    EnsembleEndpointWorkerPayload,
    aggregate_candidate_seed_features, build_ensemble_feature_surface,
    build_ensemble_utility_policy, build_target_ensemble_feature_surfaces,
    cyclically_permute_target_scalar, derive_label_free_global_source_control,
    evaluate_ensemble_cardinality_transfer, fit_ensemble_utility_model,
    make_endpoint_worker_payload, restore_endpoint_worker_payload,
)
from ..utility_aligned_identities import CENTERS
from .config import UtilityAlignedResidualPolicyConfig
from .inputs import PolicyInputs, TargetFeatureSet


@dataclass(frozen=True)
class TargetFitResult:
    target: str
    model_payload: Mapping[str, object]
    permutation_model_payload: Mapping[str, object]
    transfer_payload: Mapping[str, object]
    permutation_transfer_payload: Mapping[str, object]
    global_policy_payload: Mapping[str, object]
    routed_policy_payload: Mapping[str, object]
    permutation_policy_payload: Mapping[str, object]


def fit_all_targets(
    config: UtilityAlignedResidualPolicyConfig,
    inputs: PolicyInputs,
    *,
    spawn_workers: bool = True,
) -> Mapping[str, TargetFitResult]:
    if not spawn_workers:
        completed = {
            target: _fit_target_from_config(target, config, inputs)
            for target in CENTERS
        }
        return MappingProxyType(completed)
    context = mp.get_context("spawn"); completed = {}
    blas_names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in blas_names}
    try:
        for name in blas_names: os.environ[name] = str(config.runtime["launch_blas_threads"])
        with ProcessPoolExecutor(max_workers=int(config.runtime["model_workers"]), mp_context=context) as pool:
            futures = {
                pool.submit(
                    _fit_target, target, inputs.inner_feature_surfaces[target],
                    make_endpoint_worker_payload(
                        inputs.ensemble_endpoint, outer_target_id=target
                    ),
                    inputs.target_features_by_target[target],
                    inputs.target_action_shift_cases_by_target[target],
                    tuple(float(value) for value in config.model["alphas"]),
                    int(config.model["permutation_seed"]),
                    float(config.model["confidence_multiplier"]),
                    float(config.model["minimum_gain"]),
                    int(config.runtime["threads_per_model_worker"]),
                ): target for target in CENTERS
            }
            for future in as_completed(futures):
                result = future.result()
                if result.target in completed: raise ProtocolError("Utility-aligned target completed twice.")
                completed[result.target] = result
    finally:
        for name, value in previous.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
    if set(completed) != set(CENTERS): raise ProtocolError("Utility-aligned target coverage drifted.")
    return MappingProxyType({target: completed[target] for target in CENTERS})


def _fit_target_from_config(
    target: str,
    config: UtilityAlignedResidualPolicyConfig,
    inputs: PolicyInputs,
) -> TargetFitResult:
    return _fit_target(
        target,
        inputs.inner_feature_surfaces[target],
        make_endpoint_worker_payload(
            inputs.ensemble_endpoint, outer_target_id=target
        ),
        inputs.target_features_by_target[target],
        inputs.target_action_shift_cases_by_target[target],
        tuple(float(value) for value in config.model["alphas"]),
        int(config.model["permutation_seed"]),
        float(config.model["confidence_multiplier"]),
        float(config.model["minimum_gain"]),
        int(config.runtime["threads_per_model_worker"]),
    )


def _fit_target(target: str, inner_features: object, endpoint_payload: EnsembleEndpointWorkerPayload, target_features: TargetFeatureSet, target_shift_cases: object, alphas: tuple[float, ...], permutation_seed: int, confidence_multiplier: float, minimum_gain: float, threads: int) -> TargetFitResult:
    try: from threadpoolctl import threadpool_limits
    except ImportError as exc: raise ProtocolError("threadpoolctl is required for bounded policy fitting.") from exc
    with threadpool_limits(limits=threads):
        if not hasattr(inner_features, "rows"):
            raise ProtocolError("Ensemble policy worker received untyped inputs.")
        utility_surface, support_shifts = restore_endpoint_worker_payload(
            endpoint_payload, outer_target_id=target
        )
        global_control = derive_label_free_global_source_control(inner_features.rows)
        source_rows = aggregate_candidate_seed_features(inner_features.rows)
        shifted_rows = aggregate_candidate_seed_features(
            inner_features.rows,
            support_action_shift_by_candidate=support_shifts,
        )
        global_surface = build_ensemble_feature_surface(
            source_rows,
            global_source_control_by_source=global_control.value_by_source,
            global_source_control_semantics=global_control.semantics,
            global_source_control_provenance_hash=global_control.provenance_hash,
        )
        routed_surface = build_ensemble_feature_surface(
            shifted_rows,
            global_source_control_by_source=global_control.value_by_source,
            global_source_control_semantics=global_control.semantics,
            global_source_control_provenance_hash=global_control.provenance_hash,
        )
        permuted_surface = cyclically_permute_target_scalar(
            routed_surface, permutation_seed=permutation_seed
        )
        global_model = fit_ensemble_utility_model(global_surface, utility_surface, alphas=alphas)
        routed_model = fit_ensemble_utility_model(routed_surface, utility_surface, alphas=alphas)
        permutation_model = fit_ensemble_utility_model(permuted_surface, utility_surface, alphas=alphas)
        transfer = evaluate_ensemble_cardinality_transfer(
            global_model, routed_model, permutation_model, utility_surface
        )
        target_production = build_target_ensemble_feature_surfaces(
            target_features.point_surface.rows, target_shift_cases, target_features.plan,
            global_source_control=global_control,
        )
        policy = build_ensemble_utility_policy(
            global_model, routed_model, permutation_model,
            target_production.point_surface, target_production.bootstrap_surfaces, transfer,
        )
    # Cross the spawn boundary only with canonical dictionaries/lists/scalars.
    # Core model objects contain MappingProxyType fold audits and are not
    # standard-pickle safe.
    return TargetFitResult(
        target,
        _ensemble_models_payload(global_model, routed_model, permutation_model, global_control),
        _ensemble_model_payload(permutation_model),
        transfer.to_payload(),
        transfer.to_payload(),
        _ensemble_policy_payload(policy, "G", global_model, target_production),
        _ensemble_policy_payload(policy, "R", routed_model, target_production),
        _ensemble_policy_payload(policy, "P", permutation_model, target_production),
    )


def _ensemble_model_payload(model: object) -> dict[str, object]:
    return {
        "outer_target_id": model.outer_target_id,
        "feature_names": list(model.feature_names),
        "selected_alpha": model.selected_alpha,
        "routing_tuning_endpoint": model.routing_tuning_endpoint,
        "routing_loss_by_alpha": {str(key): value for key, value in model.routing_loss_by_alpha.items()},
        "selected_alpha_by_heldout_query": dict(model.selected_alpha_by_heldout_query),
        "candidate_models": {source: _ridge_payload(value) for source, value in model.candidate_models.items()},
        "candidate_capacity_reports": {source: value.to_payload() for source, value in model.candidate_capacity_reports.items()},
        "crossfit_row_keys": [list(key) for key in model.crossfit_row_keys],
        "crossfit_prediction_sha256": __import__("midogpp_thesis.cvae.routing.residual_topup.hashing", fromlist=["array_sha256"]).array_sha256(model.crossfit_predictions),
        "feature_surface_hash": model.feature_surface_hash,
        "utility_surface_hash": model.utility_surface_hash,
        "permutation_seed": model.permutation_seed,
        "model_hash": model.model_hash,
    }


def _ensemble_models_payload(global_model: object, routed_model: object, permutation_model: object, control: object) -> dict[str, object]:
    routed = _ensemble_model_payload(routed_model)
    reports = {
        role: [report.to_payload() for report in model.candidate_capacity_reports.values()]
        for role, model in {"G": global_model, "R": routed_model, "P": permutation_model}.items()
    }
    routed["global_model"] = _ensemble_model_payload(global_model)
    routed["permutation_model"] = _ensemble_model_payload(permutation_model)
    routed["global_source_control"] = control.to_payload()
    routed["model_capacity_report"] = {
        "gate_passed": all(item["gate_passed"] for values in reports.values() for item in values),
        "reports_by_role": reports,
        "report_hash": __import__("midogpp_thesis.cvae.routing.residual_topup.hashing", fromlist=["canonical_sha256"]).canonical_sha256(reports),
    }
    return routed


def _ensemble_policy_payload(policy: object, role: str, model: object, target_production: object) -> dict[str, object]:
    selected = policy.role_selected_source[role]
    action = policy.role_selected_action[role]
    candidates = tuple(model.candidate_models)
    proposed = {"G": "G_delta", "R": "R", "P": "P"}[role]
    candidate = min(policy.role_prediction_by_source[role], key=lambda source: (-policy.role_prediction_by_source[role][source], source))
    fallback = action == "B"
    return {
        "schema_version": "midogpp_utility_aligned_ensemble_policy_v1",
        "target_id": policy.target_id, "role": role, "candidate_sources": list(candidates),
        "proposed_action_id": proposed, "action_id": "B" if fallback else proposed,
        "proposed_source": candidate, "selected_source": selected,
        "predicted_gain": policy.role_prediction_by_source[role][candidate],
        "standard_error": policy.role_combined_standard_error_by_source[role][candidate],
        "lower_confidence_bound": policy.role_lower_confidence_bound_by_source[role][candidate],
        "support_case_count": target_production.point_surface.rows[0].support_case_count,
        "support_bootstrap_replicates": len(target_production.bootstrap_surfaces),
        "used_exact_base_fallback": fallback,
        "fallback_reason": policy.fallback_reason if role == "R" else ("role_gain_lcb_not_positive" if fallback else None),
        "model_hash": model.model_hash, "feature_surface_hash": target_production.point_surface.surface_hash,
        "cardinality_eligibility_hash": policy.cardinality_transfer_hash,
        "policy_hash": policy.policy_hash,
        "ensemble_policy": policy.to_payload(),
    }


def _ridge_payload(model: object) -> dict[str, object]:
    return {
        "feature_names": list(model.feature_names), "alpha": model.alpha,
        "feature_mean": np.asarray(model.feature_mean).tolist(),
        "feature_scale": np.asarray(model.feature_scale).tolist(),
        "intercept": model.intercept,
        "coefficients": np.asarray(model.coefficients).tolist(),
        "coefficient_covariance": np.asarray(model.coefficient_covariance).tolist(),
        "residual_variance": model.residual_variance,
        "training_query_clusters": list(model.training_query_clusters),
        "observation_count": model.observation_count, "effective_rank": model.effective_rank,
    }


def _transfer_payload(result: object) -> dict[str, object]:
    return {
        "outer_target_id": result.outer_target_id, "candidate_sources": list(result.candidate_sources),
        "training_candidate_count": result.training_candidate_count,
        "evaluation_candidate_count": result.evaluation_candidate_count,
        "deployment_candidate_count": result.deployment_candidate_count,
        "global_metrics": vars(result.global_metrics), "interaction_metrics": vars(result.interaction_metrics),
        "top1_delta": result.top1_delta, "spearman_delta": result.spearman_delta,
        "normalized_gap_reduction": result.normalized_gap_reduction,
        "normalized_gap_reduction_lower_bound": result.normalized_gap_reduction_lower_bound,
        "pairwise_accuracy_delta": result.pairwise_accuracy_delta,
        "selected_utility_delta": result.selected_utility_delta,
        "selected_utility_delta_lower_bound": result.selected_utility_delta_lower_bound,
        "global_gate_passed": result.global_gate_passed, "global_gate_reason": result.global_gate_reason,
        "eligibility_passed": result.eligibility_passed, "eligibility_reason": result.eligibility_reason,
        "claim_role": result.claim_role, "model_hash": result.model_hash, "result_hash": result.result_hash,
    }


__all__ = ("TargetFitResult", "fit_all_targets")
