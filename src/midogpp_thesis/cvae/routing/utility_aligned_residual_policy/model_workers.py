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
from ..residual_topup.hashing import array_sha256
from ..utility_aligned import (
    build_utility_aligned_policy, fit_utility_aligned_models,
    nested_cardinality_transfer_evaluation,
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
                    inputs.exact_utility, inputs.target_features_by_target[target],
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
        inputs.exact_utility,
        inputs.target_features_by_target[target],
        tuple(float(value) for value in config.model["alphas"]),
        int(config.model["permutation_seed"]),
        float(config.model["confidence_multiplier"]),
        float(config.model["minimum_gain"]),
        int(config.runtime["threads_per_model_worker"]),
    )


def _fit_target(target: str, inner_features: object, exact_utility: object, target_features: TargetFeatureSet, alphas: tuple[float, ...], permutation_seed: int, confidence_multiplier: float, minimum_gain: float, threads: int) -> TargetFitResult:
    try: from threadpoolctl import threadpool_limits
    except ImportError as exc: raise ProtocolError("threadpoolctl is required for bounded policy fitting.") from exc
    with threadpool_limits(limits=threads):
        models = fit_utility_aligned_models(inner_features, exact_utility, alphas=alphas)
        permutation = fit_utility_aligned_models(inner_features, exact_utility, alphas=alphas, permutation_seed=permutation_seed)
        transfer = nested_cardinality_transfer_evaluation(models, inner_features, exact_utility)
        permutation_transfer = nested_cardinality_transfer_evaluation(permutation, inner_features, exact_utility)
        global_policy = build_utility_aligned_policy(models, target_features.point_surface, transfer, global_only=True, confidence_multiplier=confidence_multiplier, minimum_gain=minimum_gain)
        routed = build_utility_aligned_policy(models, target_features.point_surface, transfer, confidence_multiplier=confidence_multiplier, minimum_gain=minimum_gain, support_bootstrap_features=target_features.bootstrap_surfaces, case_bootstrap_plan=target_features.plan)
        control = build_utility_aligned_policy(permutation, target_features.point_surface, permutation_transfer, confidence_multiplier=confidence_multiplier, minimum_gain=minimum_gain, support_bootstrap_features=target_features.bootstrap_surfaces, case_bootstrap_plan=target_features.plan)
    # Cross the spawn boundary only with canonical dictionaries/lists/scalars.
    # Core model objects contain MappingProxyType fold audits and are not
    # standard-pickle safe.
    return TargetFitResult(
        target,
        _model_payload(models),
        _model_payload(permutation),
        _transfer_payload(transfer),
        _transfer_payload(permutation_transfer),
        global_policy.to_payload(),
        routed.to_payload(),
        control.to_payload(),
    )


def _model_payload(models: object) -> dict[str, object]:
    return {
        "outer_target_id": models.outer_target_id,
        "candidate_sources": list(models.candidate_sources),
        "global_model": _ridge_payload(models.global_model),
        "interaction_model": _ridge_payload(models.interaction_model),
        "global_crossfit_hash": models.global_crossfit.crossfit_hash,
        "interaction_crossfit_hash": models.interaction_crossfit.crossfit_hash,
        "global_crossfit_prediction_sha256": array_sha256(models.global_crossfit.predictions),
        "interaction_crossfit_prediction_sha256": array_sha256(models.interaction_crossfit.predictions),
        "feature_surface_hash": models.feature_surface_hash,
        "utility_surface_hash": models.utility_surface_hash,
        "permutation_seed": models.permutation_seed,
        "model_hash": models.model_hash,
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
