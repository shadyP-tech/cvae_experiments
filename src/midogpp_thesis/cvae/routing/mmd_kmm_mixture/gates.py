"""Fail-closed stability and duplicate-direction gates for MMD/KMM."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .config import KMMGateConfig, KMMOptimizationConfig, PriorControlConfig
from .contracts import (
    DirectionIdentityAudit,
    EnergyDirectionReference,
    KMMRouteDecision,
    KMMWeightSolution,
    KernelMeanProblem,
    StabilityAudit,
)
from .kmm import solve_kmm_weights


def route_mmd_kmm(
    base_problem: KernelMeanProblem,
    *,
    support_case_problems: Mapping[str, KernelMeanProblem],
    training_seed_problems: Mapping[str, KernelMeanProblem],
    generation_seed_problems: Mapping[str, KernelMeanProblem],
    prior_sensitivity_problems: Mapping[str, KernelMeanProblem],
    energy_direction_reference: EnergyDirectionReference,
    prior_control: PriorControlConfig,
    optimization: KMMOptimizationConfig,
    gates: KMMGateConfig,
) -> KMMRouteDecision:
    """Solve one proxy route and apply every predeclared abstention gate."""

    if base_problem.prior_control_hash != prior_control.state_hash:
        raise ProtocolError("MMD/KMM route used a different prior-control state.")
    _validate_energy_reference(base_problem, energy_direction_reference)
    base = solve_kmm_weights(base_problem, optimization)
    groups = (
        (
            "support_case",
            support_case_problems,
            tuple(base_problem.protocol.support_case_ids),
            gates.maximum_support_l1,
        ),
        (
            "training_seed",
            training_seed_problems,
            tuple(str(seed) for seed in base_problem.protocol.training_seeds),
            gates.maximum_training_seed_l1,
        ),
        (
            "generation_seed",
            generation_seed_problems,
            tuple(str(seed) for seed in base_problem.protocol.generation_seeds),
            gates.maximum_generation_seed_l1,
        ),
        (
            "class_prior_sensitivity",
            prior_sensitivity_problems,
            tuple(
                _prior_variant_id(value)
                for value in (
                    prior_control.reference_positive_prior,
                    *prior_control.sensitivity_positive_priors,
                )
            ),
            gates.maximum_prior_sensitivity_l1,
        ),
    )
    audits: list[StabilityAudit] = []
    for axis, problems, expected_ids, max_l1 in groups:
        observed_ids = tuple(sorted(str(key) for key in problems))
        complete = observed_ids == tuple(sorted(expected_ids))
        if not complete:
            audits.append(
                StabilityAudit(
                    axis=axis,
                    variant_ids=observed_ids,
                    maximum_l1_distance=float("inf"),
                    minimum_direction_cosine=-1.0,
                    passed=False,
                    failure_reason="incomplete_stability_grid",
                )
            )
            continue
        variants: dict[str, KMMWeightSolution] = {}
        family_valid = True
        for variant_id, problem in sorted(
            problems.items(), key=lambda item: str(item[0])
        ):
            if axis == "class_prior_sensitivity" and not _prior_variant_matches(
                str(variant_id), problem, base_problem, prior_control
            ):
                family_valid = False
                break
            if not _same_problem_family(
                base_problem,
                problem,
                variant_axis=axis,
            ):
                family_valid = False
                break
            variants[str(variant_id)] = solve_kmm_weights(problem, optimization)
        if not family_valid:
            raise ProtocolError("MMD/KMM stability problem crossed a routing family.")
        audits.append(
            audit_weight_stability(
                base,
                variants,
                axis=axis,
                maximum_l1=float(max_l1),
                minimum_direction_cosine=float(gates.minimum_direction_cosine),
            )
        )

    identity = audit_direction_identity(
        base,
        energy_direction_reference,
        duplicate_direction_cosine=float(gates.duplicate_direction_cosine),
        duplicate_weight_l1=float(gates.duplicate_weight_l1),
    )
    fallback_reason = base.fallback_reason if base.used_uniform_fallback else None
    if fallback_reason is None:
        failed = next((audit for audit in audits if not audit.passed), None)
        if failed is not None:
            fallback_reason = f"{failed.axis}_{failed.failure_reason}_uniform"
    if fallback_reason is None and identity.duplicate:
        fallback_reason = "duplicate_energy_direction_uniform"
    uniform = dict(base.uniform_weights)
    final = uniform if fallback_reason is not None else dict(base.weights)
    return KMMRouteDecision(
        candidate_sources=base.candidate_sources,
        base_solution=base,
        final_weights=final,
        used_uniform_fallback=fallback_reason is not None,
        fallback_reason=fallback_reason,
        stability_audits=tuple(audits),
        direction_identity=identity,
        claim_role="proxy_compatibility_only",
        downstream_utility_claimed=False,
        promotion_eligible=False,
        target_labels_used=False,
        stage90_inputs_used=False,
    )


def audit_weight_stability(
    base: KMMWeightSolution,
    variants: Mapping[str, KMMWeightSolution],
    *,
    axis: str,
    maximum_l1: float,
    minimum_direction_cosine: float,
) -> StabilityAudit:
    """Compare every label-free variant with the all-replica solution."""

    if not variants:
        raise ProtocolError("MMD/KMM stability audit requires variants.")
    sources = base.candidate_sources
    base_weights = _weight_vector(base.weights, sources)
    uniform = _weight_vector(base.uniform_weights, sources)
    l1_values: list[float] = []
    cosines: list[float] = []
    fallback_variant = False
    for solution in variants.values():
        if solution.candidate_sources != sources:
            raise ProtocolError("MMD/KMM stability weights use different sources.")
        variant = _weight_vector(solution.weights, sources)
        l1_values.append(float(np.abs(variant - base_weights).sum()))
        cosines.append(_direction_cosine(base_weights - uniform, variant - uniform))
        fallback_variant = fallback_variant or solution.used_uniform_fallback
    max_l1 = max(l1_values)
    min_cosine = min(cosines)
    passed = (
        not base.used_uniform_fallback
        and not fallback_variant
        and max_l1 <= float(maximum_l1)
        and min_cosine >= float(minimum_direction_cosine)
    )
    reason = None
    if base.used_uniform_fallback or fallback_variant:
        reason = "variant_uniform_fallback"
    elif max_l1 > float(maximum_l1):
        reason = "l1_instability"
    elif min_cosine < float(minimum_direction_cosine):
        reason = "direction_instability"
    return StabilityAudit(
        axis=str(axis),
        variant_ids=tuple(sorted(str(key) for key in variants)),
        maximum_l1_distance=max_l1,
        minimum_direction_cosine=min_cosine,
        passed=passed,
        failure_reason=reason,
    )


def audit_direction_identity(
    solution: KMMWeightSolution,
    reference: EnergyDirectionReference,
    *,
    duplicate_direction_cosine: float,
    duplicate_weight_l1: float,
) -> DirectionIdentityAudit:
    """Reject a numerical re-expression of the previous energy route."""

    sources = solution.candidate_sources
    reference_weights = _weight_vector(reference.weights, sources)
    if (
        np.any(reference_weights < 0.0)
        or not np.isfinite(reference_weights).all()
        or not np.isclose(
            reference_weights.sum(), 1.0, rtol=0.0, atol=1e-10
        )
    ):
        raise ProtocolError("Energy-reference weights violate the simplex.")
    weights = _weight_vector(solution.weights, sources)
    uniform = _weight_vector(solution.uniform_weights, sources)
    cosine = _direction_cosine(weights - uniform, reference_weights - uniform)
    l1_distance = float(np.abs(weights - reference_weights).sum())
    duplicate = (
        not solution.used_uniform_fallback
        and (
            cosine >= float(duplicate_direction_cosine)
            or l1_distance <= float(duplicate_weight_l1)
        )
    )
    return DirectionIdentityAudit(
        reference_role="label_free_calibrated_energy_path",
        direction_cosine=cosine,
        weight_l1_distance=l1_distance,
        duplicate=duplicate,
    )


def _validate_energy_reference(
    problem: KernelMeanProblem,
    reference: EnergyDirectionReference,
) -> None:
    protocol = problem.protocol
    if (
        not isinstance(reference, EnergyDirectionReference)
        or reference.target_center != protocol.target_center
        or reference.candidate_sources != problem.candidate_sources
        or reference.support_partition_hash != protocol.support_partition_hash
        or reference.common_frame_hash != problem.common_frame_hash
        or reference.preprocessing_hash != problem.preprocessing_hash
        or reference.candidate_pool_fit_hash != problem.candidate_pool_fit_hash
        or reference.kernel_map_hash != problem.kernel_map_hash
        or reference.training_seeds != protocol.training_seeds
        or reference.generation_seeds != protocol.generation_seeds
    ):
        raise ProtocolError(
            "Energy-direction reference crossed the frozen routing context."
        )


def _same_problem_family(
    left: KernelMeanProblem,
    right: KernelMeanProblem,
    *,
    variant_axis: str,
) -> bool:
    allow_prior_variant = variant_axis == "class_prior_sensitivity"
    prior_valid = left.prior_family_hash == right.prior_family_hash and (
        left.prior_control_hash == right.prior_control_hash
    ) and (
        allow_prior_variant
        or (
            right.prior_state_hash == left.prior_state_hash
            and right.prior_sensitivity_positive_prior
            == left.prior_sensitivity_positive_prior
        )
    )
    return (
        left.protocol == right.protocol
        and left.candidate_sources == right.candidate_sources
        and left.protocol.target_center == right.protocol.target_center
        and left.common_frame_hash == right.common_frame_hash
        and left.kernel_map_hash == right.kernel_map_hash
        and left.preprocessing_hash == right.preprocessing_hash
        and left.candidate_pool_fit_hash == right.candidate_pool_fit_hash
        and left.kernel_transform_role == right.kernel_transform_role
        and (
            variant_axis == "support_case"
            or left.target_kernel_feature_sha256
            == right.target_kernel_feature_sha256
        )
        and (
            variant_axis in {"support_case", "class_prior_sensitivity"}
            or left.target_responsibility_sha256
            == right.target_responsibility_sha256
        )
        and prior_valid
    )


def _prior_variant_matches(
    variant_id: str,
    problem: KernelMeanProblem,
    base: KernelMeanProblem,
    config: PriorControlConfig,
) -> bool:
    reference_id = _prior_variant_id(config.reference_positive_prior)
    if variant_id == reference_id:
        return (
            problem.prior_sensitivity_positive_prior is None
            and problem.prior_state_hash == base.prior_state_hash
        )
    expected = {
        _prior_variant_id(value): float(value)
        for value in config.sensitivity_positive_priors
    }
    return (
        variant_id in expected
        and problem.prior_sensitivity_positive_prior is not None
        and np.isclose(
            problem.prior_sensitivity_positive_prior,
            expected[variant_id],
            rtol=0.0,
            atol=0.0,
        )
    )


def _prior_variant_id(value: float) -> str:
    return f"positive_prior_{float(value):.12g}"


def _weight_vector(
    weights: Mapping[str, float], sources: tuple[str, ...]
) -> np.ndarray:
    normalized = {str(key): float(value) for key, value in weights.items()}
    if set(normalized) != set(sources):
        raise ProtocolError("MMD/KMM weight keys do not match candidate sources.")
    values = np.asarray([normalized[source] for source in sources], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ProtocolError("MMD/KMM weights must be finite.")
    return values


def _direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-15 or right_norm <= 1e-15:
        return 1.0 if left_norm <= 1e-15 and right_norm <= 1e-15 else -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


__all__ = (
    "audit_direction_identity",
    "audit_weight_stability",
    "route_mmd_kmm",
)
