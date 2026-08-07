"""Class-conditional contrast MMD problems and conservative route gates.

The public problem is represented as one augmented finite-dimensional MMD
problem so the existing convex KMM solver, feasibility checks, seed audits,
and energy-direction identity gate remain unchanged.  No labels or downstream
utility enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .config import KMMGateConfig, KMMOptimizationConfig, PriorControlConfig
from .contracts import (
    EnergyDirectionReference,
    KMMRouteDecision,
    KernelMeanProblem,
    MMDKMMProtocol,
    SourceKernelReplica,
    TargetSupportKernelFeatures,
    readonly_matrix,
)
from .gates import route_mmd_kmm
from .moments import build_kernel_mean_problem
from .prior import shift_source_only_prior_prediction


CONDITIONAL_PROXY_FAMILY = "class_conditional_contrast_mmd_kmm"


@dataclass(frozen=True)
class ConditionalContrastConfig:
    """Frozen conditional objective and label-free abstention controls."""

    class_weights: tuple[float, float]
    contrast_weight: float
    maximum_uniform_l1: float
    minimum_soft_class_mass_per_case: float
    minimum_soft_class_effective_rows_per_case: float
    component_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        weights = tuple(float(value) for value in self.class_weights)
        numeric = (
            *weights,
            float(self.contrast_weight),
            float(self.maximum_uniform_l1),
            float(self.minimum_soft_class_mass_per_case),
            float(self.minimum_soft_class_effective_rows_per_case),
            float(self.component_tolerance),
        )
        if (
            len(weights) != 2
            or any(not math.isfinite(value) for value in numeric)
            or any(value <= 0.0 for value in weights)
            or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or float(self.contrast_weight) <= 0.0
            or not 0.0 < float(self.maximum_uniform_l1) <= 2.0
            or float(self.minimum_soft_class_mass_per_case) <= 0.0
            or float(self.minimum_soft_class_effective_rows_per_case) <= 0.0
            or float(self.component_tolerance) < 0.0
        ):
            raise ProtocolError("Conditional contrast-MMD configuration is invalid.")
        object.__setattr__(self, "class_weights", weights)


@dataclass(frozen=True)
class ConditionalContrastProblem:
    """Augmented KMM problem plus reconstructible conditional components."""

    kernel_problem: KernelMeanProblem
    source_class_kernel_means: np.ndarray
    target_class_kernel_means: np.ndarray
    class_weights: tuple[float, float]
    contrast_weight: float
    soft_class_mass_by_case: Mapping[str, tuple[float, float]]
    soft_class_effective_rows_by_case: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        source = np.asarray(self.source_class_kernel_means, dtype=np.float64)
        target = readonly_matrix(
            self.target_class_kernel_means, "target conditional kernel means"
        )
        n_sources = len(self.kernel_problem.candidate_sources)
        if (
            self.kernel_problem.proxy_family != CONDITIONAL_PROXY_FAMILY
            or source.ndim != 3
            or source.shape[:2] != (n_sources, 2)
            or target.shape != (2, source.shape[2])
            or tuple(float(value) for value in self.class_weights) != self.class_weights
            or len(self.class_weights) != 2
            or not math.isclose(sum(self.class_weights), 1.0, abs_tol=1e-12)
            or float(self.contrast_weight) <= 0.0
        ):
            raise ProtocolError("Conditional contrast-MMD problem is invalid.")
        source_copy = np.array(source, copy=True)
        source_copy.setflags(write=False)
        masses = _quality_mapping(self.soft_class_mass_by_case, "soft class mass")
        effective = _quality_mapping(
            self.soft_class_effective_rows_by_case, "soft class effective rows"
        )
        expected_cases = set(self.kernel_problem.protocol.support_case_ids)
        observed_cases = set(masses)
        if (
            not observed_cases
            or observed_cases != set(effective)
            or not observed_cases.issubset(expected_cases)
        ):
            raise ProtocolError("Conditional support-quality case coverage drifted.")
        object.__setattr__(self, "source_class_kernel_means", source_copy)
        object.__setattr__(self, "target_class_kernel_means", target)
        object.__setattr__(self, "soft_class_mass_by_case", MappingProxyType(masses))
        object.__setattr__(
            self, "soft_class_effective_rows_by_case", MappingProxyType(effective)
        )

    def component_losses(self, weights: Mapping[str, float]) -> dict[str, float]:
        sources = self.kernel_problem.candidate_sources
        if set(weights) != set(sources):
            raise ProtocolError("Conditional component weights use the wrong sources.")
        vector = np.asarray([float(weights[source]) for source in sources])
        if (
            not np.isfinite(vector).all()
            or np.any(vector < 0.0)
            or not np.isclose(vector.sum(), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ProtocolError("Conditional component weights violate the simplex.")
        mixed = np.einsum("s,scd->cd", vector, self.source_class_kernel_means)
        discrepancy = mixed - self.target_class_kernel_means
        class_zero = float(self.class_weights[0]) * float(
            np.dot(discrepancy[0], discrepancy[0])
        )
        class_one = float(self.class_weights[1]) * float(
            np.dot(discrepancy[1], discrepancy[1])
        )
        contrast_delta = discrepancy[1] - discrepancy[0]
        contrast = float(self.contrast_weight) * float(
            np.dot(contrast_delta, contrast_delta)
        )
        return {
            "class_0_weighted_mmd_squared": class_zero,
            "class_1_weighted_mmd_squared": class_one,
            "contrast_weighted_mmd_squared": contrast,
            "conditional_discrepancy": class_zero + class_one + contrast,
        }


@dataclass(frozen=True)
class ConditionalRouteResult:
    decision: KMMRouteDecision
    routed_components: Mapping[str, float]
    uniform_components: Mapping[str, float]
    maximum_uniform_l1_observed: float
    support_quality_passed: bool
    component_nonworsening_passed: bool
    casewise_component_improvement_passed: bool
    pooled_direction_cosine: float
    pooled_weight_l1_distance: float
    duplicate_pooled_direction: bool
    conditional_fallback_reason: str | None


def case_equal_soft_class_kernel_means(
    kernel_features: object,
    soft_class_probabilities: object,
    case_ids: Sequence[object],
) -> tuple[np.ndarray, dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Return two case-equal soft conditional means and quality diagnostics."""

    features = np.asarray(kernel_features, dtype=np.float64)
    probabilities = np.asarray(soft_class_probabilities, dtype=np.float64)
    cases = np.asarray([str(value) for value in case_ids], dtype=object)
    if (
        features.ndim != 2
        or not features.size
        or probabilities.shape != (len(features), 2)
        or len(cases) != len(features)
        or not np.isfinite(features).all()
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        or any(not value for value in cases.tolist())
    ):
        raise ProtocolError("Conditional target kernel-mean inputs do not align.")
    class_case_means: list[list[np.ndarray]] = [[], []]
    masses: dict[str, tuple[float, float]] = {}
    effective: dict[str, tuple[float, float]] = {}
    for case_id in sorted(set(cases.tolist())):
        mask = cases == case_id
        case_masses: list[float] = []
        case_effective: list[float] = []
        for label in (0, 1):
            responsibility = probabilities[mask, label]
            mass = float(responsibility.sum())
            squared_mass = float(np.dot(responsibility, responsibility))
            if mass <= 0.0 or squared_mass <= 0.0:
                raise ProtocolError("Conditional support case has zero soft class mass.")
            class_case_means[label].append(
                np.sum(features[mask] * responsibility[:, None], axis=0) / mass
            )
            case_masses.append(mass)
            case_effective.append((mass * mass) / squared_mass)
        masses[case_id] = (case_masses[0], case_masses[1])
        effective[case_id] = (case_effective[0], case_effective[1])
    means = np.asarray(
        [np.mean(np.asarray(values), axis=0) for values in class_case_means],
        dtype=np.float64,
    )
    if means.shape != (2, features.shape[1]) or not np.isfinite(means).all():
        raise ProtocolError("Conditional target kernel means are invalid.")
    means.setflags(write=False)
    return means, masses, effective


def build_conditional_contrast_problem(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    config: ConditionalContrastConfig,
    training_seeds: Sequence[int] | None = None,
    generation_seeds: Sequence[int] | None = None,
    require_complete_target_support: bool = True,
) -> ConditionalContrastProblem:
    """Build the exact conditional objective as an augmented MMD problem."""

    pooled = build_kernel_mean_problem(
        protocol,
        source_replicas,
        target_support,
        training_seeds=training_seeds,
        generation_seeds=generation_seeds,
        require_complete_target_support=require_complete_target_support,
    )
    train = _selected_seeds(training_seeds, protocol.training_seeds, "training")
    generation = _selected_seeds(
        generation_seeds, protocol.generation_seeds, "generation"
    )
    by_key = {
        (
            replica.source_center,
            replica.training_seed,
            replica.generation_seed,
            replica.class_label,
        ): replica
        for replica in source_replicas
        if replica.training_seed in train and replica.generation_seed in generation
    }
    expected = {
        (source, training_seed, generation_seed, label)
        for source, training_seed, generation_seed, label in product(
            protocol.candidate_sources, train, generation, (0, 1)
        )
    }
    if set(by_key) != expected:
        raise ProtocolError("Conditional source replica grid is incomplete.")
    dimension = target_support.kernel_features.values.shape[1]
    source_class = np.empty(
        (len(protocol.candidate_sources), 2, dimension), dtype=np.float64
    )
    for source_index, source in enumerate(protocol.candidate_sources):
        for label in (0, 1):
            means = [
                np.asarray(
                    by_key[(source, training_seed, generation_seed, label)]
                    .kernel_features.values
                ).mean(axis=0)
                for training_seed, generation_seed in product(train, generation)
            ]
            source_class[source_index, label] = np.mean(np.asarray(means), axis=0)
    target_class, masses, effective = case_equal_soft_class_kernel_means(
        target_support.kernel_features.values,
        target_support.soft_class_probabilities,
        target_support.case_ids,
    )
    augmented_source = np.concatenate(
        (
            math.sqrt(config.class_weights[0]) * source_class[:, 0],
            math.sqrt(config.class_weights[1]) * source_class[:, 1],
            math.sqrt(config.contrast_weight)
            * (source_class[:, 1] - source_class[:, 0]),
        ),
        axis=1,
    )
    augmented_target = np.concatenate(
        (
            math.sqrt(config.class_weights[0]) * target_class[0],
            math.sqrt(config.class_weights[1]) * target_class[1],
            math.sqrt(config.contrast_weight) * (target_class[1] - target_class[0]),
        )
    )
    kernel_problem = replace(
        pooled,
        source_kernel_means=augmented_source,
        target_kernel_mean=augmented_target,
        proxy_family=CONDITIONAL_PROXY_FAMILY,
    )
    return ConditionalContrastProblem(
        kernel_problem=kernel_problem,
        source_class_kernel_means=source_class,
        target_class_kernel_means=target_class,
        class_weights=config.class_weights,
        contrast_weight=config.contrast_weight,
        soft_class_mass_by_case=masses,
        soft_class_effective_rows_by_case=effective,
    )


def build_conditional_support_case_problems(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    config: ConditionalContrastConfig,
) -> dict[str, ConditionalContrastProblem]:
    output: dict[str, ConditionalContrastProblem] = {}
    cases = np.asarray(target_support.case_ids, dtype=object)
    for case_id in protocol.support_case_ids:
        mask = cases != case_id
        if not np.any(mask) or np.all(mask):
            raise ProtocolError("Conditional support case has no rows to hold out.")
        subset = TargetSupportKernelFeatures(
            target_center=target_support.target_center,
            case_ids=tuple(np.asarray(target_support.case_ids, dtype=object)[mask]),
            kernel_features=replace(
                target_support.kernel_features,
                values=target_support.kernel_features.values[mask],
            ),
            prior_prediction=replace(
                target_support.prior_prediction,
                probabilities=target_support.soft_class_probabilities[mask],
            ),
            support_labels_used=False,
            evaluation_embeddings_used=False,
        )
        output[case_id] = build_conditional_contrast_problem(
            protocol,
            source_replicas,
            subset,
            config=config,
            require_complete_target_support=False,
        )
    return output


def build_conditional_seed_axis_problems(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    config: ConditionalContrastConfig,
    axis: str,
) -> dict[str, ConditionalContrastProblem]:
    if axis == "training_seed":
        return {
            str(seed): build_conditional_contrast_problem(
                protocol,
                source_replicas,
                target_support,
                config=config,
                training_seeds=(seed,),
            )
            for seed in protocol.training_seeds
        }
    if axis == "generation_seed":
        return {
            str(seed): build_conditional_contrast_problem(
                protocol,
                source_replicas,
                target_support,
                config=config,
                generation_seeds=(seed,),
            )
            for seed in protocol.generation_seeds
        }
    raise ProtocolError("Conditional seed stability axis is invalid.")


def build_conditional_prior_sensitivity_problems(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    conditional_config: ConditionalContrastConfig,
    prior_config: PriorControlConfig,
) -> dict[str, ConditionalContrastProblem]:
    values = (prior_config.reference_positive_prior, *prior_config.sensitivity_positive_priors)
    output: dict[str, ConditionalContrastProblem] = {}
    for value in values:
        key = f"positive_prior_{float(value):.12g}"
        if math.isclose(value, prior_config.reference_positive_prior, abs_tol=0.0):
            support = target_support
        else:
            prediction = shift_source_only_prior_prediction(
                target_support.prior_prediction,
                positive_prior=value,
                config=prior_config,
            )
            support = replace(target_support, prior_prediction=prediction)
        output[key] = build_conditional_contrast_problem(
            protocol,
            source_replicas,
            support,
            config=conditional_config,
        )
    return output


def route_conditional_contrast_mmd(
    base_problem: ConditionalContrastProblem,
    *,
    support_case_problems: Mapping[str, ConditionalContrastProblem],
    training_seed_problems: Mapping[str, ConditionalContrastProblem],
    generation_seed_problems: Mapping[str, ConditionalContrastProblem],
    prior_sensitivity_problems: Mapping[str, ConditionalContrastProblem],
    energy_direction_reference: EnergyDirectionReference,
    prior_control: PriorControlConfig,
    optimization: KMMOptimizationConfig,
    gates: KMMGateConfig,
    conditional_config: ConditionalContrastConfig,
    pooled_reference_weights: Mapping[str, float],
    duplicate_direction_cosine: float,
    duplicate_weight_l1: float,
) -> ConditionalRouteResult:
    """Route through the existing numerical gates, then fail closed on class terms."""

    expected_support_cases = set(base_problem.kernel_problem.protocol.support_case_ids)
    if (
        set(base_problem.soft_class_mass_by_case) != expected_support_cases
        or set(base_problem.soft_class_effective_rows_by_case)
        != expected_support_cases
    ):
        raise ProtocolError("Conditional base support-quality coverage is incomplete.")

    decision = route_mmd_kmm(
        base_problem.kernel_problem,
        support_case_problems={
            key: value.kernel_problem for key, value in support_case_problems.items()
        },
        training_seed_problems={
            key: value.kernel_problem for key, value in training_seed_problems.items()
        },
        generation_seed_problems={
            key: value.kernel_problem for key, value in generation_seed_problems.items()
        },
        prior_sensitivity_problems={
            key: value.kernel_problem
            for key, value in prior_sensitivity_problems.items()
        },
        energy_direction_reference=energy_direction_reference,
        prior_control=prior_control,
        optimization=optimization,
        gates=gates,
    )
    uniform = dict(decision.base_solution.uniform_weights)
    routed = dict(decision.base_solution.weights)
    uniform_components = base_problem.component_losses(uniform)
    routed_components = base_problem.component_losses(routed)
    l1 = float(sum(abs(routed[key] - uniform[key]) for key in uniform))
    support_quality = all(
        mass >= conditional_config.minimum_soft_class_mass_per_case
        for values in base_problem.soft_class_mass_by_case.values()
        for mass in values
    ) and all(
        value >= conditional_config.minimum_soft_class_effective_rows_per_case
        for values in base_problem.soft_class_effective_rows_by_case.values()
        for value in values
    )
    component_keys = (
        "class_0_weighted_mmd_squared",
        "class_1_weighted_mmd_squared",
        "contrast_weighted_mmd_squared",
    )
    component_nonworsening = all(
        routed_components[key]
        <= uniform_components[key] + conditional_config.component_tolerance
        for key in component_keys
    )
    casewise_component_improvement = True
    for problem in support_case_problems.values():
        case_uniform = problem.component_losses(uniform)
        case_routed = problem.component_losses(routed)
        if not (
            case_routed["conditional_discrepancy"]
            < case_uniform["conditional_discrepancy"]
            - conditional_config.component_tolerance
            and all(
                case_routed[key]
                <= case_uniform[key] + conditional_config.component_tolerance
                for key in component_keys
            )
        ):
            casewise_component_improvement = False
            break
    pooled = _weight_vector(
        pooled_reference_weights, decision.base_solution.candidate_sources
    )
    routed_vector = _weight_vector(routed, decision.base_solution.candidate_sources)
    uniform_vector = _weight_vector(uniform, decision.base_solution.candidate_sources)
    pooled_cosine = _direction_cosine(
        routed_vector - uniform_vector, pooled - uniform_vector
    )
    pooled_l1 = float(np.abs(routed_vector - pooled).sum())
    duplicate_pooled = (
        not decision.base_solution.used_uniform_fallback
        and (
            pooled_cosine >= float(duplicate_direction_cosine)
            or pooled_l1 <= float(duplicate_weight_l1)
        )
    )
    reason = decision.fallback_reason if decision.used_uniform_fallback else None
    if reason is None and not support_quality:
        reason = "insufficient_soft_class_support_uniform"
    if reason is None and l1 > conditional_config.maximum_uniform_l1:
        reason = "uniform_l1_trust_region_exceeded"
    if reason is None and not component_nonworsening:
        reason = "conditional_component_worsening_uniform"
    if reason is None and not casewise_component_improvement:
        reason = "support_case_component_improvement_failed_uniform"
    if reason is None and duplicate_pooled:
        reason = "duplicate_pooled_mmd_direction_uniform"
    if reason is not None and not decision.used_uniform_fallback:
        decision = replace(
            decision,
            final_weights=uniform,
            used_uniform_fallback=True,
            fallback_reason=reason,
        )
    return ConditionalRouteResult(
        decision=decision,
        routed_components=MappingProxyType(dict(routed_components)),
        uniform_components=MappingProxyType(dict(uniform_components)),
        maximum_uniform_l1_observed=l1,
        support_quality_passed=support_quality,
        component_nonworsening_passed=component_nonworsening,
        casewise_component_improvement_passed=casewise_component_improvement,
        pooled_direction_cosine=pooled_cosine,
        pooled_weight_l1_distance=pooled_l1,
        duplicate_pooled_direction=duplicate_pooled,
        conditional_fallback_reason=reason,
    )


def _selected_seeds(
    requested: Sequence[int] | None,
    allowed: tuple[int, ...],
    role: str,
) -> tuple[int, ...]:
    if requested is not None and any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in requested
    ):
        raise ProtocolError(f"Conditional {role} seed subset is invalid.")
    values = allowed if requested is None else tuple(sorted(int(value) for value in requested))
    if not values or len(set(values)) != len(values) or not set(values).issubset(allowed):
        raise ProtocolError(f"Conditional {role} seed subset is invalid.")
    return values


def _quality_mapping(
    values: Mapping[str, tuple[float, float]], role: str
) -> dict[str, tuple[float, float]]:
    output: dict[str, tuple[float, float]] = {}
    for key, pair in values.items():
        parsed = tuple(float(value) for value in pair)
        if len(parsed) != 2 or any(not math.isfinite(value) or value <= 0.0 for value in parsed):
            raise ProtocolError(f"Conditional {role} is invalid.")
        output[str(key)] = (parsed[0], parsed[1])
    return output


def _weight_vector(
    values: Mapping[str, float], sources: tuple[str, ...]
) -> np.ndarray:
    if set(values) != set(sources):
        raise ProtocolError("Conditional direction weights use the wrong sources.")
    vector = np.asarray([float(values[source]) for source in sources])
    if (
        not np.isfinite(vector).all()
        or np.any(vector < 0.0)
        or not np.isclose(vector.sum(), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise ProtocolError("Conditional direction weights violate the simplex.")
    return vector


def _direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-15 or right_norm <= 1e-15:
        return 1.0 if left_norm <= 1e-15 and right_norm <= 1e-15 else -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))


__all__ = (
    "CONDITIONAL_PROXY_FAMILY",
    "ConditionalContrastConfig",
    "ConditionalContrastProblem",
    "ConditionalRouteResult",
    "build_conditional_contrast_problem",
    "build_conditional_prior_sensitivity_problems",
    "build_conditional_seed_axis_problems",
    "build_conditional_support_case_problems",
    "case_equal_soft_class_kernel_means",
    "route_conditional_contrast_mmd",
)
