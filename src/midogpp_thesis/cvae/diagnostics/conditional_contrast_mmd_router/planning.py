"""Scientific planning step for the conditional contrast-MMD diagnostic."""

from __future__ import annotations

from typing import Sequence

from ...routing.mmd_kmm_mixture import (
    ConditionalContrastConfig,
    EnergyDirectionReference,
    KMMGateConfig,
    KMMOptimizationConfig,
    KMMRouteDecision,
    KernelMeanProblem,
    MMDKMMProtocol,
    PriorControlConfig,
    SourceKernelReplica,
    TargetSupportKernelFeatures,
    build_conditional_contrast_problem,
    build_conditional_prior_sensitivity_problems,
    build_conditional_seed_axis_problems,
    build_conditional_support_case_problems,
    route_conditional_contrast_mmd,
    solve_kmm_weights,
)


def build_conditional_route_and_diagnostics(
    protocol: MMDKMMProtocol,
    source_replicas: Sequence[SourceKernelReplica],
    target_support: TargetSupportKernelFeatures,
    *,
    pooled_problem: KernelMeanProblem,
    energy_direction_reference: EnergyDirectionReference,
    prior_control: PriorControlConfig,
    optimization: KMMOptimizationConfig,
    gates: KMMGateConfig,
    conditional_config: ConditionalContrastConfig,
    pooled_reference_optimization: KMMOptimizationConfig,
    duplicate_direction_cosine: float,
    duplicate_weight_l1: float,
) -> tuple[KMMRouteDecision, dict[str, object]]:
    """Build one frozen label-free route and its validator-facing audit payload."""

    conditional_base = build_conditional_contrast_problem(
        protocol,
        source_replicas,
        target_support,
        config=conditional_config,
    )
    conditional_route = route_conditional_contrast_mmd(
        conditional_base,
        support_case_problems=build_conditional_support_case_problems(
            protocol,
            source_replicas,
            target_support,
            config=conditional_config,
        ),
        training_seed_problems=build_conditional_seed_axis_problems(
            protocol,
            source_replicas,
            target_support,
            config=conditional_config,
            axis="training_seed",
        ),
        generation_seed_problems=build_conditional_seed_axis_problems(
            protocol,
            source_replicas,
            target_support,
            config=conditional_config,
            axis="generation_seed",
        ),
        prior_sensitivity_problems=build_conditional_prior_sensitivity_problems(
            protocol,
            source_replicas,
            target_support,
            conditional_config=conditional_config,
            prior_config=prior_control,
        ),
        energy_direction_reference=energy_direction_reference,
        prior_control=prior_control,
        optimization=optimization,
        gates=gates,
        conditional_config=conditional_config,
        pooled_reference_weights=solve_kmm_weights(
            pooled_problem,
            pooled_reference_optimization,
        ).weights,
        duplicate_direction_cosine=duplicate_direction_cosine,
        duplicate_weight_l1=duplicate_weight_l1,
    )
    diagnostics: dict[str, object] = {
        "proxy_family": "class_conditional_contrast_mmd_kmm",
        "class_weights": list(conditional_config.class_weights),
        "contrast_weight": conditional_config.contrast_weight,
        "maximum_uniform_l1": conditional_config.maximum_uniform_l1,
        "observed_uniform_l1": conditional_route.maximum_uniform_l1_observed,
        "routed_components": dict(conditional_route.routed_components),
        "uniform_components": dict(conditional_route.uniform_components),
        "support_quality_passed": conditional_route.support_quality_passed,
        "component_nonworsening_passed": (
            conditional_route.component_nonworsening_passed
        ),
        "casewise_component_improvement_passed": (
            conditional_route.casewise_component_improvement_passed
        ),
        "soft_class_mass_by_case": {
            key: list(value)
            for key, value in conditional_base.soft_class_mass_by_case.items()
        },
        "soft_class_effective_rows_by_case": {
            key: list(value)
            for key, value in (
                conditional_base.soft_class_effective_rows_by_case.items()
            )
        },
        "pooled_direction_cosine": conditional_route.pooled_direction_cosine,
        "pooled_weight_l1_distance": conditional_route.pooled_weight_l1_distance,
        "duplicate_pooled_direction": conditional_route.duplicate_pooled_direction,
        "previous_pooled_mmd_output_used": False,
        "conditional_fallback_reason": conditional_route.conditional_fallback_reason,
    }
    return conditional_route.decision, diagnostics


__all__ = ("build_conditional_route_and_diagnostics",)
