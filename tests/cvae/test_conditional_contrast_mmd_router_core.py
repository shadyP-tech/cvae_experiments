from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.mmd_kmm_mixture import (
    ConditionalContrastConfig,
    ConditionalContrastProblem,
    DirectionIdentityAudit,
    KMMGateConfig,
    KMMOptimizationConfig,
    KMMRouteDecision,
    KMMWeightSolution,
    KernelMeanProblem,
    MMDKMMProtocol,
    PriorControlConfig,
    SourceKernelReplica,
    StabilityAudit,
    TargetSupportKernelFeatures,
    TransformedKernelFeatures,
    build_conditional_contrast_problem,
    build_conditional_support_case_problems,
    case_equal_soft_class_kernel_means,
    prepare_source_only_responsibilities,
    route_conditional_contrast_mmd,
)


TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)


def _protocol() -> MMDKMMProtocol:
    return MMDKMMProtocol(
        target_center="target",
        candidate_sources=tuple(f"source-{index}" for index in range(7)),
        support_case_ids=("case-a", "case-b"),
        evaluation_case_ids=("eval-a", "eval-b"),
        common_frame_hash="common-frame",
    )


def _conditional_config() -> ConditionalContrastConfig:
    return ConditionalContrastConfig(
        class_weights=(0.5, 0.5),
        contrast_weight=1.0,
        maximum_uniform_l1=0.25,
        minimum_soft_class_mass_per_case=1.0,
        minimum_soft_class_effective_rows_per_case=2.0,
    )


def _prior_config() -> PriorControlConfig:
    return PriorControlConfig(
        probability_clip=1.0e-3,
        temperature=1.0,
        sensitivity_positive_priors=(0.35, 0.65),
    )


def _kernel_block(values: object) -> TransformedKernelFeatures:
    return TransformedKernelFeatures(
        values=np.asarray(values, dtype=np.float64),
        common_frame_hash="common-frame",
        preprocessing_hash="source-pool-scaler",
        candidate_pool_fit_hash="equal-count-source-pool",
        kernel_map_hash="shared-kernel-map",
    )


def _replicas(protocol: MMDKMMProtocol) -> tuple[SourceKernelReplica, ...]:
    output: list[SourceKernelReplica] = []
    for source_index, source in enumerate(protocol.candidate_sources):
        class_rows = {
            0: np.asarray(((float(source_index), 0.0),) * 2),
            1: np.asarray(((float(source_index), 1.0 + 0.2 * source_index),) * 2),
        }
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for label in (0, 1):
                    output.append(
                        SourceKernelReplica(
                            source_center=source,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            class_label=label,
                            kernel_features=_kernel_block(class_rows[label]),
                        )
                    )
    return tuple(output)


def _target_support(protocol: MMDKMMProtocol) -> TargetSupportKernelFeatures:
    probabilities = np.asarray(
        (
            (0.85, 0.15),
            (0.15, 0.85),
            (0.80, 0.20),
            (0.20, 0.80),
        ),
        dtype=np.float64,
    )
    prediction = prepare_source_only_responsibilities(
        probabilities,
        protocol=protocol,
        prior_model_hash="source-only-prior",
        prior_fit_pool_hash="target-excluded-balanced-pool",
        config=_prior_config(),
    )
    return TargetSupportKernelFeatures(
        target_center=protocol.target_center,
        case_ids=("case-a", "case-a", "case-b", "case-b"),
        kernel_features=_kernel_block(
            ((1.0, 0.0), (1.0, 1.2), (2.0, 0.0), (2.0, 1.4))
        ),
        prior_prediction=prediction,
    )


def test_soft_conditional_means_are_case_equal_and_case_duplication_invariant(
) -> None:
    features = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 4.0)))
    probabilities = np.asarray(
        ((0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8))
    )
    cases = ("a", "a", "b", "b")
    observed, mass, effective = case_equal_soft_class_kernel_means(
        features, probabilities, cases
    )

    duplicate_entire_case_a = np.asarray((0, 1, 0, 1, 2, 3))
    duplicated, duplicated_mass, duplicated_effective = (
        case_equal_soft_class_kernel_means(
            features[duplicate_entire_case_a],
            probabilities[duplicate_entire_case_a],
            tuple(cases[index] for index in duplicate_entire_case_a),
        )
    )
    np.testing.assert_allclose(duplicated, observed, atol=1.0e-12)
    assert duplicated_mass["a"] == pytest.approx(
        tuple(2.0 * value for value in mass["a"])
    )
    assert duplicated_effective["a"] == pytest.approx(
        tuple(2.0 * value for value in effective["a"])
    )
    assert observed.shape == (2, 2)
    assert observed.flags.writeable is False


def test_lifted_kernel_problem_equals_explicit_conditional_contrast_loss() -> None:
    protocol = _protocol()
    replicas = _replicas(protocol)
    problem = build_conditional_contrast_problem(
        protocol,
        replicas,
        _target_support(protocol),
        config=_conditional_config(),
    )
    weights = {
        source: value
        for source, value in zip(
            protocol.candidate_sources,
            (0.18, 0.16, 0.132, 0.132, 0.132, 0.132, 0.132),
            strict=True,
        )
    }
    explicit = problem.component_losses(weights)
    vector = np.asarray([weights[source] for source in protocol.candidate_sources])
    lifted_delta = (
        vector @ problem.kernel_problem.source_kernel_means
        - problem.kernel_problem.target_kernel_mean
    )
    np.testing.assert_allclose(
        np.dot(lifted_delta, lifted_delta),
        explicit["conditional_discrepancy"],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert problem.kernel_problem.proxy_family == "class_conditional_contrast_mmd_kmm"
    assert not np.allclose(
        problem.source_class_kernel_means[:, 0],
        problem.source_class_kernel_means[:, 1],
    )
    support_variants = build_conditional_support_case_problems(
        protocol,
        replicas,
        _target_support(protocol),
        config=_conditional_config(),
    )
    assert tuple(support_variants) == protocol.support_case_ids
    assert set(support_variants["case-a"].soft_class_mass_by_case) == {"case-b"}
    assert set(support_variants["case-b"].soft_class_mass_by_case) == {"case-a"}

    with pytest.raises(ProtocolError, match="complete candidate"):
        build_conditional_contrast_problem(
            protocol,
            replicas[:-1],
            _target_support(protocol),
            config=_conditional_config(),
        )
    with pytest.raises(ProtocolError, match="training seed subset"):
        build_conditional_contrast_problem(
            protocol,
            replicas,
            _target_support(protocol),
            config=_conditional_config(),
            training_seeds=(17.5,),
        )


def _manual_problem(
    routed_weights: Mapping[str, float],
    *,
    weak_support: bool = False,
) -> ConditionalContrastProblem:
    protocol = _protocol()
    sources = protocol.candidate_sources
    source_axis = np.arange(len(sources), dtype=np.float64)
    source_class = np.stack(
        (
            source_axis[:, None],
            (1.0 + 2.0 * source_axis)[:, None],
        ),
        axis=1,
    )
    vector = np.asarray([routed_weights[source] for source in sources])
    target_class = np.einsum("s,scd->cd", vector, source_class)
    config = _conditional_config()
    augmented_source = np.concatenate(
        (
            np.sqrt(config.class_weights[0]) * source_class[:, 0],
            np.sqrt(config.class_weights[1]) * source_class[:, 1],
            np.sqrt(config.contrast_weight)
            * (source_class[:, 1] - source_class[:, 0]),
        ),
        axis=1,
    )
    augmented_target = np.concatenate(
        (
            np.sqrt(config.class_weights[0]) * target_class[0],
            np.sqrt(config.class_weights[1]) * target_class[1],
            np.sqrt(config.contrast_weight) * (target_class[1] - target_class[0]),
        )
    )
    kernel_problem = KernelMeanProblem(
        protocol=protocol,
        candidate_sources=sources,
        source_kernel_means=augmented_source,
        target_kernel_mean=augmented_target,
        common_frame_hash=protocol.common_frame_hash,
        kernel_map_hash="shared-kernel-map",
        preprocessing_hash="source-pool-scaler",
        candidate_pool_fit_hash="equal-count-source-pool",
        kernel_transform_role="shared_frozen_source_pool_nystroem",
        prior_family_hash="prior-family",
        prior_control_hash="prior-control",
        prior_state_hash="prior-state",
        prior_sensitivity_positive_prior=None,
        target_kernel_feature_sha256="a" * 64,
        target_responsibility_sha256="b" * 64,
        source_replica_count=126,
        target_support_row_count=4,
        proxy_family="class_conditional_contrast_mmd_kmm",
    )
    mass = (0.5, 1.5) if weak_support else (2.0, 2.0)
    return ConditionalContrastProblem(
        kernel_problem=kernel_problem,
        source_class_kernel_means=source_class,
        target_class_kernel_means=target_class,
        class_weights=config.class_weights,
        contrast_weight=config.contrast_weight,
        soft_class_mass_by_case={"case-a": mass, "case-b": (2.0, 2.0)},
        soft_class_effective_rows_by_case={
            "case-a": (2.0, 2.0),
            "case-b": (2.0, 2.0),
        },
    )


def _decision(weights: Mapping[str, float]) -> KMMRouteDecision:
    sources = _protocol().candidate_sources
    uniform = {source: 1.0 / len(sources) for source in sources}
    vector = np.asarray([weights[source] for source in sources])
    uniform_vector = np.asarray([uniform[source] for source in sources])
    solution = KMMWeightSolution(
        candidate_sources=sources,
        uniform_weights=uniform,
        weights=weights,
        delta={
            source: float(vector[index] - uniform_vector[index])
            for index, source in enumerate(sources)
        },
        proxy_objective=0.9,
        uniform_proxy_objective=1.0,
        proxy_improvement=0.1,
        mmd_squared=0.8,
        uniform_mmd_squared=1.0,
        regularization_value=0.1,
        effective_source_count=float(1.0 / np.dot(vector, vector)),
        maximum_source_weight=float(vector.max()),
        used_uniform_fallback=False,
        fallback_reason=None,
        solver_success=True,
        solver_message="synthetic fixture",
        solver_iterations=1,
        solver_method="scipy_slsqp_continuous_convex_proxy",
        solver_version="fixture",
        optimality_residual=0.0,
    )
    return KMMRouteDecision(
        candidate_sources=sources,
        base_solution=solution,
        final_weights=weights,
        used_uniform_fallback=False,
        fallback_reason=None,
        stability_audits=(
            StabilityAudit(
                axis="fixture",
                variant_ids=("fixture",),
                maximum_l1_distance=0.0,
                minimum_direction_cosine=1.0,
                passed=True,
                failure_reason=None,
            ),
        ),
        direction_identity=DirectionIdentityAudit(
            reference_role="fixture",
            direction_cosine=0.0,
            weight_l1_distance=1.0,
            duplicate=False,
        ),
    )


@pytest.mark.parametrize(
    ("weights", "weak_support", "pooled_same", "expected_reason"),
    (
        (
            (0.18, 0.16, 0.132, 0.132, 0.132, 0.132, 0.132),
            True,
            False,
            "insufficient_soft_class_support_uniform",
        ),
        (
            (0.24, 0.18, 0.116, 0.116, 0.116, 0.116, 0.116),
            False,
            False,
            "uniform_l1_trust_region_exceeded",
        ),
        (
            (0.18, 0.16, 0.132, 0.132, 0.132, 0.132, 0.132),
            False,
            True,
            "duplicate_pooled_mmd_direction_uniform",
        ),
    ),
)
def test_conditional_gates_fail_closed_to_exact_equal_union(
    monkeypatch: pytest.MonkeyPatch,
    weights: tuple[float, ...],
    weak_support: bool,
    pooled_same: bool,
    expected_reason: str,
) -> None:
    protocol = _protocol()
    routed = dict(zip(protocol.candidate_sources, weights, strict=True))
    decision = _decision(routed)
    monkeypatch.setattr(
        "midogpp_thesis.cvae.routing.mmd_kmm_mixture.conditional.route_mmd_kmm",
        lambda *args, **kwargs: decision,
    )
    problem = _manual_problem(routed, weak_support=weak_support)
    pooled = (
        routed
        if pooled_same
        else {source: 1.0 / len(protocol.candidate_sources) for source in protocol.candidate_sources}
    )
    result = route_conditional_contrast_mmd(
        problem,
        support_case_problems={"case-a": problem, "case-b": problem},
        training_seed_problems={},
        generation_seed_problems={},
        prior_sensitivity_problems={},
        energy_direction_reference=object(),
        prior_control=_prior_config(),
        optimization=KMMOptimizationConfig(
            regularization=0.1,
            minimum_proxy_improvement=1.0e-6,
        ),
        gates=KMMGateConfig(
            maximum_support_l1=0.2,
            maximum_training_seed_l1=0.2,
            maximum_generation_seed_l1=0.2,
            maximum_prior_sensitivity_l1=0.15,
            minimum_direction_cosine=0.5,
            duplicate_direction_cosine=0.995,
            duplicate_weight_l1=0.02,
        ),
        conditional_config=_conditional_config(),
        pooled_reference_weights=pooled,
        duplicate_direction_cosine=0.995,
        duplicate_weight_l1=0.02,
    )
    expected_uniform = {
        source: 1.0 / len(protocol.candidate_sources)
        for source in protocol.candidate_sources
    }
    assert result.decision.used_uniform_fallback is True
    assert result.decision.fallback_reason == expected_reason
    assert dict(result.decision.final_weights) == expected_uniform
    assert result.conditional_fallback_reason == expected_reason
