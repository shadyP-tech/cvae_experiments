from __future__ import annotations

from dataclasses import replace
import importlib
import inspect
from typing import Mapping

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.mmd_kmm_mixture import (
    EnergyDirectionReference,
    FrozenNystroemFeatureMap,
    KMMGateConfig,
    KMMOptimizationConfig,
    KernelMeanProblem,
    MMDKMMProtocol,
    PriorControlConfig,
    SourceKernelReplica,
    TargetSupportKernelFeatures,
    TransformedKernelFeatures,
    build_kernel_mean_problem,
    build_prior_sensitivity_problems,
    build_seed_axis_problems,
    build_support_case_problems,
    case_equal_class_balanced_kernel_mean,
    prepare_source_only_responsibilities,
    route_mmd_kmm,
    solve_kmm_weights,
    transform_frozen_nystroem,
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


def _prior_config() -> PriorControlConfig:
    return PriorControlConfig(
        probability_clip=1e-6,
        temperature=1.0,
        sensitivity_positive_priors=(0.35, 0.65),
    )


def _kernel_block(
    values: object,
    protocol: MMDKMMProtocol,
    *,
    kernel_map_hash: str = "kernel-map",
) -> TransformedKernelFeatures:
    return TransformedKernelFeatures(
        values=np.asarray(values, dtype=np.float64),
        common_frame_hash=protocol.common_frame_hash,
        preprocessing_hash="source-pool-scaler",
        candidate_pool_fit_hash="equal-count-source-pool",
        kernel_map_hash=kernel_map_hash,
    )


def _source_replicas(
    protocol: MMDKMMProtocol,
) -> tuple[SourceKernelReplica, ...]:
    output: list[SourceKernelReplica] = []
    denominator = float(len(protocol.candidate_sources) - 1)
    for source_index, source in enumerate(protocol.candidate_sources):
        location = float(source_index) / denominator
        features = _kernel_block(
            ((location, location**2), (location, location**2)), protocol
        )
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for class_label in (0, 1):
                    output.append(
                        SourceKernelReplica(
                            source_center=source,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            class_label=class_label,
                            kernel_features=features,
                        )
                    )
    return tuple(output)


def _target_support(
    protocol: MMDKMMProtocol,
) -> TargetSupportKernelFeatures:
    probabilities = np.asarray(
        ((0.8, 0.2), (0.2, 0.8), (0.7, 0.3), (0.3, 0.7))
    )
    prediction = prepare_source_only_responsibilities(
        probabilities,
        protocol=protocol,
        prior_model_hash="pooled-prior",
        prior_fit_pool_hash="target-excluded-equal-pool",
        config=_prior_config(),
    )
    return TargetSupportKernelFeatures(
        target_center=protocol.target_center,
        case_ids=("case-a", "case-a", "case-b", "case-b"),
        kernel_features=_kernel_block(np.zeros((4, 2)), protocol),
        prior_prediction=prediction,
    )


def _optimization() -> KMMOptimizationConfig:
    return KMMOptimizationConfig(
        regularization=0.05,
        minimum_proxy_improvement=1e-12,
    )


def _energy_reference(
    problem: KernelMeanProblem,
    weights: Mapping[str, float],
) -> EnergyDirectionReference:
    return EnergyDirectionReference(
        target_center=problem.protocol.target_center,
        candidate_sources=problem.candidate_sources,
        support_partition_hash=problem.protocol.support_partition_hash,
        common_frame_hash=problem.common_frame_hash,
        preprocessing_hash=problem.preprocessing_hash,
        candidate_pool_fit_hash=problem.candidate_pool_fit_hash,
        kernel_map_hash=problem.kernel_map_hash,
        training_seeds=problem.protocol.training_seeds,
        generation_seeds=problem.protocol.generation_seeds,
        weights=weights,
        energy_calibration_hash="frozen-three-seed-energy-calibration",
        action_id="rho_0.50",
    )


def _problems() -> tuple[
    KernelMeanProblem,
    dict[str, KernelMeanProblem],
    dict[str, KernelMeanProblem],
    dict[str, KernelMeanProblem],
    dict[str, KernelMeanProblem],
]:
    protocol = _protocol()
    replicas = _source_replicas(protocol)
    support = _target_support(protocol)
    return (
        build_kernel_mean_problem(protocol, replicas, support),
        build_support_case_problems(protocol, replicas, support),
        build_seed_axis_problems(
            protocol, replicas, support, axis="training_seed"
        ),
        build_seed_axis_problems(
            protocol, replicas, support, axis="generation_seed"
        ),
        build_prior_sensitivity_problems(
            protocol,
            replicas,
            support,
            config=_prior_config(),
        ),
    )


def test_target_kernel_mean_is_case_equal_class_balanced_and_case_duplication_invariant(
) -> None:
    features = np.asarray(((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, 4.0)))
    probabilities = np.asarray(((0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8)))
    cases = ("a", "a", "b", "b")
    expected = np.asarray((0.5, 1.5))
    observed = case_equal_class_balanced_kernel_mean(features, probabilities, cases)
    np.testing.assert_allclose(observed, expected, atol=1e-12)

    duplicate_whole_case_a = np.asarray((0, 1, 0, 1, 2, 3))
    duplicated = case_equal_class_balanced_kernel_mean(
        features[duplicate_whole_case_a],
        probabilities[duplicate_whole_case_a],
        tuple(cases[index] for index in duplicate_whole_case_a),
    )
    np.testing.assert_allclose(duplicated, observed, atol=1e-12)
    assert observed.flags.writeable is False


def test_problem_builder_is_order_invariant_and_requires_complete_replica_grid(
) -> None:
    protocol = _protocol()
    replicas = _source_replicas(protocol)
    support = _target_support(protocol)
    first = build_kernel_mean_problem(protocol, replicas, support)
    reversed_input = build_kernel_mean_problem(
        protocol, tuple(reversed(replicas)), support
    )
    np.testing.assert_allclose(
        first.source_kernel_means, reversed_input.source_kernel_means
    )
    np.testing.assert_allclose(
        first.target_kernel_mean, reversed_input.target_kernel_mean
    )
    assert first.source_replica_count == 7 * 3 * 3 * 2

    with pytest.raises(ProtocolError, match="complete candidate"):
        build_kernel_mean_problem(protocol, replicas[:-1], support)
    mixed_map = replace(
        replicas[0],
        kernel_features=replace(
            replicas[0].kernel_features,
            kernel_map_hash="other-map",
        ),
    )
    with pytest.raises(ProtocolError, match="protocol boundary"):
        build_kernel_mean_problem(protocol, (mixed_map, *replicas[1:]), support)
    mixed_preprocessing = replace(
        replicas[0],
        kernel_features=replace(
            replicas[0].kernel_features,
            preprocessing_hash="other-source-pool-scaler",
        ),
    )
    with pytest.raises(ProtocolError, match="protocol boundary"):
        build_kernel_mean_problem(
            protocol, (mixed_preprocessing, *replicas[1:]), support
        )
    short_replica = replace(
        replicas[0],
        kernel_features=replace(
            replicas[0].kernel_features,
            values=np.zeros((1, 2)),
        ),
    )
    with pytest.raises(ProtocolError, match="equal per-class sample budget"):
        build_kernel_mean_problem(protocol, (short_replica, *replicas[1:]), support)
    wrong_prior_pool = replace(
        support,
        prior_prediction=replace(
            support.prior_prediction,
            candidate_sources=tuple(f"other-{index}" for index in range(7)),
        ),
    )
    with pytest.raises(ProtocolError, match="prior crossed"):
        build_kernel_mean_problem(protocol, replicas, wrong_prior_pool)


def test_seed_axis_problems_retain_every_seed_without_selecting_one() -> None:
    protocol = _protocol()
    replicas = _source_replicas(protocol)
    support = _target_support(protocol)
    training = build_seed_axis_problems(
        protocol, replicas, support, axis="training_seed"
    )
    generation = build_seed_axis_problems(
        protocol, replicas, support, axis="generation_seed"
    )
    assert tuple(sorted(training)) == ("101", "17", "42")
    assert tuple(sorted(generation)) == ("101", "17", "42")
    assert {problem.source_replica_count for problem in training.values()} == {
        7 * 1 * 3 * 2
    }
    assert {problem.source_replica_count for problem in generation.values()} == {
        7 * 3 * 1 * 2
    }


def test_kmm_solver_returns_dense_feasible_nonuniform_weights() -> None:
    problem, *_ = _problems()
    solution = solve_kmm_weights(problem, _optimization())
    weights = np.asarray(tuple(solution.weights.values()))
    assert solution.used_uniform_fallback is False
    assert solution.proxy_improvement > 0.0
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert weights.min() >= 0.0
    assert weights.max() <= 0.25 + 1e-8
    assert solution.effective_source_count >= 6.0 - 1e-7
    assert solution.optimality_residual is not None
    assert solution.optimality_residual <= _optimization().optimality_tolerance
    assert solution.solver_method == "scipy_slsqp_continuous_convex_proxy"
    assert solution.solver_version
    assert solution.downstream_utility_claimed is False


def test_kmm_policy_constraints_are_immutable() -> None:
    with pytest.raises(ProtocolError, match="optimizer configuration"):
        KMMOptimizationConfig(
            regularization=0.05,
            minimum_proxy_improvement=0.0,
            max_source_weight=0.3,
        )
    with pytest.raises(ProtocolError, match="optimizer configuration"):
        KMMOptimizationConfig(
            regularization=0.05,
            minimum_proxy_improvement=0.0,
            minimum_effective_sources=5.0,
        )


def test_kmm_solver_returns_exact_equal_union_when_proxy_cannot_improve() -> None:
    problem, *_ = _problems()
    uniform_target = np.mean(problem.source_kernel_means, axis=0)
    no_gain = replace(problem, target_kernel_mean=uniform_target)
    solution = solve_kmm_weights(no_gain, _optimization())
    expected = {
        source: 1.0 / len(no_gain.candidate_sources)
        for source in no_gain.candidate_sources
    }
    assert solution.weights == expected
    assert solution.delta == {source: 0.0 for source in no_gain.candidate_sources}
    assert solution.used_uniform_fallback is True
    assert solution.fallback_reason == "insufficient_proxy_improvement_uniform"


def test_route_requires_stability_and_rejects_duplicate_energy_direction() -> None:
    base, support, training, generation, priors = _problems()
    sources = base.candidate_sources
    uniform = 1.0 / len(sources)
    opposite_reference = {source: uniform for source in sources}
    opposite_reference[sources[0]] -= 0.05
    opposite_reference[sources[-1]] += 0.05
    gates = KMMGateConfig(
        maximum_support_l1=1e-8,
        maximum_training_seed_l1=1e-8,
        maximum_generation_seed_l1=1e-8,
        maximum_prior_sensitivity_l1=1e-8,
        minimum_direction_cosine=0.999999,
        duplicate_direction_cosine=0.999999,
        duplicate_weight_l1=1e-8,
    )
    accepted = route_mmd_kmm(
        base,
        support_case_problems=support,
        training_seed_problems=training,
        generation_seed_problems=generation,
        prior_sensitivity_problems=priors,
        energy_direction_reference=_energy_reference(base, opposite_reference),
        prior_control=_prior_config(),
        optimization=_optimization(),
        gates=gates,
    )
    assert accepted.used_uniform_fallback is False
    assert accepted.direction_identity.duplicate is False
    assert all(audit.passed for audit in accepted.stability_audits)
    assert accepted.promotion_eligible is False
    assert accepted.target_labels_used is False
    assert accepted.stage90_inputs_used is False

    valid_reference = _energy_reference(base, opposite_reference)
    with pytest.raises(ProtocolError, match="reference contract"):
        replace(valid_reference, target_labels_used=True)
    with pytest.raises(ProtocolError, match="frozen routing context"):
        route_mmd_kmm(
            base,
            support_case_problems=support,
            training_seed_problems=training,
            generation_seed_problems=generation,
            prior_sensitivity_problems=priors,
            energy_direction_reference=replace(
                valid_reference,
                support_partition_hash="different-support-partition",
            ),
            prior_control=_prior_config(),
            optimization=_optimization(),
            gates=gates,
        )

    duplicate = route_mmd_kmm(
        base,
        support_case_problems=support,
        training_seed_problems=training,
        generation_seed_problems=generation,
        prior_sensitivity_problems=priors,
        energy_direction_reference=_energy_reference(
            base, accepted.base_solution.weights
        ),
        prior_control=_prior_config(),
        optimization=_optimization(),
        gates=gates,
    )
    expected = {source: uniform for source in sources}
    assert duplicate.final_weights == expected
    assert duplicate.used_uniform_fallback is True
    assert duplicate.fallback_reason == "duplicate_energy_direction_uniform"


def test_route_fails_closed_on_incomplete_stability_grid() -> None:
    base, support, training, generation, priors = _problems()
    sources = base.candidate_sources
    uniform = 1.0 / len(sources)
    reference = {source: uniform for source in sources}
    reference[sources[0]] -= 0.05
    reference[sources[-1]] += 0.05
    decision = route_mmd_kmm(
        base,
        support_case_problems={},
        training_seed_problems=training,
        generation_seed_problems=generation,
        prior_sensitivity_problems=priors,
        energy_direction_reference=_energy_reference(base, reference),
        prior_control=_prior_config(),
        optimization=_optimization(),
        gates=KMMGateConfig(
            maximum_support_l1=1.0,
            maximum_training_seed_l1=1.0,
            maximum_generation_seed_l1=1.0,
            maximum_prior_sensitivity_l1=1.0,
            minimum_direction_cosine=-1.0,
            duplicate_direction_cosine=1.0,
            duplicate_weight_l1=0.0,
        ),
    )
    assert decision.used_uniform_fallback is True
    assert decision.fallback_reason == (
        "support_case_incomplete_stability_grid_uniform"
    )
    assert decision.final_weights == {source: uniform for source in sources}


def test_prior_grid_is_exact_predeclared_and_bound_to_route() -> None:
    base, support, training, generation, priors = _problems()
    assert tuple(sorted(priors)) == (
        "positive_prior_0.35",
        "positive_prior_0.5",
        "positive_prior_0.65",
    )
    wrong_prior = PriorControlConfig(
        probability_clip=1e-5,
        temperature=1.0,
        sensitivity_positive_priors=(0.35, 0.65),
    )
    sources = base.candidate_sources
    reference = {source: 1.0 / len(sources) for source in sources}
    with pytest.raises(ProtocolError, match="different prior-control state"):
        route_mmd_kmm(
            base,
            support_case_problems=support,
            training_seed_problems=training,
            generation_seed_problems=generation,
            prior_sensitivity_problems=priors,
            energy_direction_reference=_energy_reference(base, reference),
            prior_control=wrong_prior,
            optimization=_optimization(),
            gates=KMMGateConfig(
                maximum_support_l1=1.0,
                maximum_training_seed_l1=1.0,
                maximum_generation_seed_l1=1.0,
                maximum_prior_sensitivity_l1=1.0,
                minimum_direction_cosine=-1.0,
                duplicate_direction_cosine=1.0,
                duplicate_weight_l1=0.0,
            ),
        )


def test_solver_exception_returns_exact_equal_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, *_ = _problems()
    kmm_module = importlib.import_module(
        "midogpp_thesis.cvae.routing.mmd_kmm_mixture.kmm"
    )

    def _explode(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(kmm_module, "minimize", _explode)
    solution = solve_kmm_weights(problem, _optimization())
    expected = {
        source: 1.0 / len(problem.candidate_sources)
        for source in problem.candidate_sources
    }
    assert solution.weights == expected
    assert solution.used_uniform_fallback is True
    assert solution.fallback_reason == "solver_failure_uniform"
    assert "synthetic optimizer failure" in solution.solver_message


def test_failed_kkt_audit_returns_exact_equal_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, *_ = _problems()
    kmm_module = importlib.import_module(
        "midogpp_thesis.cvae.routing.mmd_kmm_mixture.kmm"
    )

    def _explode(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("synthetic KKT failure")

    monkeypatch.setattr(kmm_module, "lsq_linear", _explode)
    solution = solve_kmm_weights(problem, _optimization())
    expected = {
        source: 1.0 / len(problem.candidate_sources)
        for source in problem.candidate_sources
    }
    assert solution.weights == expected
    assert solution.fallback_reason == "postsolve_optimality_failure_uniform"
    assert "synthetic KKT failure" in solution.solver_message


def test_shared_frozen_nystroem_map_is_deterministic_and_hash_guarded() -> None:
    feature_map = FrozenNystroemFeatureMap(
        components=np.asarray(((0.0, 0.0), (1.0, 1.0))),
        normalization=np.eye(2),
        gamma=0.5,
        common_frame_hash="common-frame",
        preprocessing_hash="source-pool-scaler",
        candidate_pool_fit_hash="equal-count-source-pool",
        random_state=20260807,
    )
    features = np.asarray(((0.0, 0.0), (0.5, 0.5)))
    first = transform_frozen_nystroem(
        feature_map,
        features,
        common_frame_hash="common-frame",
        preprocessing_hash="source-pool-scaler",
    )
    second = transform_frozen_nystroem(
        feature_map,
        features,
        common_frame_hash="common-frame",
        preprocessing_hash="source-pool-scaler",
    )
    np.testing.assert_array_equal(first.values, second.values)
    assert first.values.flags.writeable is False
    assert first.preprocessing_hash == "source-pool-scaler"
    assert first.candidate_pool_fit_hash == "equal-count-source-pool"
    assert first.kernel_map_hash == feature_map.kernel_map_hash
    changed_map = FrozenNystroemFeatureMap(
        components=np.asarray(((0.0, 0.0), (1.0, 1.1))),
        normalization=np.eye(2),
        gamma=0.5,
        common_frame_hash="common-frame",
        preprocessing_hash="source-pool-scaler",
        candidate_pool_fit_hash="equal-count-source-pool",
        random_state=20260807,
    )
    assert changed_map.kernel_map_hash != feature_map.kernel_map_hash
    with pytest.raises(ProtocolError, match="preprocessing states"):
        transform_frozen_nystroem(
            feature_map,
            features,
            common_frame_hash="common-frame",
            preprocessing_hash="target-fitted-scaler",
        )


def test_source_only_responsibilities_are_stable_and_expose_no_label_api() -> None:
    config = PriorControlConfig(
        probability_clip=1e-6,
        temperature=1e-8,
        sensitivity_positive_priors=(0.35, 0.65),
    )
    prediction = prepare_source_only_responsibilities(
        np.asarray(((1.0 - 1e-12, 1e-12), (1e-12, 1.0 - 1e-12))),
        protocol=_protocol(),
        prior_model_hash="pooled-model",
        prior_fit_pool_hash="target-excluded-equal-pool",
        config=config,
    )
    assert np.isfinite(prediction.probabilities).all()
    assert prediction.probabilities.min() >= 1e-6
    assert prediction.target_labels_used is False
    assert len(prediction.responsibility_sha256) == 64
    assert len(prediction.prior_family_hash) == 64
    assert len(prediction.prior_state_hash) == 64
    changed = prepare_source_only_responsibilities(
        np.asarray(((0.1, 0.9), (0.9, 0.1))),
        protocol=_protocol(),
        prior_model_hash="pooled-model",
        prior_fit_pool_hash="target-excluded-equal-pool",
        config=config,
    )
    assert changed.responsibility_sha256 != prediction.responsibility_sha256
    assert changed.prior_family_hash == prediction.prior_family_hash
    assert changed.prior_control_hash == config.state_hash
    with pytest.raises(ProtocolError, match="prior prediction contract"):
        replace(prediction, target_labels_used=True)
    with pytest.raises(ProtocolError, match="kernel-feature provenance"):
        TransformedKernelFeatures(
            values=np.zeros((2, 2)),
            common_frame_hash="common-frame",
            preprocessing_hash="source-pool-scaler",
            candidate_pool_fit_hash="equal-count-source-pool",
            kernel_map_hash="kernel-map",
            target_rows_used_to_fit=True,
        )
    assert "label" not in inspect.signature(
        prepare_source_only_responsibilities
    ).parameters
    assert "label" not in inspect.signature(route_mmd_kmm).parameters
    assert "utility" not in inspect.signature(route_mmd_kmm).parameters
