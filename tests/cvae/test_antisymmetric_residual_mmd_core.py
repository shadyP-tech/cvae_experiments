from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.antisymmetric_residual_mmd import (
    AntisymmetricResidualConfig,
    build_antisymmetric_allocation,
    solve_antisymmetric_residual_mmd,
)
from midogpp_thesis.cvae.routing.mmd_kmm_mixture import (
    ConditionalContrastProblem,
    KernelMeanProblem,
    MMDKMMProtocol,
)


SOURCES = tuple(str(index) for index in range(8))


def _protocol(*, target: str = "8") -> MMDKMMProtocol:
    return MMDKMMProtocol(
        target_center=target,
        candidate_sources=SOURCES,
        support_case_ids=("case-a", "case-b"),
        evaluation_case_ids=("case-eval",),
        common_frame_hash="common-frame",
    )


def _problem(
    target_delta: object,
    *,
    protocol: MMDKMMProtocol | None = None,
    weak_quality: bool = False,
) -> ConditionalContrastProblem:
    routing_protocol = protocol or _protocol()
    delta = np.asarray(target_delta, dtype=np.float64)
    assert delta.shape == (8,)
    uniform = np.full(8, 0.125, dtype=np.float64)
    source_identity = np.eye(8, dtype=np.float64)
    source_class = np.stack((source_identity, source_identity), axis=1)
    target_class = np.stack((uniform + delta, uniform - delta), axis=0)
    class_weights = (0.5, 0.5)
    contrast_weight = 0.25
    augmented_source = np.concatenate(
        (
            np.sqrt(class_weights[0]) * source_class[:, 0],
            np.sqrt(class_weights[1]) * source_class[:, 1],
            np.sqrt(contrast_weight)
            * (source_class[:, 1] - source_class[:, 0]),
        ),
        axis=1,
    )
    augmented_target = np.concatenate(
        (
            np.sqrt(class_weights[0]) * target_class[0],
            np.sqrt(class_weights[1]) * target_class[1],
            np.sqrt(contrast_weight) * (target_class[1] - target_class[0]),
        )
    )
    kernel_problem = KernelMeanProblem(
        protocol=routing_protocol,
        candidate_sources=routing_protocol.candidate_sources,
        source_kernel_means=augmented_source,
        target_kernel_mean=augmented_target,
        common_frame_hash=routing_protocol.common_frame_hash,
        kernel_map_hash="shared-kernel-map",
        preprocessing_hash="source-pool-scaler",
        candidate_pool_fit_hash="equal-source-pool",
        kernel_transform_role="shared_frozen_source_pool_nystroem",
        prior_family_hash="source-only-prior-family",
        prior_control_hash="prior-control",
        prior_state_hash="prior-state",
        prior_sensitivity_positive_prior=None,
        target_kernel_feature_sha256="a" * 64,
        target_responsibility_sha256="b" * 64,
        source_replica_count=144,
        target_support_row_count=8,
        proxy_family="class_conditional_contrast_mmd_kmm",
    )
    weak = 0.5 if weak_quality else 2.0
    return ConditionalContrastProblem(
        kernel_problem=kernel_problem,
        source_class_kernel_means=source_class,
        target_class_kernel_means=target_class,
        class_weights=class_weights,
        contrast_weight=contrast_weight,
        soft_class_mass_by_case={
            "case-a": (weak, 2.0),
            "case-b": (2.0, 2.0),
        },
        soft_class_effective_rows_by_case={
            "case-a": (2.0, 2.0),
            "case-b": (2.0, 2.0),
        },
    )


def _config(**overrides: object) -> AntisymmetricResidualConfig:
    values: dict[str, object] = {
        "worst_variant_penalty": 1.0,
        "l2_shrinkage": 0.0,
        "minimum_robust_improvement": 1.0e-12,
        "solver_tolerance": 1.0e-12,
    }
    values.update(overrides)
    return AntisymmetricResidualConfig(**values)  # type: ignore[arg-type]


def _delta(first: float, second: float) -> np.ndarray:
    values = np.zeros(8, dtype=np.float64)
    values[0] = first
    values[1] = second
    return values


def _vector(values: Mapping[str, float]) -> np.ndarray:
    return np.asarray([values[source] for source in SOURCES], dtype=np.float64)


def test_solution_is_feasible_label_free_and_reports_every_axis() -> None:
    desired = _delta(0.04, -0.04)
    problem = _problem(desired)
    solution = solve_antisymmetric_residual_mmd(
        problem,
        support_case_problems={"case-a": problem},
        training_seed_problems={"17": problem},
        generation_seed_problems={"101": problem},
        prior_sensitivity_problems={"0.5": problem},
        config=_config(),
    )
    assert solution.used_uniform_fallback is False
    assert solution.claim_role == "proxy_compatibility_only"
    assert solution.labels_used is False
    assert set(solution.axis_diagnostics) == {
        "base",
        "support_case",
        "training_seed",
        "generation_seed",
        "class_prior_sensitivity",
    }
    class_zero = _vector(solution.class_0_weights)
    class_one = _vector(solution.class_1_weights)
    uniform = np.full(8, 0.125)
    np.testing.assert_allclose(class_zero + class_one, 2.0 * uniform, atol=1e-10)
    assert class_zero.sum() == pytest.approx(1.0)
    assert class_one.sum() == pytest.approx(1.0)
    assert class_zero.min() >= -1e-9
    assert class_one.min() >= -1e-9
    assert max(class_zero.max(), class_one.max()) <= 0.25 + 1e-9
    assert 1.0 / np.dot(class_zero, class_zero) >= 6.0 - 1e-8
    assert 1.0 / np.dot(class_one, class_one) >= 6.0 - 1e-8
    assert np.abs(class_zero - uniform).sum() <= 0.25 + 1e-8
    assert np.abs(class_one - uniform).sum() <= 0.25 + 1e-8
    assert set(solution.base_diagnostic.final_components) == {
        "class_0_weighted_mmd_squared",
        "class_1_weighted_mmd_squared",
        "contrast_weighted_mmd_squared",
        "conditional_discrepancy",
    }


def test_zero_residual_is_exact_uniform_and_matches_conditional_identity() -> None:
    problem = _problem(np.zeros(8))
    solution = solve_antisymmetric_residual_mmd(problem, config=_config())
    expected = {source: 0.125 for source in SOURCES}
    assert solution.used_uniform_fallback is True
    assert solution.fallback_reason == "nonpositive_robust_improvement_uniform"
    assert dict(solution.delta) == {source: 0.0 for source in SOURCES}
    assert dict(solution.class_0_weights) == expected
    assert dict(solution.class_1_weights) == expected
    assert dict(solution.base_diagnostic.uniform_components) == pytest.approx(
        problem.component_losses(expected)
    )
    assert dict(solution.base_diagnostic.final_components) == pytest.approx(
        problem.component_losses(expected)
    )


def test_solver_can_reach_the_joint_cap_and_l1_boundary() -> None:
    desired = _delta(0.125, -0.125)
    solution = solve_antisymmetric_residual_mmd(
        _problem(desired), config=_config()
    )
    assert solution.used_uniform_fallback is False
    observed = _vector(solution.delta)
    np.testing.assert_allclose(observed[:2], (0.125, -0.125), atol=2e-6)
    assert solution.maximum_source_weight == pytest.approx(0.25, abs=2e-6)
    assert solution.class_0_uniform_l1 == pytest.approx(0.25, abs=3e-6)
    assert solution.class_1_uniform_l1 == pytest.approx(0.25, abs=3e-6)


def test_any_variant_worsening_fails_closed_even_when_mean_improves() -> None:
    base = _problem(_delta(0.08, -0.08))
    conflicting = _problem(_delta(-0.01, 0.01))
    solution = solve_antisymmetric_residual_mmd(
        base,
        support_case_problems={"conflict": conflicting},
        config=_config(worst_variant_penalty=0.0),
    )
    assert solution.proposed_robust_improvement > 0.0
    assert solution.all_variants_nonworsening is False
    assert solution.used_uniform_fallback is True
    assert solution.fallback_reason == "variant_worsening_uniform"
    assert dict(solution.delta) == {source: 0.0 for source in SOURCES}
    assert solution.axis_diagnostics[
        "support_case"
    ].maximum_proposed_variant_worsening > 0.0


def test_swapping_class_targets_negates_the_residual_and_swaps_weights() -> None:
    desired = _delta(0.055, -0.055)
    forward = solve_antisymmetric_residual_mmd(
        _problem(desired), config=_config()
    )
    swapped = solve_antisymmetric_residual_mmd(
        _problem(-desired), config=_config()
    )
    assert forward.used_uniform_fallback is False
    assert swapped.used_uniform_fallback is False
    np.testing.assert_allclose(
        _vector(swapped.delta), -_vector(forward.delta), atol=1e-8
    )
    np.testing.assert_allclose(
        _vector(swapped.class_0_weights),
        _vector(forward.class_1_weights),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        _vector(swapped.class_1_weights),
        _vector(forward.class_0_weights),
        atol=1e-8,
    )


def test_integer_allocation_is_exact_positive_antisymmetric_and_deterministic() -> None:
    delta = {
        source: value
        for source, value in zip(
            reversed(SOURCES),
            reversed((0.5 / 1024.0, -0.5 / 1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            strict=True,
        )
    }
    first = build_antisymmetric_allocation(delta)
    second = build_antisymmetric_allocation(dict(reversed(tuple(delta.items()))))
    assert dict(first.integer_offsets) == dict(second.integer_offsets)
    assert first.allocation_hash == second.allocation_hash
    assert sum(first.integer_offsets.values()) == 0
    assert sum(first.class_0_allocations.values()) == 1024
    assert sum(first.class_1_allocations.values()) == 1024
    assert all(value > 0 for value in first.class_0_allocations.values())
    assert all(value > 0 for value in first.class_1_allocations.values())
    for source in SOURCES:
        offset = first.integer_offsets[source]
        assert first.class_0_allocations[source] == 128 + offset
        assert first.class_1_allocations[source] == 128 - offset
    assert first.integer_offsets["0"] == 1
    assert first.integer_offsets["1"] == -1

    boundary = build_antisymmetric_allocation(
        {source: value for source, value in zip(SOURCES, _delta(0.125, -0.125))}
    )
    assert boundary.class_0_allocations["0"] == 255
    assert boundary.class_0_allocations["1"] == 1
    assert boundary.class_1_allocations["0"] == 1
    assert boundary.class_1_allocations["1"] == 255


def test_solver_exception_fails_closed_to_bit_exact_uniform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(
        "midogpp_thesis.cvae.routing.antisymmetric_residual_mmd.solver.minimize",
        fail,
    )
    solution = solve_antisymmetric_residual_mmd(
        _problem(_delta(0.04, -0.04)), config=_config()
    )
    expected = {source: 0.125 for source in SOURCES}
    assert solution.used_uniform_fallback is True
    assert solution.fallback_reason == "solver_failure_uniform"
    assert solution.solver_success is False
    assert dict(solution.class_0_weights) == expected
    assert dict(solution.class_1_weights) == expected


def test_soft_quality_and_malformed_inputs_fail_closed_or_raise() -> None:
    weak = solve_antisymmetric_residual_mmd(
        _problem(_delta(0.04, -0.04), weak_quality=True), config=_config()
    )
    assert weak.used_uniform_fallback is True
    assert weak.fallback_reason == "insufficient_soft_class_quality_uniform"
    assert weak.support_quality_passed is False
    assert dict(weak.class_0_weights) == {source: 0.125 for source in SOURCES}

    with pytest.raises(ProtocolError, match="configuration"):
        AntisymmetricResidualConfig(max_source_weight=0.3)
    with pytest.raises(ProtocolError, match="ConditionalContrastProblem"):
        solve_antisymmetric_residual_mmd(
            object(), config=_config()  # type: ignore[arg-type]
        )
    with pytest.raises(ProtocolError, match="problem family"):
        solve_antisymmetric_residual_mmd(
            _problem(_delta(0.04, -0.04)),
            support_case_problems={
                "wrong-target": _problem(
                    _delta(0.04, -0.04), protocol=_protocol(target="9")
                )
            },
            config=_config(),
        )
    with pytest.raises(ProtocolError, match="residual"):
        build_antisymmetric_allocation(
            {source: (float("nan") if source == "0" else 0.0) for source in SOURCES}
        )
