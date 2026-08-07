"""Robust label-free solver for antisymmetric class-paired source weights."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import minimize

from ...protocol import ProtocolError
from ..mmd_kmm_mixture.conditional import ConditionalContrastProblem
from .contracts import (
    AntisymmetricAxisDiagnostic,
    AntisymmetricResidualConfig,
    AntisymmetricResidualSolution,
    AntisymmetricVariantDiagnostic,
    PROXY_CLAIM_ROLE,
)


@dataclass(frozen=True)
class _NamedProblem:
    axis: str
    variant_id: str
    problem: ConditionalContrastProblem


def solve_antisymmetric_residual_mmd(
    base_problem: ConditionalContrastProblem,
    config: AntisymmetricResidualConfig | None = None,
    *,
    support_case_problems: Mapping[object, ConditionalContrastProblem] | None = None,
    training_seed_problems: Mapping[object, ConditionalContrastProblem] | None = None,
    generation_seed_problems: Mapping[object, ConditionalContrastProblem] | None = None,
    prior_sensitivity_problems: Mapping[
        object, ConditionalContrastProblem
    ] | None = None,
    prior_problems: Mapping[object, ConditionalContrastProblem] | None = None,
) -> AntisymmetricResidualSolution:
    """Minimize a robust conditional-MMD proxy with class-paired weights.

    For uniform ``u`` and a fitted residual ``d``, the two generation-class
    routes are ``w0 = u + d`` and ``w1 = u - d``.  The SLSQP variables are
    ``(d, t)`` where ``t`` is a worst-variant epigraph.  The equality,
    per-class box, effective-source, and L1 trust-region constraints all live
    inside the numerical problem.  A successful point is audited but never
    projected post hoc.

    Every loss is a label-free compatibility proxy reconstructed from
    :class:`ConditionalContrastProblem`; no downstream label or utility enters
    this function.
    """

    controls = config or AntisymmetricResidualConfig()
    if not isinstance(controls, AntisymmetricResidualConfig):
        raise ProtocolError("Antisymmetric solver config has the wrong type.")
    if prior_sensitivity_problems is not None and prior_problems is not None:
        raise ProtocolError(
            "Specify either prior_sensitivity_problems or prior_problems, not both."
        )
    prior_variants = (
        prior_sensitivity_problems
        if prior_sensitivity_problems is not None
        else prior_problems
    )
    named = _named_problems(
        base_problem,
        support_case_problems=support_case_problems,
        training_seed_problems=training_seed_problems,
        generation_seed_problems=generation_seed_problems,
        prior_sensitivity_problems=prior_variants,
    )
    sources = base_problem.kernel_problem.candidate_sources
    n_sources = len(sources)
    uniform = np.full(n_sources, 1.0 / float(n_sources), dtype=np.float64)
    cap = float(controls.max_source_weight)
    concentration_limit = 1.0 / float(controls.minimum_effective_sources)
    if (
        float(uniform.max()) > cap + 1.0e-15
        or float(np.dot(uniform, uniform)) > concentration_limit + 1.0e-15
        or float(controls.minimum_effective_sources) > float(n_sources)
    ):
        raise ProtocolError(
            "Antisymmetric density constraints exclude the equal-union anchor."
        )

    support_quality = _support_quality_passes(named, controls)
    zero = np.zeros(n_sources, dtype=np.float64)
    if not support_quality:
        return _solution(
            named=named,
            sources=sources,
            uniform=uniform,
            proposed_delta=zero,
            final_delta=zero,
            config=controls,
            reason="insufficient_soft_class_quality_uniform",
            solver_success=False,
            solver_message="Soft class mass/effective-row gate failed.",
            solver_iterations=0,
            support_quality_passed=False,
        )

    uniform_losses = np.asarray(
        [_loss_and_gradient(item.problem, uniform, zero)[0] for item in named],
        dtype=np.float64,
    )
    number_of_variables = n_sources + 1
    delta_slice = slice(0, n_sources)
    epigraph_index = n_sources
    shrinkage = float(controls.l2_shrinkage)
    worst_penalty = float(controls.worst_variant_penalty)

    def objective(values: np.ndarray) -> float:
        delta = values[delta_slice]
        losses = np.asarray(
            [_loss_and_gradient(item.problem, uniform, delta)[0] for item in named]
        )
        return float(
            losses.mean()
            + worst_penalty * values[epigraph_index]
            + shrinkage * np.dot(delta, delta)
        )

    def objective_jacobian(values: np.ndarray) -> np.ndarray:
        delta = values[delta_slice]
        gradients = np.asarray(
            [_loss_and_gradient(item.problem, uniform, delta)[1] for item in named]
        )
        output = np.zeros(number_of_variables, dtype=np.float64)
        output[delta_slice] = gradients.mean(axis=0) + 2.0 * shrinkage * delta
        output[epigraph_index] = worst_penalty
        return output

    equality_jacobian = np.zeros(number_of_variables, dtype=np.float64)
    equality_jacobian[delta_slice] = 1.0
    class_zero_effective_jacobian = np.zeros(number_of_variables, dtype=np.float64)
    class_one_effective_jacobian = np.zeros(number_of_variables, dtype=np.float64)

    def class_zero_effective(values: np.ndarray) -> float:
        weights = uniform + values[delta_slice]
        return float(concentration_limit - np.dot(weights, weights))

    def class_zero_effective_jac(values: np.ndarray) -> np.ndarray:
        class_zero_effective_jacobian[delta_slice] = -2.0 * (
            uniform + values[delta_slice]
        )
        return class_zero_effective_jacobian.copy()

    def class_one_effective(values: np.ndarray) -> float:
        weights = uniform - values[delta_slice]
        return float(concentration_limit - np.dot(weights, weights))

    def class_one_effective_jac(values: np.ndarray) -> np.ndarray:
        class_one_effective_jacobian[delta_slice] = 2.0 * (
            uniform - values[delta_slice]
        )
        return class_one_effective_jacobian.copy()

    # ||d||_1 <= radius is exactly the intersection of the 2**K halfspaces
    # s^T d <= radius for all sign vectors s in {-1,+1}^K.  K is seven or
    # eight in the fenced ConditionalContrastProblem protocol, so this direct
    # smooth representation is small and avoids a post-solve L1 projection.
    sign_matrix = np.asarray(
        tuple(product((-1.0, 1.0), repeat=n_sources)), dtype=np.float64
    )
    l1_jacobian = np.zeros(
        (len(sign_matrix), number_of_variables), dtype=np.float64
    )
    l1_jacobian[:, delta_slice] = -sign_matrix

    def l1_constraints(values: np.ndarray) -> np.ndarray:
        return float(controls.maximum_uniform_l1) - sign_matrix @ values[delta_slice]

    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda values: float(values[delta_slice].sum()),
            "jac": lambda values: equality_jacobian,
        },
        {
            "type": "ineq",
            "fun": class_zero_effective,
            "jac": class_zero_effective_jac,
        },
        {
            "type": "ineq",
            "fun": class_one_effective,
            "jac": class_one_effective_jac,
        },
        {
            "type": "ineq",
            "fun": l1_constraints,
            "jac": lambda values: l1_jacobian,
        },
    ]
    for item in named:
        constraints.append(_epigraph_constraint(item.problem, uniform, n_sources))

    lower_delta = max(-float(uniform[0]), float(uniform[0]) - cap)
    upper_delta = min(cap - float(uniform[0]), float(uniform[0]))
    bounds = tuple((lower_delta, upper_delta) for _ in sources) + ((0.0, None),)
    initial = np.concatenate((zero, (float(uniform_losses.max()),)))
    try:
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=objective_jacobian,
            bounds=bounds,
            constraints=tuple(constraints),
            options={
                "ftol": float(controls.solver_tolerance),
                "maxiter": int(controls.max_iterations),
                "disp": False,
            },
        )
    except Exception as exc:  # numerical failures are safe-control outcomes.
        return _solution(
            named=named,
            sources=sources,
            uniform=uniform,
            proposed_delta=zero,
            final_delta=zero,
            config=controls,
            reason="solver_failure_uniform",
            solver_success=False,
            solver_message=f"{type(exc).__name__}: {exc}",
            solver_iterations=0,
            support_quality_passed=True,
        )

    iterations = int(getattr(result, "nit", 0))
    message = str(getattr(result, "message", ""))
    if (
        not bool(result.success)
        or np.asarray(result.x).shape != (number_of_variables,)
        or not np.isfinite(result.x).all()
    ):
        return _solution(
            named=named,
            sources=sources,
            uniform=uniform,
            proposed_delta=zero,
            final_delta=zero,
            config=controls,
            reason="solver_failure_uniform",
            solver_success=False,
            solver_message=message,
            solver_iterations=iterations,
            support_quality_passed=True,
        )

    delta = np.asarray(result.x[delta_slice], dtype=np.float64)
    epigraph = float(result.x[epigraph_index])
    losses = np.asarray(
        [_loss_and_gradient(item.problem, uniform, delta)[0] for item in named],
        dtype=np.float64,
    )
    feasibility_tolerance = max(
        1.0e-8, 100.0 * float(controls.solver_tolerance)
    )
    if not _feasible(
        delta,
        uniform=uniform,
        cap=cap,
        concentration_limit=concentration_limit,
        maximum_uniform_l1=float(controls.maximum_uniform_l1),
        epigraph=epigraph,
        losses=losses,
        tolerance=feasibility_tolerance,
    ):
        return _solution(
            named=named,
            sources=sources,
            uniform=uniform,
            proposed_delta=delta,
            final_delta=zero,
            config=controls,
            reason="postsolve_infeasible_uniform",
            solver_success=False,
            solver_message=message,
            solver_iterations=iterations,
            support_quality_passed=True,
        )

    uniform_robust = _robust_objective(uniform_losses, zero, controls)
    proposed_robust = _robust_objective(losses, delta, controls)
    improvement = uniform_robust - proposed_robust
    if (
        not math.isfinite(improvement)
        or improvement <= float(controls.minimum_robust_improvement)
    ):
        return _solution(
            named=named,
            sources=sources,
            uniform=uniform,
            proposed_delta=delta,
            final_delta=zero,
            config=controls,
            reason="nonpositive_robust_improvement_uniform",
            solver_success=True,
            solver_message=message,
            solver_iterations=iterations,
            support_quality_passed=True,
        )
    if np.any(
        losses
        > uniform_losses + float(controls.variant_worsening_tolerance)
    ):
        return _solution(
            named=named,
            sources=sources,
            uniform=uniform,
            proposed_delta=delta,
            final_delta=zero,
            config=controls,
            reason="variant_worsening_uniform",
            solver_success=True,
            solver_message=message,
            solver_iterations=iterations,
            support_quality_passed=True,
        )
    return _solution(
        named=named,
        sources=sources,
        uniform=uniform,
        proposed_delta=delta,
        final_delta=delta,
        config=controls,
        reason=None,
        solver_success=True,
        solver_message=message,
        solver_iterations=iterations,
        support_quality_passed=True,
    )


def solve_antisymmetric_residual_weights(
    base_problem: ConditionalContrastProblem,
    config: AntisymmetricResidualConfig | None = None,
    **variant_problems: object,
) -> AntisymmetricResidualSolution:
    """Compatibility alias for :func:`solve_antisymmetric_residual_mmd`."""

    return solve_antisymmetric_residual_mmd(
        base_problem, config=config, **variant_problems  # type: ignore[arg-type]
    )


def _named_problems(
    base_problem: ConditionalContrastProblem,
    *,
    support_case_problems: Mapping[object, ConditionalContrastProblem] | None,
    training_seed_problems: Mapping[object, ConditionalContrastProblem] | None,
    generation_seed_problems: Mapping[object, ConditionalContrastProblem] | None,
    prior_sensitivity_problems: Mapping[object, ConditionalContrastProblem] | None,
) -> tuple[_NamedProblem, ...]:
    if not isinstance(base_problem, ConditionalContrastProblem):
        raise ProtocolError(
            "Antisymmetric routing requires a ConditionalContrastProblem base."
        )
    named: list[_NamedProblem] = [_NamedProblem("base", "base", base_problem)]
    groups = (
        ("support_case", support_case_problems),
        ("training_seed", training_seed_problems),
        ("generation_seed", generation_seed_problems),
        ("class_prior_sensitivity", prior_sensitivity_problems),
    )
    for axis, values in groups:
        normalized = _problem_mapping(values, axis)
        named.extend(
            _NamedProblem(axis, variant_id, problem)
            for variant_id, problem in normalized.items()
        )
    _validate_problem_family(tuple(named))
    return tuple(named)


def _problem_mapping(
    values: Mapping[object, ConditionalContrastProblem] | None,
    axis: str,
) -> dict[str, ConditionalContrastProblem]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ProtocolError(f"Antisymmetric {axis} variants must be a mapping.")
    output: dict[str, ConditionalContrastProblem] = {}
    for raw_key, problem in values.items():
        key = str(raw_key)
        if (
            not key
            or key.strip() != key
            or key in output
            or not isinstance(problem, ConditionalContrastProblem)
        ):
            raise ProtocolError(f"Antisymmetric {axis} variant mapping is invalid.")
        output[key] = problem
    return {key: output[key] for key in sorted(output)}


def _validate_problem_family(named: Sequence[_NamedProblem]) -> None:
    base = named[0].problem
    base_kernel = base.kernel_problem
    base_protocol = base_kernel.protocol
    sources = base_kernel.candidate_sources
    if base_kernel.claim_role != PROXY_CLAIM_ROLE:
        raise ProtocolError("Antisymmetric base problem crossed the proxy claim role.")
    for item in named:
        problem = item.problem
        kernel = problem.kernel_problem
        protocol = kernel.protocol
        if (
            kernel.candidate_sources != sources
            or protocol != base_protocol
            or protocol.candidate_sources != sources
            or protocol.target_center != base_protocol.target_center
            or protocol.support_partition_hash != base_protocol.support_partition_hash
            or kernel.common_frame_hash != base_kernel.common_frame_hash
            or kernel.kernel_map_hash != base_kernel.kernel_map_hash
            or kernel.preprocessing_hash != base_kernel.preprocessing_hash
            or kernel.candidate_pool_fit_hash != base_kernel.candidate_pool_fit_hash
            or kernel.claim_role != PROXY_CLAIM_ROLE
            or tuple(problem.class_weights) != tuple(base.class_weights)
            or not math.isclose(
                float(problem.contrast_weight),
                float(base.contrast_weight),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or protocol.support_labels_used is not False
            or protocol.evaluation_labels_available_to_router is not False
            or protocol.heldout_evaluation_embeddings_available_to_own_route
            is not False
            or protocol.source_experts_frozen is not True
            or protocol.target_expert_excluded is not True
        ):
            raise ProtocolError(
                "Antisymmetric robust variants crossed a routing/problem family."
            )


def _support_quality_passes(
    named: Sequence[_NamedProblem], config: AntisymmetricResidualConfig
) -> bool:
    return all(
        value >= float(config.minimum_soft_class_mass_per_case)
        for item in named
        for pair in item.problem.soft_class_mass_by_case.values()
        for value in pair
    ) and all(
        value >= float(config.minimum_soft_class_effective_rows_per_case)
        for item in named
        for pair in item.problem.soft_class_effective_rows_by_case.values()
        for value in pair
    )


def _loss_and_gradient(
    problem: ConditionalContrastProblem,
    uniform: np.ndarray,
    delta: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, float]]:
    source = np.asarray(problem.source_class_kernel_means, dtype=np.float64)
    target = np.asarray(problem.target_class_kernel_means, dtype=np.float64)
    class_zero_weights = uniform + delta
    class_one_weights = uniform - delta
    discrepancy_zero = class_zero_weights @ source[:, 0] - target[0]
    discrepancy_one = class_one_weights @ source[:, 1] - target[1]
    contrast_delta = discrepancy_one - discrepancy_zero
    class_zero = float(problem.class_weights[0]) * float(
        np.dot(discrepancy_zero, discrepancy_zero)
    )
    class_one = float(problem.class_weights[1]) * float(
        np.dot(discrepancy_one, discrepancy_one)
    )
    contrast = float(problem.contrast_weight) * float(
        np.dot(contrast_delta, contrast_delta)
    )
    total = class_zero + class_one + contrast
    gradient = (
        2.0
        * float(problem.class_weights[0])
        * (source[:, 0] @ discrepancy_zero)
        - 2.0
        * float(problem.class_weights[1])
        * (source[:, 1] @ discrepancy_one)
        - 2.0
        * float(problem.contrast_weight)
        * ((source[:, 1] + source[:, 0]) @ contrast_delta)
    )
    components = {
        "class_0_weighted_mmd_squared": max(0.0, class_zero),
        "class_1_weighted_mmd_squared": max(0.0, class_one),
        "contrast_weighted_mmd_squared": max(0.0, contrast),
        "conditional_discrepancy": max(0.0, total),
    }
    return max(0.0, total), np.asarray(gradient, dtype=np.float64), components


def _epigraph_constraint(
    problem: ConditionalContrastProblem,
    uniform: np.ndarray,
    n_sources: int,
) -> dict[str, object]:
    def fun(values: np.ndarray) -> float:
        loss = _loss_and_gradient(problem, uniform, values[:n_sources])[0]
        return float(values[n_sources] - loss)

    def jac(values: np.ndarray) -> np.ndarray:
        gradient = _loss_and_gradient(problem, uniform, values[:n_sources])[1]
        output = np.empty(n_sources + 1, dtype=np.float64)
        output[:n_sources] = -gradient
        output[n_sources] = 1.0
        return output

    return {"type": "ineq", "fun": fun, "jac": jac}


def _robust_objective(
    losses: np.ndarray,
    delta: np.ndarray,
    config: AntisymmetricResidualConfig,
) -> float:
    return float(
        losses.mean()
        + float(config.worst_variant_penalty) * losses.max()
        + float(config.l2_shrinkage) * np.dot(delta, delta)
    )


def _feasible(
    delta: np.ndarray,
    *,
    uniform: np.ndarray,
    cap: float,
    concentration_limit: float,
    maximum_uniform_l1: float,
    epigraph: float,
    losses: np.ndarray,
    tolerance: float,
) -> bool:
    class_zero = uniform + delta
    class_one = uniform - delta
    return bool(
        np.isfinite(delta).all()
        and math.isfinite(epigraph)
        and abs(float(delta.sum())) <= tolerance
        and float(class_zero.min()) >= -tolerance
        and float(class_one.min()) >= -tolerance
        and float(class_zero.max()) <= cap + tolerance
        and float(class_one.max()) <= cap + tolerance
        and abs(float(class_zero.sum()) - 1.0) <= tolerance
        and abs(float(class_one.sum()) - 1.0) <= tolerance
        and float(np.dot(class_zero, class_zero))
        <= concentration_limit + tolerance
        and float(np.dot(class_one, class_one)) <= concentration_limit + tolerance
        and float(np.abs(delta).sum()) <= maximum_uniform_l1 + tolerance
        and epigraph + tolerance >= float(losses.max())
    )


def _solution(
    *,
    named: Sequence[_NamedProblem],
    sources: tuple[str, ...],
    uniform: np.ndarray,
    proposed_delta: np.ndarray,
    final_delta: np.ndarray,
    config: AntisymmetricResidualConfig,
    reason: str | None,
    solver_success: bool,
    solver_message: str,
    solver_iterations: int,
    support_quality_passed: bool,
) -> AntisymmetricResidualSolution:
    proposed_losses = np.asarray(
        [
            _loss_and_gradient(item.problem, uniform, proposed_delta)[0]
            for item in named
        ],
        dtype=np.float64,
    )
    final_losses = np.asarray(
        [_loss_and_gradient(item.problem, uniform, final_delta)[0] for item in named],
        dtype=np.float64,
    )
    uniform_delta = np.zeros_like(uniform)
    uniform_losses = np.asarray(
        [
            _loss_and_gradient(item.problem, uniform, uniform_delta)[0]
            for item in named
        ],
        dtype=np.float64,
    )
    proposed_robust = _robust_objective(proposed_losses, proposed_delta, config)
    final_robust = _robust_objective(final_losses, final_delta, config)
    uniform_robust = _robust_objective(uniform_losses, uniform_delta, config)
    tolerance = float(config.variant_worsening_tolerance)
    proposed_nonworsening = proposed_losses <= uniform_losses + tolerance

    diagnostics: list[AntisymmetricVariantDiagnostic] = []
    for index, item in enumerate(named):
        uniform_components = _loss_and_gradient(
            item.problem, uniform, uniform_delta
        )[2]
        proposed_components = _loss_and_gradient(
            item.problem, uniform, proposed_delta
        )[2]
        final_components = _loss_and_gradient(item.problem, uniform, final_delta)[2]
        diagnostics.append(
            AntisymmetricVariantDiagnostic(
                axis=item.axis,
                variant_id=item.variant_id,
                uniform_components=uniform_components,
                proposed_components=proposed_components,
                final_components=final_components,
                proposed_improvement=float(
                    uniform_losses[index] - proposed_losses[index]
                ),
                final_improvement=float(uniform_losses[index] - final_losses[index]),
                proposed_worsened=not bool(proposed_nonworsening[index]),
            )
        )
    axis_diagnostics: dict[str, AntisymmetricAxisDiagnostic] = {}
    for axis in dict.fromkeys(item.axis for item in named):
        indices = [index for index, item in enumerate(named) if item.axis == axis]
        uniform_axis = uniform_losses[indices]
        proposed_axis = proposed_losses[indices]
        final_axis = final_losses[indices]
        improvements = uniform_axis - proposed_axis
        worsening = proposed_axis - uniform_axis
        axis_diagnostics[axis] = AntisymmetricAxisDiagnostic(
            axis=axis,
            variant_ids=tuple(named[index].variant_id for index in indices),
            uniform_mean_loss=float(uniform_axis.mean()),
            proposed_mean_loss=float(proposed_axis.mean()),
            final_mean_loss=float(final_axis.mean()),
            uniform_worst_loss=float(uniform_axis.max()),
            proposed_worst_loss=float(proposed_axis.max()),
            final_worst_loss=float(final_axis.max()),
            minimum_proposed_variant_improvement=float(improvements.min()),
            maximum_proposed_variant_worsening=float(max(0.0, worsening.max())),
            all_proposed_variants_nonworsening=bool(
                np.all(proposed_axis <= uniform_axis + tolerance)
            ),
        )

    proposed_class_zero = uniform + proposed_delta
    proposed_class_one = uniform - proposed_delta
    final_class_zero = uniform + final_delta
    final_class_one = uniform - final_delta
    return AntisymmetricResidualSolution(
        candidate_sources=sources,
        uniform_weights=_mapping(sources, uniform),
        proposed_delta=_mapping(sources, proposed_delta),
        proposed_class_0_weights=_mapping(sources, proposed_class_zero),
        proposed_class_1_weights=_mapping(sources, proposed_class_one),
        delta=_mapping(sources, final_delta),
        class_0_weights=_mapping(sources, final_class_zero),
        class_1_weights=_mapping(sources, final_class_one),
        robust_objective=final_robust,
        uniform_robust_objective=uniform_robust,
        proposed_robust_improvement=float(uniform_robust - proposed_robust),
        robust_improvement=float(uniform_robust - final_robust),
        proposed_mean_conditional_loss=float(proposed_losses.mean()),
        final_mean_conditional_loss=float(final_losses.mean()),
        uniform_mean_conditional_loss=float(uniform_losses.mean()),
        proposed_worst_conditional_loss=float(proposed_losses.max()),
        final_worst_conditional_loss=float(final_losses.max()),
        uniform_worst_conditional_loss=float(uniform_losses.max()),
        l2_penalty_value=float(
            float(config.l2_shrinkage) * np.dot(final_delta, final_delta)
        ),
        class_0_effective_source_count=float(
            1.0 / np.dot(final_class_zero, final_class_zero)
        ),
        class_1_effective_source_count=float(
            1.0 / np.dot(final_class_one, final_class_one)
        ),
        maximum_source_weight=float(
            max(final_class_zero.max(), final_class_one.max())
        ),
        class_0_uniform_l1=float(np.abs(final_class_zero - uniform).sum()),
        class_1_uniform_l1=float(np.abs(final_class_one - uniform).sum()),
        used_uniform_fallback=reason is not None,
        fallback_reason=reason,
        solver_success=bool(solver_success),
        solver_message=str(solver_message),
        solver_iterations=int(solver_iterations),
        solver_version=str(scipy.__version__),
        variant_diagnostics=tuple(diagnostics),
        axis_diagnostics=axis_diagnostics,
        support_quality_passed=bool(support_quality_passed),
        all_variants_nonworsening=bool(np.all(proposed_nonworsening)),
    )


def _mapping(sources: Sequence[str], values: np.ndarray) -> dict[str, float]:
    return {
        str(source): float(value)
        for source, value in zip(sources, values, strict=True)
    }


__all__ = (
    "solve_antisymmetric_residual_mmd",
    "solve_antisymmetric_residual_weights",
)
