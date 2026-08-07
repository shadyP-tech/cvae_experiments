"""Convex, equal-union-anchored kernel mean matching weights."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy
from scipy.optimize import lsq_linear, minimize

from ...protocol import ProtocolError
from .config import KMMOptimizationConfig
from .contracts import KMMWeightSolution, KernelMeanProblem, weight_mapping


SOLVER_METHOD = "scipy_slsqp_continuous_convex_proxy"


def solve_kmm_weights(
    problem: KernelMeanProblem,
    config: KMMOptimizationConfig,
) -> KMMWeightSolution:
    """Minimize squared MMD plus shrinkage under dense-simplex constraints.

    This objective is a label-free compatibility proxy.  A smaller objective
    is not downstream utility evidence.  Numerical failure, infeasibility, or
    insufficient proxy improvement returns equal-union exactly.
    """

    sources = problem.candidate_sources
    source_means = np.asarray(problem.source_kernel_means, dtype=np.float64)
    target_mean = np.asarray(problem.target_kernel_mean, dtype=np.float64)
    n_sources = len(sources)
    uniform = np.full(n_sources, 1.0 / float(n_sources), dtype=np.float64)
    cap = float(config.max_source_weight)
    minimum_effective = float(config.minimum_effective_sources)
    concentration_limit = 1.0 / minimum_effective
    regularization = float(config.regularization)
    tolerance = float(config.solver_tolerance)
    if (
        minimum_effective > float(n_sources)
        or float(uniform.max()) > cap + 1e-15
        or float(np.dot(uniform, uniform)) > concentration_limit + 1e-15
    ):
        raise ProtocolError("MMD/KMM constraints exclude the equal-union anchor.")

    def terms(weights: np.ndarray) -> tuple[float, float, float]:
        discrepancy = weights @ source_means - target_mean
        mmd_squared = max(0.0, float(np.dot(discrepancy, discrepancy)))
        delta = weights - uniform
        penalty = regularization * float(np.dot(delta, delta))
        return mmd_squared, penalty, mmd_squared + penalty

    def objective(weights: np.ndarray) -> float:
        return terms(weights)[2]

    def jacobian(weights: np.ndarray) -> np.ndarray:
        discrepancy = weights @ source_means - target_mean
        return 2.0 * (source_means @ discrepancy) + 2.0 * regularization * (
            weights - uniform
        )

    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                concentration_limit - np.dot(weights, weights)
            ),
            "jac": lambda weights: -2.0 * weights,
        },
    )
    uniform_mmd, _, uniform_objective = terms(uniform)
    try:
        result = minimize(
            objective,
            uniform.copy(),
            method="SLSQP",
            jac=jacobian,
            bounds=tuple((0.0, cap) for _ in sources),
            constraints=constraints,
            options={
                "ftol": tolerance,
                "maxiter": int(config.max_iterations),
                "disp": False,
            },
        )
    except Exception as exc:  # scipy failures are safe-control outcomes.
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="solver_failure_uniform",
            solver_message=f"{type(exc).__name__}: {exc}",
        )
    iterations = int(getattr(result, "nit", 0))
    message = str(getattr(result, "message", ""))
    if not bool(result.success) or not np.isfinite(result.x).all():
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="solver_failure_uniform",
            solver_message=message,
            solver_iterations=iterations,
        )
    weights = np.asarray(result.x, dtype=np.float64)
    feasibility_tolerance = max(1e-8, 100.0 * tolerance)
    if (
        abs(float(weights.sum()) - 1.0) > feasibility_tolerance
        or float(weights.min()) < -feasibility_tolerance
        or float(weights.max()) > cap + feasibility_tolerance
        or float(np.dot(weights, weights))
        > concentration_limit + feasibility_tolerance
    ):
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="postsolve_infeasible_uniform",
            solver_message=message,
            solver_iterations=iterations,
        )
    # Project roundoff only, then verify the scientific constraints again.
    weights = np.maximum(weights, 0.0)
    weights /= float(weights.sum())
    if (
        float(weights.max()) > cap + feasibility_tolerance
        or float(np.dot(weights, weights))
        > concentration_limit + feasibility_tolerance
    ):
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="postsolve_infeasible_uniform",
            solver_message=message,
            solver_iterations=iterations,
        )
    try:
        optimality_residual = _kkt_stationarity_residual(
            weights,
            jacobian(weights),
            cap=cap,
            concentration_limit=concentration_limit,
            solver_tolerance=tolerance,
        )
    except Exception as exc:
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="postsolve_optimality_failure_uniform",
            solver_message=f"{message}; KKT audit: {type(exc).__name__}: {exc}",
            solver_iterations=iterations,
        )
    if optimality_residual > float(config.optimality_tolerance):
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="postsolve_optimality_failure_uniform",
            solver_message=message,
            solver_iterations=iterations,
        )
    mmd_squared, penalty, proxy_objective = terms(weights)
    improvement = uniform_objective - proxy_objective
    if (
        not np.isfinite(improvement)
        or improvement <= float(config.minimum_proxy_improvement)
    ):
        return _uniform_solution(
            sources,
            uniform_mmd=uniform_mmd,
            uniform_objective=uniform_objective,
            reason="insufficient_proxy_improvement_uniform",
            solver_success=True,
            solver_message=message,
            solver_iterations=iterations,
        )
    delta = weights - uniform
    effective = float(1.0 / np.dot(weights, weights))
    return KMMWeightSolution(
        candidate_sources=sources,
        uniform_weights=weight_mapping(sources, uniform),
        weights=weight_mapping(sources, weights),
        delta=weight_mapping(sources, delta),
        proxy_objective=float(proxy_objective),
        uniform_proxy_objective=float(uniform_objective),
        proxy_improvement=float(improvement),
        mmd_squared=float(mmd_squared),
        uniform_mmd_squared=float(uniform_mmd),
        regularization_value=float(penalty),
        effective_source_count=effective,
        maximum_source_weight=float(weights.max()),
        used_uniform_fallback=False,
        fallback_reason=None,
        solver_success=True,
        solver_message=message,
        solver_iterations=iterations,
        solver_method=SOLVER_METHOD,
        solver_version=str(scipy.__version__),
        optimality_residual=optimality_residual,
    )


def _uniform_solution(
    sources: Sequence[str],
    *,
    uniform_mmd: float,
    uniform_objective: float,
    reason: str,
    solver_success: bool = False,
    solver_message: str = "",
    solver_iterations: int = 0,
) -> KMMWeightSolution:
    source_order = tuple(str(value) for value in sources)
    uniform = np.full(len(source_order), 1.0 / float(len(source_order)))
    return KMMWeightSolution(
        candidate_sources=source_order,
        uniform_weights=weight_mapping(source_order, uniform),
        weights=weight_mapping(source_order, uniform),
        delta=weight_mapping(source_order, np.zeros_like(uniform)),
        proxy_objective=float(uniform_objective),
        uniform_proxy_objective=float(uniform_objective),
        proxy_improvement=0.0,
        mmd_squared=float(uniform_mmd),
        uniform_mmd_squared=float(uniform_mmd),
        regularization_value=0.0,
        effective_source_count=float(len(source_order)),
        maximum_source_weight=float(uniform.max()),
        used_uniform_fallback=True,
        fallback_reason=reason,
        solver_success=bool(solver_success),
        solver_message=str(solver_message),
        solver_iterations=int(solver_iterations),
        solver_method=SOLVER_METHOD,
        solver_version=str(scipy.__version__),
        optimality_residual=None,
    )


def _kkt_stationarity_residual(
    weights: np.ndarray,
    gradient: np.ndarray,
    *,
    cap: float,
    concentration_limit: float,
    solver_tolerance: float,
) -> float:
    """Return a normalized KKT stationarity residual for the convex feasible set.

    SLSQP is permitted here only for the continuous, deterministic convex proxy;
    it is not used to optimize discrete allocations or observed BACC.  This
    independent active-constraint multiplier fit fails closed when the returned
    point is not first-order optimal.
    """

    active_tolerance = max(1e-7, 1000.0 * float(solver_tolerance))
    columns: list[np.ndarray] = [np.ones_like(weights)]
    lower_bounds: list[float] = [float("-inf")]
    upper_bounds: list[float] = [float("inf")]
    for index, weight in enumerate(weights):
        if float(weight) <= active_tolerance:
            column = np.zeros_like(weights)
            column[index] = -1.0
            columns.append(column)
            lower_bounds.append(0.0)
            upper_bounds.append(float("inf"))
        if float(cap - weight) <= active_tolerance:
            column = np.zeros_like(weights)
            column[index] = 1.0
            columns.append(column)
            lower_bounds.append(0.0)
            upper_bounds.append(float("inf"))
    if float(concentration_limit - np.dot(weights, weights)) <= active_tolerance:
        columns.append(2.0 * weights)
        lower_bounds.append(0.0)
        upper_bounds.append(float("inf"))
    design = np.column_stack(columns)
    multipliers = lsq_linear(
        design,
        -np.asarray(gradient, dtype=np.float64),
        bounds=(np.asarray(lower_bounds), np.asarray(upper_bounds)),
        method="bvls",
        tol=max(1e-12, float(solver_tolerance)),
        max_iter=1000,
    )
    if not bool(multipliers.success) or not np.isfinite(multipliers.x).all():
        raise ProtocolError("MMD/KMM KKT multiplier fit failed.")
    stationarity = gradient + design @ multipliers.x
    scale = max(1.0, float(np.linalg.norm(gradient, ord=np.inf)))
    residual = float(np.linalg.norm(stationarity, ord=np.inf) / scale)
    if not np.isfinite(residual):
        raise ProtocolError("MMD/KMM KKT residual is non-finite.")
    return residual


__all__ = ("SOLVER_METHOD", "solve_kmm_weights")
