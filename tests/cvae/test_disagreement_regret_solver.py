from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.disagreement_regret_core import _solver
from midogpp_thesis.cvae.routing.disagreement_regret_core.design import (
    PairwiseTrainingDesign,
)


def _ill_scaled_design() -> PairwiseTrainingDesign:
    """Return a fixed design whose final improvement is below binary64 resolution."""

    rng = np.random.default_rng(2)
    row_count = 500
    dimension = 80
    values = rng.normal(size=(row_count, dimension))
    values[:, 0] *= 10.0 ** rng.uniform(0.0, 5.0)
    outcomes = rng.integers(0, 2, row_count).astype(np.float64)
    weights = rng.random(row_count)
    weights /= weights.sum(dtype=np.float64)
    penalty = np.concatenate(
        (
            np.ones(dimension // 2, dtype=np.float64),
            np.full(dimension - dimension // 2, 4.0, dtype=np.float64),
        )
    )
    query_ids = tuple(f"q{index % 4}" for index in range(row_count))
    return PairwiseTrainingDesign(
        encoder=SimpleNamespace(
            dimension=dimension,
            penalty_diagonal=penalty,
        ),
        values=values,
        outcomes=outcomes,
        weights=weights,
        query_ids=query_ids,
        informative_query_ids=("q0", "q1", "q2", "q3"),
    )


def test_solver_treats_unresolvable_objective_decrease_as_converged() -> None:
    design = _ill_scaled_design()

    coefficients, covariance, iteration_count = _solver.solve_pairwise_logit(design)

    probability = _solver._sigmoid(design.values @ coefficients)
    gradient = (
        design.values.T @ (design.weights * (probability - design.outcomes))
        + design.encoder.penalty_diagonal * coefficients
    )
    curvature = design.weights * probability * (1.0 - probability)
    hessian = (
        design.values.T @ (curvature[:, None] * design.values)
        + np.diag(design.encoder.penalty_diagonal)
    )
    step = np.linalg.solve(hessian, gradient)
    objective = _solver._objective(
        coefficients,
        design.values,
        design.outcomes,
        design.weights,
        design.encoder.penalty_diagonal,
    )
    decrement = float(np.dot(gradient, step))
    resolution = (
        _solver.OBJECTIVE_RESOLUTION_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, abs(objective))
    )

    assert iteration_count == 3
    assert float(np.linalg.norm(gradient, ord=np.inf)) > _solver.GRADIENT_TOLERANCE
    assert float(np.linalg.norm(step, ord=np.inf)) > _solver.STEP_TOLERANCE
    assert 0.0 < decrement <= resolution
    assert np.isfinite(coefficients).all()
    assert np.isfinite(covariance).all()


def test_objective_resolution_fallback_does_not_mask_large_failed_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _ill_scaled_design()
    monkeypatch.setattr(_solver, "_objective", lambda *_arguments: 1.0)

    with pytest.raises(ProtocolError, match="backtracking failed"):
        _solver.solve_pairwise_logit(design)
