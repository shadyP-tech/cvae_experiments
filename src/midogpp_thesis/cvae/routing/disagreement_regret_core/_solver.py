"""Deterministic fixed-hyperparameter pairwise logistic solver."""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError
from .design import PairwiseTrainingDesign


MAX_NEWTON_ITERATIONS = 100
GRADIENT_TOLERANCE = 1.0e-8
STEP_TOLERANCE = 1.0e-10
BACKTRACK_STEPS = 24


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _objective(
    coefficients: np.ndarray,
    values: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    penalty: np.ndarray,
) -> float:
    logits = values @ coefficients
    data_loss = float(
        np.sum(weights * (np.logaddexp(0.0, logits) - outcomes * logits), dtype=np.float64)
    )
    regularization = 0.5 * float(np.dot(penalty, coefficients * coefficients))
    return data_loss + regularization


def _project_psd(covariance: np.ndarray) -> np.ndarray:
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ProtocolError("Pairwise covariance must be square.")
    symmetric = 0.5 * (covariance + covariance.T)
    if not np.isfinite(symmetric).all():
        raise ProtocolError("Pairwise covariance must be finite.")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -1.0e-9 * scale:
        raise ProtocolError("Pairwise covariance is not positive semidefinite.")
    result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    result = 0.5 * (result + result.T)
    result.setflags(write=False)
    return result


def solve_pairwise_logit(
    design: PairwiseTrainingDesign,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit coefficients and a query-clustered covariance for a fixed design."""

    values = design.values
    outcomes = design.outcomes
    weights = design.weights
    penalty = design.encoder.penalty_diagonal
    coefficients = np.zeros(design.encoder.dimension, dtype=np.float64)
    converged = False
    iteration_count = 0
    for iteration in range(1, MAX_NEWTON_ITERATIONS + 1):
        iteration_count = iteration
        probability = _sigmoid(values @ coefficients)
        residual = weights * (probability - outcomes)
        gradient = values.T @ residual + penalty * coefficients
        curvature = weights * probability * (1.0 - probability)
        hessian = values.T @ (curvature[:, None] * values) + np.diag(penalty)
        if float(np.linalg.norm(gradient, ord=np.inf)) <= GRADIENT_TOLERANCE:
            converged = True
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ProtocolError("Pairwise Newton system is singular.") from exc
        current = _objective(coefficients, values, outcomes, weights, penalty)
        accepted = False
        scale = 1.0
        for _ in range(BACKTRACK_STEPS):
            proposal = coefficients - scale * step
            if _objective(proposal, values, outcomes, weights, penalty) < current:
                coefficients = proposal
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            if float(np.linalg.norm(step, ord=np.inf)) <= STEP_TOLERANCE:
                converged = True
                break
            raise ProtocolError("Pairwise Newton backtracking failed to improve the objective.")
        if float(np.linalg.norm(scale * step, ord=np.inf)) <= STEP_TOLERANCE:
            converged = True
            break
    if not converged:
        raise ProtocolError("Pairwise hierarchical regret fit did not converge.")

    probability = _sigmoid(values @ coefficients)
    curvature = weights * probability * (1.0 - probability)
    hessian = values.T @ (curvature[:, None] * values) + np.diag(penalty)
    try:
        bread = np.linalg.solve(hessian, np.eye(hessian.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("Pairwise covariance system is singular.") from exc
    meat = np.zeros_like(hessian)
    for query in design.informative_query_ids:
        mask = np.fromiter((value == query for value in design.query_ids), dtype=bool)
        score = values[mask].T @ (
            weights[mask] * (outcomes[mask] - probability[mask])
        )
        meat += np.outer(score, score)
    cluster_count = len(design.informative_query_ids)
    covariance = _project_psd(
        (cluster_count / (cluster_count - 1.0)) * (bread @ meat @ bread.T)
    )
    coefficients.setflags(write=False)
    return coefficients, covariance, iteration_count


__all__ = (
    "BACKTRACK_STEPS",
    "GRADIENT_TOLERANCE",
    "MAX_NEWTON_ITERATIONS",
    "STEP_TOLERANCE",
    "solve_pairwise_logit",
)
