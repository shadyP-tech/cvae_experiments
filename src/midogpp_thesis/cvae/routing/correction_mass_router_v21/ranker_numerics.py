"""Weighted numerical solvers for the donor ranker."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SupportActionOutcome,
    SupportCaseClassProfile,
    canonical_text,
)
from .hashing import canonical_hash

_DIRECTIONS = (Direction.D01, Direction.D10)
_SOLVER_ITERATIONS = 64
_SOLVER_TOLERANCE = 1.0e-10


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _solve_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> np.ndarray:
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    normal = matrix.T @ (weights[:, None] * matrix) + penalty
    right = matrix.T @ (weights * response)
    try:
        coefficients = np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, right, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise ProtocolError("HARP v21 ridge fit produced non-finite coefficients.")
    return coefficients


def _solve_logistic_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> np.ndarray:
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    prevalence = float(np.dot(weights, response) / np.sum(weights, dtype=np.float64))
    prevalence = min(max(prevalence, 1.0e-6), 1.0 - 1.0e-6)
    if not penalize_intercept:
        coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    for _ in range(_SOLVER_ITERATIONS):
        probability = np.clip(_sigmoid(matrix @ coefficients), 1.0e-6, 1.0 - 1.0e-6)
        curvature = weights * probability * (1.0 - probability)
        gradient = matrix.T @ (weights * (response - probability)) - penalty @ coefficients
        hessian = matrix.T @ (curvature[:, None] * matrix) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients += step
        if float(np.max(np.abs(step))) <= _SOLVER_TOLERANCE:
            break
    if not np.isfinite(coefficients).all():
        raise ProtocolError("HARP v21 logistic ridge produced non-finite coefficients.")
    return coefficients


def _case_weights(keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], float]:
    unique = tuple(sorted(set(keys)))
    centers = tuple(sorted({center for center, _ in unique}))
    cases_by_center = Counter(center for center, _ in unique)
    if not centers or any(cases_by_center[center] < 1 for center in centers):
        raise ProtocolError("HARP v21 equal-center case weights are undefined.")
    return {
        key: 1.0 / (len(centers) * cases_by_center[key[0]])
        for key in unique
    }
