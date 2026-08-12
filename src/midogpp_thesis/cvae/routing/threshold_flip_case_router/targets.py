"""Exact-nine thresholding and additive TP/TN contribution targets."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import ContributionTarget


HARD_THRESHOLD = 0.5
EXACT_SEED_COUNT = 9


def exact_nine_mean_probabilities(
    seed_probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Average the exact nine seed-pair probabilities before thresholding."""

    matrix = np.asarray(seed_probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != EXACT_SEED_COUNT or matrix.shape[0] == 0:
        raise ProtocolError("Expected a non-empty sample-by-exact-nine probability matrix.")
    if not np.isfinite(matrix).all() or np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ProtocolError("Seed probabilities must be finite and lie in [0,1].")
    result = np.mean(matrix, axis=1, dtype=np.float64)
    result.setflags(write=False)
    return result


def hard_predictions(
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = HARD_THRESHOLD,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ProtocolError("Hard predictions require a finite non-empty vector.")
    if float(threshold) != HARD_THRESHOLD:
        raise ProtocolError("The threshold-flip router freezes the threshold at 0.5.")
    result = values >= HARD_THRESHOLD
    result.setflags(write=False)
    return result


def contribution_target(
    *,
    case_id: str,
    action_id: str,
    baseline_probabilities: Sequence[float] | np.ndarray,
    action_probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> ContributionTarget:
    """Compute additive confusion-count deltas; single-class cases are retained."""

    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    action = np.asarray(action_probabilities, dtype=np.float64)
    truth = np.asarray(labels)
    if baseline.ndim != 1 or action.shape != baseline.shape or truth.shape != baseline.shape:
        raise ProtocolError("Contribution inputs must be aligned one-dimensional vectors.")
    if len(baseline) == 0 or not np.isfinite(baseline).all() or not np.isfinite(action).all():
        raise ProtocolError("Contribution probability vectors must be finite and non-empty.")
    if np.any((baseline < 0.0) | (baseline > 1.0) | (action < 0.0) | (action > 1.0)):
        raise ProtocolError("Contribution probabilities lie outside [0,1].")
    if not np.all(np.isin(truth, (0, 1))):
        raise ProtocolError("Contribution labels must be binary.")
    truth = truth.astype(np.int8, copy=False)
    b = baseline >= HARD_THRESHOLD
    a = action >= HARD_THRESHOLD
    positive = truth == 1
    negative = ~positive
    return ContributionTarget(
        case_id=str(case_id),
        action_id=str(action_id),
        delta_tp=int(np.sum(a & positive) - np.sum(b & positive)),
        delta_tn=int(np.sum((~a) & negative) - np.sum((~b) & negative)),
        n_positive=int(np.sum(positive)),
        n_negative=int(np.sum(negative)),
    )


def pooled_gain(targets: Sequence[ContributionTarget]) -> float:
    """Return exact pooled BACC change represented by additive case targets."""

    if not targets:
        raise ProtocolError("Pooled gain requires at least one case target.")
    n_positive = sum(row.n_positive for row in targets)
    n_negative = sum(row.n_negative for row in targets)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Pooled BACC gain requires both classes across the pool.")
    return float(
        0.5 * sum(row.delta_tp for row in targets) / n_positive
        + 0.5 * sum(row.delta_tn for row in targets) / n_negative
    )


__all__ = (
    "EXACT_SEED_COUNT",
    "HARD_THRESHOLD",
    "contribution_target",
    "exact_nine_mean_probabilities",
    "hard_predictions",
    "pooled_gain",
)
