"""Bounded logit-residual primitives shared by feature and composition paths."""

from __future__ import annotations

import math

from ...protocol import ProtocolError
from .core_hashing import finite_float
from .scientific_constants import PROBABILITY_EPSILON


def clipped_probability(value: float, *, epsilon: float = PROBABILITY_EPSILON) -> float:
    probability = finite_float(value, "probability")
    eps = finite_float(epsilon, "epsilon")
    if not 0.0 < eps < 0.5:
        raise ProtocolError("Probability epsilon must lie strictly inside (0, 0.5).")
    if not 0.0 <= probability <= 1.0:
        raise ProtocolError("Probability must lie in [0, 1].")
    return min(max(probability, eps), 1.0 - eps)


def logit_clip(value: float, *, epsilon: float = PROBABILITY_EPSILON) -> float:
    probability = clipped_probability(value, epsilon=epsilon)
    return math.log(probability) - math.log1p(-probability)


def sigmoid(value: float) -> float:
    number = finite_float(value, "logit")
    if number >= 0.0:
        inverse = math.exp(-number)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(number)
    return exponent / (1.0 + exponent)


def residual_logit(
    candidate_probability: float,
    baseline_probability: float,
    *,
    epsilon: float = PROBABILITY_EPSILON,
) -> float:
    return logit_clip(candidate_probability, epsilon=epsilon) - logit_clip(
        baseline_probability, epsilon=epsilon
    )


__all__ = ("clipped_probability", "logit_clip", "residual_logit", "sigmoid")
