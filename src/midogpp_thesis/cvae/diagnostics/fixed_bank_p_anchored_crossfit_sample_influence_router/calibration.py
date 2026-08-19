"""Fixed sign-preserving probability shrinkage for directional actions."""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    DIRECTION_IDS,
    HARD_THRESHOLD,
    PORTFOLIO_METHOD_ID,
    SIGN_PRESERVING_SHRINKAGE,
)
from .contracts import EndpointCasePrediction


def sign_preserving_shrink(values: np.ndarray) -> np.ndarray:
    """Temper confidence without changing any non-tied hard prediction."""

    probabilities = np.asarray(values, dtype=np.float64)
    output = HARD_THRESHOLD + SIGN_PRESERVING_SHRINKAGE * (
        probabilities - HARD_THRESHOLD
    )
    if (
        not np.isfinite(output).all()
        or np.any((output < 0.0) | (output > 1.0))
        or np.any((probabilities >= HARD_THRESHOLD) != (output >= HARD_THRESHOLD))
    ):
        raise ProtocolError("PCSI sign-preserving shrinkage drifted.")
    return output


def directional_candidate(
    endpoint: EndpointCasePrediction,
    alternative: str,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P except on one directional P-vs-alternative crossing branch."""

    if direction not in DIRECTION_IDS or alternative not in endpoint.probabilities:
        raise ProtocolError("PCSI directional candidate identity drifted.")
    portfolio = np.asarray(
        endpoint.probabilities[PORTFOLIO_METHOD_ID], dtype=np.float64
    )
    candidate = np.asarray(endpoint.probabilities[alternative], dtype=np.float64)
    p_hard = portfolio >= HARD_THRESHOLD
    a_hard = candidate >= HARD_THRESHOLD
    if direction == "zero_to_one":
        mask = (~p_hard) & a_hard
    else:
        mask = p_hard & (~a_hard)
    output = portfolio.copy()
    output[mask] = sign_preserving_shrink(candidate[mask])
    if np.any((output >= HARD_THRESHOLD)[mask] != a_hard[mask]):
        raise ProtocolError("PCSI calibrated action changed its selected hard class.")
    return output, mask


__all__ = ("directional_candidate", "sign_preserving_shrink")
