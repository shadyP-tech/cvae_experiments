"""Unified posterior expected utility for P-anchored physical actions.

All three coordinates use the favourable-gain convention: positive means the
candidate is expected to improve over P.  Target evaluation labels never enter
this module; ``eta`` is an H/c-excluded posterior probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import (
    canonical_float32_probabilities,
    canonical_hash,
    require_sha256,
)
from .constants import LOG_LOSS_CLIP_EPSILON


UTILITY_COORDINATES = ("bacc_gain", "brier_gain", "log_gain")
LOG_CLIP_EPSILON = LOG_LOSS_CLIP_EPSILON


@dataclass(frozen=True)
class FavorableUtility:
    """Three-coordinate utility vector where larger is always better."""

    bacc_gain: float
    brier_gain: float
    log_gain: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("CBPUPR utility coordinates must be finite.")

    @classmethod
    def zeros(cls) -> "FavorableUtility":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | Sequence[float]) -> "FavorableUtility":
        if isinstance(payload, Mapping):
            return cls(
                float(payload["bacc_gain"]),
                float(payload["brier_gain"]),
                float(payload["log_gain"]),
            )
        values = tuple(float(value) for value in payload)
        if len(values) != 3:
            raise ProtocolError("CBPUPR utility payload must have three coordinates.")
        return cls(*values)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc_gain, self.brier_gain, self.log_gain

    def to_payload(self) -> dict[str, float]:
        return dict(zip(UTILITY_COORDINATES, self.as_tuple(), strict=True))

    def __add__(self, other: "FavorableUtility") -> "FavorableUtility":
        return FavorableUtility(
            self.bacc_gain + other.bacc_gain,
            self.brier_gain + other.brier_gain,
            self.log_gain + other.log_gain,
        )

    def __sub__(self, other: "FavorableUtility") -> "FavorableUtility":
        return FavorableUtility(
            self.bacc_gain - other.bacc_gain,
            self.brier_gain - other.brier_gain,
            self.log_gain - other.log_gain,
        )


@dataclass(frozen=True)
class PosteriorUtilityEstimate:
    center: str
    case_id: str
    action_id: str
    direction: str
    control_id: str
    crossing_count: int
    fold_utilities: tuple[FavorableUtility, ...]
    utility: FavorableUtility
    posterior_hash: str
    estimate_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not self.center
            or not self.case_id
            or not self.action_id
            or self.direction not in ("zero_to_one", "one_to_zero")
            or not self.control_id
            or self.crossing_count <= 0
            or not self.fold_utilities
        ):
            raise ProtocolError("CBPUPR posterior utility identity drifted.")
        require_sha256(self.posterior_hash, "posterior_hash")
        payload = {
            "schema_version": "cbpupr_posterior_utility_estimate_v1",
            "center": self.center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "direction": self.direction,
            "control_id": self.control_id,
            "crossing_count": self.crossing_count,
            "fold_utilities": [row.to_payload() for row in self.fold_utilities],
            "utility": self.utility.to_payload(),
            "posterior_hash": self.posterior_hash,
        }
        object.__setattr__(self, "estimate_hash", canonical_hash(payload))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PosteriorUtilityEstimate":
        row = cls(
            center=str(payload["center"]),
            case_id=str(payload["case_id"]),
            action_id=str(payload["action_id"]),
            direction=str(payload["direction"]),
            control_id=str(payload["control_id"]),
            crossing_count=int(payload["crossing_count"]),
            fold_utilities=tuple(
                FavorableUtility.from_payload(value)
                for value in payload["fold_utilities"]  # type: ignore[index]
            ),
            utility=FavorableUtility.from_payload(payload["utility"]),  # type: ignore[arg-type]
            posterior_hash=str(payload["posterior_hash"]),
        )
        if "estimate_hash" in payload and str(payload["estimate_hash"]) != row.estimate_hash:
            raise ProtocolError("CBPUPR posterior utility estimate hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "direction": self.direction,
            "control_id": self.control_id,
            "crossing_count": self.crossing_count,
            "fold_utilities": [row.to_payload() for row in self.fold_utilities],
            "utility": self.utility.to_payload(),
            "posterior_hash": self.posterior_hash,
            "estimate_hash": self.estimate_hash,
        }


def compute_expected_utility(
    portfolio_probabilities: object,
    candidate_probabilities: object,
    posterior_eta: object,
    *,
    support_n_positive: float = 0.0,
    support_n_negative: float = 0.0,
    support_row_count: int = 0,
    crossing_mask: object | None = None,
    log_clip_epsilon: float = LOG_CLIP_EPSILON,
) -> FavorableUtility:
    """Compute exact posterior expected BACC, Brier and log-loss gains.

    BACC uses posterior-augmented positive and negative denominators.  Brier and
    log-loss gains are normalised by the posterior-augmented total row count.
    """

    p = canonical_float32_probabilities(portfolio_probabilities)
    a = canonical_float32_probabilities(candidate_probabilities, expected_length=len(p))
    eta = np.asarray(posterior_eta, dtype=np.float64)
    if (
        eta.shape != p.shape
        or not np.isfinite(eta).all()
        or bool(np.any((eta < 0.0) | (eta > 1.0)))
        or not math.isfinite(float(support_n_positive))
        or not math.isfinite(float(support_n_negative))
        or float(support_n_positive) < 0.0
        or float(support_n_negative) < 0.0
        or int(support_row_count) < 0
    ):
        raise ProtocolError("CBPUPR posterior utility inputs drifted.")
    if abs(
        float(support_n_positive) + float(support_n_negative) - int(support_row_count)
    ) > 1.0e-8:
        raise ProtocolError("CBPUPR support posterior counts are inconsistent.")

    if crossing_mask is None:
        mask = (p >= np.float32(0.5)) != (a >= np.float32(0.5))
    else:
        mask = np.asarray(crossing_mask, dtype=bool)
        if mask.shape != p.shape:
            raise ProtocolError("CBPUPR crossing mask shape drifted.")
        actual = (p >= np.float32(0.5)) != (a >= np.float32(0.5))
        if not np.array_equal(mask, actual):
            raise ProtocolError("CBPUPR crossing mask does not match threshold changes.")
    if not bool(np.any(mask)):
        raise ProtocolError("CBPUPR expected utility requires a threshold crossing.")

    n_positive = float(support_n_positive) + float(np.sum(eta, dtype=np.float64))
    n_negative = float(support_n_negative) + float(
        np.sum(1.0 - eta, dtype=np.float64)
    )
    n_total = int(support_row_count) + len(eta)
    if min(n_positive, n_negative) <= 0.0 or n_total <= 0:
        raise ProtocolError("CBPUPR posterior-augmented denominator is empty.")

    old_prediction = (p >= np.float32(0.5)).astype(np.float64)
    new_prediction = (a >= np.float32(0.5)).astype(np.float64)
    prediction_delta = new_prediction - old_prediction
    bacc_gain = 0.5 * float(
        np.sum(
            prediction_delta[mask]
            * (
                eta[mask] / n_positive
                - (1.0 - eta[mask]) / n_negative
            ),
            dtype=np.float64,
        )
    )

    p64 = p.astype(np.float64, copy=False)[mask]
    a64 = a.astype(np.float64, copy=False)[mask]
    eta_masked = eta[mask]
    brier_gain = float(
        np.sum(
            p64 * p64 - a64 * a64 - 2.0 * eta_masked * (p64 - a64),
            dtype=np.float64,
        )
        / n_total
    )
    epsilon = float(log_clip_epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ProtocolError("CBPUPR log-loss clipping epsilon drifted.")
    p_clip = np.clip(p64, epsilon, 1.0 - epsilon)
    a_clip = np.clip(a64, epsilon, 1.0 - epsilon)
    log_gain = float(
        np.sum(
            eta_masked * np.log(a_clip / p_clip)
            + (1.0 - eta_masked)
            * np.log((1.0 - a_clip) / (1.0 - p_clip)),
            dtype=np.float64,
        )
        / n_total
    )
    return FavorableUtility(bacc_gain, brier_gain, log_gain)


def aggregate_fold_utilities(
    utilities: Sequence[FavorableUtility],
) -> FavorableUtility:
    """Use a deterministic componentwise posterior-fold median."""

    rows = tuple(utilities)
    if not rows:
        raise ProtocolError("CBPUPR posterior fold rectangle is empty.")
    matrix = np.asarray([row.as_tuple() for row in rows], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise ProtocolError("CBPUPR posterior fold rectangle drifted.")
    values = np.median(matrix, axis=0)
    return FavorableUtility(*(float(value) for value in values))


def score_posterior_folds(
    *,
    center: str,
    case_id: str,
    action_id: str,
    direction: str,
    control_id: str,
    portfolio_probabilities: object,
    candidate_probabilities: object,
    posterior_folds: Sequence[object],
    posterior_hash: str,
    support_n_positive: float = 0.0,
    support_n_negative: float = 0.0,
    support_row_count: int = 0,
) -> PosteriorUtilityEstimate:
    p = canonical_float32_probabilities(portfolio_probabilities)
    a = canonical_float32_probabilities(candidate_probabilities, expected_length=len(p))
    mask = (p >= np.float32(0.5)) != (a >= np.float32(0.5))
    if direction == "zero_to_one":
        valid_direction = bool(np.all((a[mask] >= 0.5) & (p[mask] < 0.5)))
    elif direction == "one_to_zero":
        valid_direction = bool(np.all((a[mask] < 0.5) & (p[mask] >= 0.5)))
    else:
        raise ProtocolError("CBPUPR action direction drifted.")
    if not bool(np.any(mask)) or not valid_direction:
        raise ProtocolError("CBPUPR candidate is not a pure directional threshold action.")
    rows = tuple(
        compute_expected_utility(
            p,
            a,
            eta,
            support_n_positive=support_n_positive,
            support_n_negative=support_n_negative,
            support_row_count=support_row_count,
            crossing_mask=mask,
        )
        for eta in posterior_folds
    )
    return PosteriorUtilityEstimate(
        center=str(center),
        case_id=str(case_id),
        action_id=str(action_id),
        direction=direction,
        control_id=str(control_id),
        crossing_count=int(np.count_nonzero(mask)),
        fold_utilities=rows,
        utility=aggregate_fold_utilities(rows),
        posterior_hash=posterior_hash,
    )


__all__ = (
    "FavorableUtility",
    "LOG_CLIP_EPSILON",
    "PosteriorUtilityEstimate",
    "UTILITY_COORDINATES",
    "aggregate_fold_utilities",
    "compute_expected_utility",
    "score_posterior_folds",
)
