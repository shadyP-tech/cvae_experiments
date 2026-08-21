"""Endpoint-state fitting and donor-prior rebinding kernels."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DIRECTION_IDS,
    ENDPOINT_IRLS_RIDGE_ALPHA,
    HELD_FEATURE_NAMES,
    K_GRID,
    W_GRID,
    candidate_sources,
)
from .endpoint_preparation import CenterCaseOutcomes, PreparedCenter
from .hashing import canonical_hash

@dataclass(frozen=True)
class EndpointState:
    target_center: str
    support_case_ids: tuple[str, ...]
    model_mean: np.ndarray
    model_scale: np.ndarray
    model_coefficients: np.ndarray
    model_valid: np.ndarray
    robust_sources: tuple[tuple[str | None, str | None], ...]
    allowed_sources: tuple[str, ...]
    support_n_positive: int
    support_n_negative: int
    support_gains: Mapping[tuple[str, str], float]
    donor_priors: Mapping[tuple[str, str], float]
    state_hash: str
    model_fit_count: int = 16

    def to_payload(self) -> dict[str, object]:
        return {
            **_endpoint_state_payload(
                target_center=self.target_center,
                support_case_ids=self.support_case_ids,
                model_mean=np.asarray(self.model_mean, dtype=np.float64),
                model_scale=np.asarray(self.model_scale, dtype=np.float64),
                model_coefficients=np.asarray(
                    self.model_coefficients, dtype=np.float64
                ),
                model_valid=np.asarray(self.model_valid, dtype=bool),
                robust_sources=self.robust_sources,
                allowed_sources=self.allowed_sources,
                support_n_positive=self.support_n_positive,
                support_n_negative=self.support_n_negative,
                support_gains=self.support_gains,
                donor_priors=self.donor_priors,
            ),
            "state_hash": self.state_hash,
            "model_fit_count": self.model_fit_count,
            "support_labels_used_indirectly": True,
            "target_evaluation_labels_used": False,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EndpointState":
        try:
            support_gains = {
                (str(source), str(direction)): float(value)
                for source, direction, value in payload["support_gains"]  # type: ignore[index]
            }
            donor_priors = {
                (str(source), str(direction)): float(value)
                for source, direction, value in payload["donor_priors"]  # type: ignore[index]
            }
            state = cls(
                str(payload["target_center"]),
                tuple(str(value) for value in payload["support_case_ids"]),  # type: ignore[index]
                np.asarray(payload["model_mean"], dtype=np.float64),
                np.asarray(payload["model_scale"], dtype=np.float64),
                np.asarray(payload["model_coefficients"], dtype=np.float64),
                np.asarray(payload["model_valid"], dtype=np.uint8).astype(bool),
                tuple(
                    (None if row[0] is None else str(row[0]), None if row[1] is None else str(row[1]))
                    for row in payload["robust_sources"]  # type: ignore[index]
                ),
                tuple(str(value) for value in payload["allowed_sources"]),  # type: ignore[index]
                int(payload["support_n_positive"]),
                int(payload["support_n_negative"]),
                MappingProxyType(support_gains),
                MappingProxyType(donor_priors),
                str(payload["state_hash"]),
                int(payload["model_fit_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("CBPUPR endpoint-state payload is malformed.") from exc
        expected = state.to_payload()
        if (
            dict(payload) != expected
            or state.state_hash
            != canonical_hash(
                {
                    key: value
                    for key, value in expected.items()
                    if key
                    not in {
                        "state_hash",
                        "model_fit_count",
                        "support_labels_used_indirectly",
                        "target_evaluation_labels_used",
                    }
                }
            )
        ):
            raise ProtocolError("CBPUPR endpoint-state hash drifted.")
        return state

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            EndpointState,
            (
                self.target_center,
                self.support_case_ids,
                self.model_mean,
                self.model_scale,
                self.model_coefficients,
                self.model_valid,
                self.robust_sources,
                self.allowed_sources,
                self.support_n_positive,
                self.support_n_negative,
                dict(self.support_gains),
                dict(self.donor_priors),
                self.state_hash,
                self.model_fit_count,
            ),
        )
def _fit_irls(raw: np.ndarray, successes: np.ndarray, trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    mean = np.mean(raw, axis=0, dtype=np.float64)
    scale = np.std(raw, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0)
    design = np.column_stack((np.ones(len(raw), dtype=np.float64), (raw - mean) / scale))
    valid = trials > 0
    beta = np.zeros(design.shape[1], dtype=np.float64)
    converged = False
    if np.any(valid):
        x = design[valid]
        y = successes[valid].astype(np.float64)
        n = trials[valid].astype(np.float64)
        penalty = np.diag(
            np.asarray(
                [0.0, *([ENDPOINT_IRLS_RIDGE_ALPHA] * raw.shape[1])],
                dtype=np.float64,
            )
        )
        for _ in range(50):
            eta = np.clip(x @ beta, -30.0, 30.0)
            probability = np.clip(1.0 / (1.0 + np.exp(-eta)), 1.0e-12, 1.0 - 1.0e-12)
            gradient = x.T @ (y - n * probability) - penalty @ beta
            information = x.T @ ((n * probability * (1.0 - probability))[:, None] * x) + penalty
            try:
                update = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(update).all():
                break
            beta += update
            if float(np.max(np.abs(update))) <= 1.0e-12:
                converged = True
                break
    return mean, scale, beta, bool(converged and np.any(valid))


def _predict_irls(mean: np.ndarray, scale: np.ndarray, beta: np.ndarray, valid: bool, held: np.ndarray) -> float | None:
    if not valid:
        return None
    design = np.concatenate((np.ones(1, dtype=np.float64), (held - mean) / scale))
    value = float(1.0 / (1.0 + np.exp(-float(np.clip(design @ beta, -30.0, 30.0)))))
    return float(np.clip(value, 1.0e-12, 1.0 - 1.0e-12)) if math.isfinite(value) else None



def fit_endpoint_state_from_outcomes(
    prepared: PreparedCenter,
    *,
    support_case_ids: Sequence[str],
    outcomes: CenterCaseOutcomes,
    donor_priors: Mapping[tuple[str, str], float],
    excluded_source_centers: Sequence[str] = (),
) -> EndpointState:
    """Fit one state from an exact, already scoped sufficient-stat surface."""

    cases = tuple(sorted(str(value) for value in support_case_ids))
    sources = candidate_sources(prepared.surface.center)
    excluded_sources = {str(value) for value in excluded_source_centers}
    allowed_sources = tuple(source for source in sources if source not in excluded_sources)
    if (
        outcomes.center != prepared.surface.center
        or cases != outcomes.case_ids
        or not allowed_sources
        or not excluded_sources <= set(sources)
        or tuple(donor_priors)
        != tuple(
            (source, direction)
            for source in candidate_sources(prepared.surface.center)
            for direction in DIRECTION_IDS
        )
    ):
        raise ProtocolError("Endpoint sufficient-stat scope or donor priors drifted.")
    mean = np.zeros((len(sources), 2, len(HELD_FEATURE_NAMES)), dtype=np.float64)
    scale = np.ones_like(mean)
    coefficients = np.zeros((len(sources), 2, 1 + len(HELD_FEATURE_NAMES)), dtype=np.float64)
    valid = np.zeros((len(sources), 2), dtype=bool)
    support_gain: dict[tuple[str, str], Fraction] = {}
    n_positive = int(np.sum(outcomes.n_positive, dtype=np.int64))
    n_negative = int(np.sum(outcomes.n_negative, dtype=np.int64))
    if not n_positive or not n_negative:
        raise ProtocolError("Endpoint support state must retain both classes.")
    case_index = {case: prepared.cases.index(case) for case in cases}
    for source_index, source in enumerate(sources):
        for direction_index, direction in enumerate(DIRECTION_IDS):
            if source not in allowed_sources:
                support_gain[(source, direction)] = Fraction(0, 1)
                continue
            successes = outcomes.successes[:, source_index, direction_index]
            trials = outcomes.trials[:, source_index, direction_index]
            fitted = _fit_irls(
                np.asarray(
                    [prepared.feature_values[case_index[case], source_index, direction_index] for case in cases],
                    dtype=np.float64,
                ),
                successes.astype(np.float64, copy=False),
                trials.astype(np.float64, copy=False),
            )
            mean[source_index, direction_index] = fitted[0]
            scale[source_index, direction_index] = fitted[1]
            coefficients[source_index, direction_index] = fitted[2]
            valid[source_index, direction_index] = fitted[3]
            favorable = int(np.sum(successes, dtype=np.int64))
            adverse = int(np.sum(trials, dtype=np.int64)) - favorable
            if direction == "zero_to_one":
                support_gain[(source, direction)] = (
                    Fraction(favorable, 2 * n_positive)
                    - Fraction(adverse, 2 * n_negative)
                )
            else:
                support_gain[(source, direction)] = (
                    Fraction(favorable, 2 * n_negative)
                    - Fraction(adverse, 2 * n_positive)
                )
    robust_sources = _select_robust_sources(
        allowed_sources,
        support_gain,
        donor_priors,
    )
    support_gain_values = MappingProxyType(
        {key: float(value) for key, value in support_gain.items()}
    )
    payload = _endpoint_state_payload(
        target_center=prepared.surface.center,
        support_case_ids=cases,
        model_mean=mean,
        model_scale=scale,
        model_coefficients=coefficients,
        model_valid=valid,
        robust_sources=robust_sources,
        allowed_sources=allowed_sources,
        support_n_positive=n_positive,
        support_n_negative=n_negative,
        support_gains=support_gain_values,
        donor_priors=donor_priors,
    )
    return EndpointState(
        prepared.surface.center,
        cases,
        mean,
        scale,
        coefficients,
        valid,
        robust_sources,
        allowed_sources,
        n_positive,
        n_negative,
        support_gain_values,
        MappingProxyType(dict(donor_priors)),
        canonical_hash(payload),
        2 * len(allowed_sources),
    )


def _select_robust_sources(
    allowed_sources: Sequence[str],
    support_gain: Mapping[tuple[str, str], object],
    donor_priors: Mapping[tuple[str, str], float],
) -> tuple[tuple[str | None, str | None], ...]:
    robust_sources: list[tuple[str | None, str | None]] = []
    for k in K_GRID:
        for weight in W_GRID:
            selections: list[str | None] = []
            for direction in DIRECTION_IDS:
                ranked = tuple(
                    sorted(
                        allowed_sources,
                        key=lambda source: (-donor_priors[(source, direction)], int(source)),
                    )
                )[:k]
                scores: dict[str | None, float] = {None: 0.0}
                for source in ranked:
                    scores[source] = (
                        weight * float(support_gain[(source, direction)])
                        + (1.0 - weight) * float(donor_priors[(source, direction)])
                    )
                maximum = max(scores.values())
                selected = min(
                    (source for source, value in scores.items() if maximum - value <= 1.0e-12),
                    key=lambda source: -1 if source is None else int(source),
                )
                selections.append(selected)
            robust_sources.append((selections[0], selections[1]))
    return tuple(robust_sources)


def _endpoint_state_payload(
    *,
    target_center: str,
    support_case_ids: Sequence[str],
    model_mean: np.ndarray,
    model_scale: np.ndarray,
    model_coefficients: np.ndarray,
    model_valid: np.ndarray,
    robust_sources: Sequence[tuple[str | None, str | None]],
    allowed_sources: Sequence[str],
    support_n_positive: int,
    support_n_negative: int,
    support_gains: Mapping[tuple[str, str], float],
    donor_priors: Mapping[tuple[str, str], float],
) -> dict[str, object]:
    sources = candidate_sources(target_center)
    return {
        "schema_version": "fixed_bank_cbpupr_endpoint_state_v1",
        "target_center": target_center,
        "support_case_ids": list(support_case_ids),
        "model_mean": model_mean.tolist(),
        "model_scale": model_scale.tolist(),
        "model_coefficients": model_coefficients.tolist(),
        "model_valid": model_valid.astype(np.uint8).tolist(),
        "robust_sources": [list(row) for row in robust_sources],
        "allowed_sources": list(allowed_sources),
        "excluded_source_centers": [
            source for source in sources if source not in set(allowed_sources)
        ],
        "support_n_positive": support_n_positive,
        "support_n_negative": support_n_negative,
        "support_gains": [
            [source, direction, support_gains[(source, direction)]]
            for source in sources
            for direction in DIRECTION_IDS
        ],
        "donor_priors": [
            [source, direction, donor_priors[(source, direction)]]
            for source in sources
            for direction in DIRECTION_IDS
        ],
        "raw_labels_persisted": False,
    }


def rebind_endpoint_state_priors(
    state: EndpointState,
    donor_priors: Mapping[tuple[str, str], float],
    *,
    excluded_source_centers: Sequence[str] = (),
) -> EndpointState:
    """Recompose only the cheap prior-dependent part of a fitted state."""

    sources = candidate_sources(state.target_center)
    excluded_sources = {str(value) for value in excluded_source_centers}
    allowed_sources = tuple(source for source in sources if source not in excluded_sources)
    expected = tuple(
        (source, direction) for source in sources for direction in DIRECTION_IDS
    )
    if (
        tuple(donor_priors) != expected
        or not allowed_sources
        or not excluded_sources <= set(sources)
    ):
        raise ProtocolError("Rebound donor prior surface drifted.")
    mean = np.array(state.model_mean, dtype=np.float64, copy=True)
    scale = np.array(state.model_scale, dtype=np.float64, copy=True)
    coefficients = np.array(state.model_coefficients, dtype=np.float64, copy=True)
    valid = np.array(state.model_valid, dtype=bool, copy=True)
    support_gains = dict(state.support_gains)
    sanitized_priors = dict(donor_priors)
    for source in excluded_sources:
        source_index = sources.index(source)
        mean[source_index] = 0.0
        scale[source_index] = 1.0
        coefficients[source_index] = 0.0
        valid[source_index] = False
        for direction in DIRECTION_IDS:
            support_gains[(source, direction)] = 0.0
            sanitized_priors[(source, direction)] = 0.0
    robust_sources = _select_robust_sources(
        allowed_sources, support_gains, sanitized_priors
    )
    payload = _endpoint_state_payload(
        target_center=state.target_center,
        support_case_ids=state.support_case_ids,
        model_mean=mean,
        model_scale=scale,
        model_coefficients=coefficients,
        model_valid=valid,
        robust_sources=robust_sources,
        allowed_sources=allowed_sources,
        support_n_positive=state.support_n_positive,
        support_n_negative=state.support_n_negative,
        support_gains=support_gains,
        donor_priors=sanitized_priors,
    )
    return EndpointState(
        state.target_center,
        state.support_case_ids,
        mean,
        scale,
        coefficients,
        valid,
        robust_sources,
        allowed_sources,
        state.support_n_positive,
        state.support_n_negative,
        MappingProxyType(support_gains),
        MappingProxyType(sanitized_priors),
        canonical_hash(payload),
        0,
    )



__all__ = (
    "EndpointState",
    "fit_endpoint_state_from_outcomes",
    "rebind_endpoint_state_priors",
)
