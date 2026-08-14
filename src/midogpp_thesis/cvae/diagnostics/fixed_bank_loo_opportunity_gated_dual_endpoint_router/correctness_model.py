"""Frozen route-local ridge-binomial IRLS and support-calibrated proxy."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    DIRECTION_IDS,
    FEATURE_NAMES,
    IRLS_CONVERGENCE_TOLERANCE,
    IRLS_ETA_CLIP,
    IRLS_MAX_ITERATIONS,
    IRLS_PROBABILITY_CLIP,
    RIDGE_ALPHA,
    candidate_sources,
)
from .correctness_products import (
    DirectionalCorrectnessModel,
    DirectionalCorrectnessObservation,
    LabelFreeDirectionalFeatures,
    SupportClassDenominators,
)
from .split_plans import WholeCaseLooPlan


def fit_directional_correctness_model(
    observations: Sequence[DirectionalCorrectnessObservation],
    *,
    target_center: object,
    case_id: object,
    source: object,
    direction: object,
) -> DirectionalCorrectnessModel:
    identity = (str(target_center), str(case_id), str(source), str(direction))
    rows = tuple(sorted(observations, key=lambda row: row.support_case_id))
    if (
        not rows
        or any((row.target_center, row.route_case_id, row.source, row.direction) != identity for row in rows)
        or any(row.support_case_id == identity[1] for row in rows)
        or len({row.support_case_id for row in rows}) != len(rows)
    ):
        raise ProtocolError("OGDE correctness model fit scope drifted.")
    raw = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    mean = np.mean(raw, axis=0, dtype=np.float64)
    scale = np.std(raw, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float64, copy=False)
    design = np.column_stack((np.ones(len(rows), dtype=np.float64), (raw - mean) / scale))
    successes = np.asarray([row.successes for row in rows], dtype=np.float64)
    trials = np.asarray([row.trials for row in rows], dtype=np.float64)
    valid = trials > 0.0
    beta = np.zeros(len(FEATURE_NAMES) + 1, dtype=np.float64)
    converged = False
    iterations = 0
    if bool(np.any(valid)):
        x = design[valid]
        y = successes[valid]
        n = trials[valid]
        penalty = np.diag(np.asarray([0.0, *([RIDGE_ALPHA] * len(FEATURE_NAMES))], dtype=np.float64))
        for iterations in range(1, IRLS_MAX_ITERATIONS + 1):
            eta = np.clip(x @ beta, -IRLS_ETA_CLIP, IRLS_ETA_CLIP)
            probability = 1.0 / (1.0 + np.exp(-eta))
            probability = np.clip(probability, IRLS_PROBABILITY_CLIP, 1.0 - IRLS_PROBABILITY_CLIP)
            gradient = x.T @ (y - n * probability) - penalty @ beta
            weights = n * probability * (1.0 - probability)
            information = x.T @ (weights[:, None] * x) + penalty
            try:
                update = np.linalg.solve(information, gradient)
            except np.linalg.LinAlgError:
                break
            if not bool(np.all(np.isfinite(update))):
                break
            beta = beta + update
            if float(np.max(np.abs(update))) <= IRLS_CONVERGENCE_TOLERANCE:
                converged = True
                break
    return DirectionalCorrectnessModel(
        *identity,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in beta),
        tuple(row.support_case_id for row in rows),
        tuple(row.observation_hash for row in rows),
        int(np.sum(trials, dtype=np.float64)),
        int(np.sum(valid, dtype=np.int64)),
        converged,
        iterations,
    )


def fit_route_correctness_models(
    observations: Sequence[DirectionalCorrectnessObservation],
    plan: WholeCaseLooPlan,
) -> tuple[DirectionalCorrectnessModel, ...]:
    rows = tuple(observations)
    output: list[DirectionalCorrectnessModel] = []
    for source in candidate_sources(plan.target_center):
        for direction in DIRECTION_IDS:
            selected = tuple(row for row in rows if row.source == source and row.direction == direction)
            if {row.support_case_id for row in selected} != set(plan.support_case_ids):
                raise ProtocolError("OGDE route model lacks exact H-minus-c observations.")
            output.append(
                fit_directional_correctness_model(
                    selected,
                    target_center=plan.target_center,
                    case_id=plan.case_id,
                    source=source,
                    direction=direction,
                )
            )
    return tuple(output)


def predict_correctness(
    model: DirectionalCorrectnessModel,
    features: LabelFreeDirectionalFeatures,
) -> float | None:
    if model.key != features.key:
        raise ProtocolError("OGDE held feature/model identity drifted.")
    if not model.is_valid:
        return None
    values = np.asarray(features.values, dtype=np.float64)
    mean = np.asarray(model.feature_mean, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    coefficients = np.asarray(model.coefficients, dtype=np.float64)
    design = np.concatenate((np.ones(1, dtype=np.float64), (values - mean) / scale))
    eta = float(np.clip(design @ coefficients, -IRLS_ETA_CLIP, IRLS_ETA_CLIP))
    result = float(1.0 / (1.0 + np.exp(-eta)))
    if not math.isfinite(result):
        return None
    return float(np.clip(result, IRLS_PROBABILITY_CLIP, 1.0 - IRLS_PROBABILITY_CLIP))


def support_calibrated_case_proxy(
    predicted_correctness: float | None,
    directional_flip_count: object,
    direction: object,
    denominators: SupportClassDenominators,
) -> float | None:
    if predicted_correctness is None:
        return None
    probability = float(predicted_correctness)
    flips = int(directional_flip_count)
    direction_id = str(direction)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0 or flips < 0 or direction_id not in DIRECTION_IDS:
        raise ProtocolError("OGDE case-proxy inputs drifted.")
    if direction_id == "zero_to_one":
        value = flips * probability / (2 * denominators.n_positive) - flips * (1 - probability) / (2 * denominators.n_negative)
    else:
        value = flips * probability / (2 * denominators.n_negative) - flips * (1 - probability) / (2 * denominators.n_positive)
    return float(value) if math.isfinite(value) else None


case_proxy = support_calibrated_case_proxy


__all__ = (
    "case_proxy",
    "fit_directional_correctness_model",
    "fit_route_correctness_models",
    "predict_correctness",
    "support_calibrated_case_proxy",
)
