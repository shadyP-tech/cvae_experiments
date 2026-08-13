"""Frozen pure-NumPy ridge-binomial response model."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    FEATURE_NAMES,
    IRLS_CONVERGENCE_TOLERANCE,
    IRLS_ETA_CLIP,
    IRLS_MAX_ITERATIONS,
    IRLS_PROBABILITY_CLIP,
    RIDGE_ALPHA,
)
from .products import (
    DirectionalCorrectnessModel,
    DirectionalCorrectnessObservation,
    LabelFreeDirectionalFeatures,
)


def fit_directional_correctness_model(
    observations: Sequence[DirectionalCorrectnessObservation],
    *,
    target_center: object,
    case_id: object,
    source: object,
    direction: object,
) -> DirectionalCorrectnessModel:
    """Fit one ephemeral `(H,c,e,d)` binomial model on H-minus-c cases."""

    target = str(target_center)
    held_case = str(case_id)
    candidate_source = str(source)
    direction_id = str(direction)
    rows = tuple(
        sorted(observations, key=lambda row: row.support_case_id)
    )
    expected_identity = (target, held_case, candidate_source, direction_id)
    if (
        not rows
        or any(
            (row.target_center, row.route_case_id, row.source, row.direction)
            != expected_identity
            for row in rows
        )
        or any(row.support_case_id == held_case for row in rows)
        or len({row.support_case_id for row in rows}) != len(rows)
    ):
        raise ProtocolError("Abstention-router model fit scope drifted.")

    raw = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    mean = np.mean(raw, axis=0, dtype=np.float64)
    scale = np.std(raw, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float64, copy=False)
    standardized = (raw - mean) / scale
    design = np.column_stack(
        (np.ones(len(rows), dtype=np.float64), standardized)
    ).astype(np.float64, copy=False)
    successes = np.asarray([row.successes for row in rows], dtype=np.float64)
    trials = np.asarray([row.trials for row in rows], dtype=np.float64)
    valid = trials > 0.0
    beta = np.zeros(len(FEATURE_NAMES) + 1, dtype=np.float64)
    iterations = 0
    converged = False

    if bool(np.any(valid)):
        x = design[valid]
        y = successes[valid]
        n = trials[valid]
        penalty = np.diag(
            np.asarray([0.0, *([RIDGE_ALPHA] * len(FEATURE_NAMES))], dtype=np.float64)
        )
        for iterations in range(1, IRLS_MAX_ITERATIONS + 1):
            eta = np.clip(x @ beta, -IRLS_ETA_CLIP, IRLS_ETA_CLIP)
            probability = 1.0 / (1.0 + np.exp(-eta))
            probability = np.clip(
                probability,
                IRLS_PROBABILITY_CLIP,
                1.0 - IRLS_PROBABILITY_CLIP,
            )
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
        target,
        held_case,
        candidate_source,
        direction_id,
        FEATURE_NAMES,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in beta),
        tuple(row.support_case_id for row in rows),
        int(np.sum(trials, dtype=np.float64)),
        int(np.sum(valid, dtype=np.int64)),
        converged,
        iterations,
        training_observation_hashes=tuple(row.observation_hash for row in rows),
    )


fit_route_directional_correctness_model = fit_directional_correctness_model


def predict_directional_correctness(
    model: DirectionalCorrectnessModel,
    features: LabelFreeDirectionalFeatures,
) -> float:
    if model.key != features.key:
        raise ProtocolError("Abstention-router held feature/model identity drifted.")
    if not model.converged or model.training_trial_count <= 0:
        return 0.5
    values = np.asarray(features.values, dtype=np.float64)
    mean = np.asarray(model.feature_mean, dtype=np.float64)
    scale = np.asarray(model.feature_scale, dtype=np.float64)
    coefficients = np.asarray(model.coefficients, dtype=np.float64)
    design = np.concatenate((np.ones(1, dtype=np.float64), (values - mean) / scale))
    eta = float(np.clip(design @ coefficients, -IRLS_ETA_CLIP, IRLS_ETA_CLIP))
    probability = float(1.0 / (1.0 + np.exp(-eta)))
    return float(
        np.clip(
            probability,
            IRLS_PROBABILITY_CLIP,
            1.0 - IRLS_PROBABILITY_CLIP,
        )
    )


def support_denominator_case_proxy(
    predicted_correctness: object,
    directional_flip_count: object,
    direction: object,
    n_positive: object,
    n_negative: object,
    *,
    valid_model: bool = True,
) -> float:
    """Map correctness to the frozen H-minus-c BACC contribution proxy."""

    if not valid_model:
        return 0.0
    probability = float(predicted_correctness)
    flips = int(directional_flip_count)
    positive = int(n_positive)
    negative = int(n_negative)
    direction_id = str(direction)
    if (
        not 0.0 <= probability <= 1.0
        or flips < 0
        or positive <= 0
        or negative <= 0
        or direction_id not in {"zero_to_one", "one_to_zero"}
    ):
        raise ProtocolError("Abstention-router case-proxy inputs drifted.")
    if direction_id == "zero_to_one":
        return float(
            flips * probability / (2.0 * positive)
            - flips * (1.0 - probability) / (2.0 * negative)
        )
    return float(
        flips * probability / (2.0 * negative)
        - flips * (1.0 - probability) / (2.0 * positive)
    )


__all__ = (
    "fit_directional_correctness_model",
    "fit_route_directional_correctness_model",
    "predict_directional_correctness",
    "support_denominator_case_proxy",
)
