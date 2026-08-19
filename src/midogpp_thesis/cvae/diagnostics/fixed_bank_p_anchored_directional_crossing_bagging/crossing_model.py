"""Deterministic center-balanced logistic helpfulness model for crossings."""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CROSSING_ETA_CLIP,
    CROSSING_FEATURE_NAMES,
    CROSSING_IRLS_MAX_ITERATIONS,
    CROSSING_IRLS_TOLERANCE,
    CROSSING_LOGISTIC_RIDGE_ALPHA,
    CROSSING_PROBABILITY_CLIP,
)
from .crossing_contracts import (
    CrossingDescriptor,
    CrossingHelpfulnessModel,
    DonorCrossingRow,
)
from .hashing import canonical_hash


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -CROSSING_ETA_CLIP, CROSSING_ETA_CLIP)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_crossing_helpfulness_model(
    rows: Sequence[DonorCrossingRow],
    *,
    outer_target_center: str,
    training_centers: Sequence[str],
    ridge_alpha: float = CROSSING_LOGISTIC_RIDGE_ALPHA,
) -> CrossingHelpfulnessModel:
    """Fit shared slopes only; target-center dummy extrapolation is forbidden."""

    outer = str(outer_target_center)
    centers = tuple(str(value) for value in training_centers)
    legal_donors = set(CENTERS).difference((outer,))
    if any(
        row.outer_target_center != outer or row.donor_center not in legal_donors
        for row in rows
    ):
        raise ProtocolError("PDCB helpfulness fit received a foreign donor row.")
    selected = tuple(
        row
        for row in rows
        if row.outer_target_center == outer and row.donor_center in centers
    )
    observed_counts = Counter(row.donor_center for row in selected)
    counts = {center: observed_counts[center] for center in centers}
    labels = np.asarray([row.helpful for row in selected], dtype=np.float64)
    if (
        outer not in CENTERS
        or outer in centers
        or len(centers) != len(set(centers))
        or any(center not in CENTERS for center in centers)
        or ridge_alpha <= 0.0
    ):
        raise ProtocolError("PDCB helpfulness fit lacks legal donor support.")
    if not len(selected) or not np.any(labels == 0.0) or not np.any(labels == 1.0):
        return _fallback_model(
            selected,
            outer_target_center=outer,
            training_centers=centers,
            counts=counts,
            ridge_alpha=ridge_alpha,
            fit_status=(
                "NO_ACTIONABLE_DONOR_CROSSINGS_P_FALLBACK"
                if not len(selected)
                else "SINGLE_CLASS_DONOR_EVIDENCE_P_FALLBACK"
            ),
        )
    raw = np.asarray([row.feature_values for row in selected], dtype=np.float64)
    case_counts = Counter((row.donor_center, row.case_id) for row in selected)
    cases_by_center = Counter(
        donor for donor, _case in set(case_counts)
    )
    center_weights = np.asarray(
        [
            1.0
            / cases_by_center[row.donor_center]
            / case_counts[(row.donor_center, row.case_id)]
            for row in selected
        ],
        dtype=np.float64,
    )
    center_weights /= np.sum(center_weights, dtype=np.float64)
    mean = np.sum(center_weights[:, None] * raw, axis=0, dtype=np.float64)
    variance = np.sum(
        center_weights[:, None] * (raw - mean) ** 2,
        axis=0,
        dtype=np.float64,
    )
    scale = np.where(np.sqrt(variance) > 1.0e-12, np.sqrt(variance), 1.0)
    standardized = (raw - mean) / scale
    design = np.column_stack((np.ones(len(selected), dtype=np.float64), standardized))
    penalty = np.diag(
        np.asarray([0.0, *([float(ridge_alpha)] * len(CROSSING_FEATURE_NAMES))])
    )
    prevalence = float(np.sum(center_weights * labels, dtype=np.float64))
    prevalence = min(max(prevalence, CROSSING_PROBABILITY_CLIP), 1.0 - CROSSING_PROBABILITY_CLIP)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = np.log(prevalence / (1.0 - prevalence))
    converged = False
    iterations = 0
    for iterations in range(1, CROSSING_IRLS_MAX_ITERATIONS + 1):
        probabilities = _sigmoid(design @ coefficients)
        variance_weights = np.maximum(
            probabilities * (1.0 - probabilities),
            CROSSING_PROBABILITY_CLIP,
        )
        gradient = design.T @ (center_weights * (labels - probabilities))
        gradient -= penalty @ coefficients
        hessian = design.T @ (
            (center_weights * variance_weights)[:, None] * design
        ) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian, rcond=1.0e-12) @ gradient
        coefficients += step
        if not np.isfinite(coefficients).all():
            raise ProtocolError("PDCB helpfulness IRLS produced nonfinite coefficients.")
        if float(np.max(np.abs(step))) <= CROSSING_IRLS_TOLERANCE:
            converged = True
            break
    if not converged:
        return _fallback_model(
            selected,
            outer_target_center=outer,
            training_centers=centers,
            counts=counts,
            ridge_alpha=ridge_alpha,
            fit_status="IRLS_NONCONVERGENCE_P_FALLBACK",
            iterations=iterations,
        )
    payload = {
        "schema_version": "fixed_bank_pdcb_helpfulness_model_v1",
        "outer_target_center": outer,
        "training_centers": list(centers),
        "feature_names": list(CROSSING_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "ridge_alpha": float(ridge_alpha),
        "training_row_count_by_center": {
            center: counts[center] for center in centers
        },
        "positive_row_count": int(np.sum(labels == 1.0)),
        "negative_row_count": int(np.sum(labels == 0.0)),
        "iterations": iterations,
        "converged": converged,
        "fit_status": "FIT",
        "equal_total_weight_per_donor_center": True,
        "equal_total_weight_per_case_within_donor_center": True,
        "center_dummy_effects_used": False,
        "structural_no_crossing_rows_used": False,
        "training_response_hash": canonical_hash(
            [
                {
                    "descriptor_hash": row.descriptor_hash,
                    "helpful": row.helpful,
                }
                for row in selected
            ]
        ),
    }
    return CrossingHelpfulnessModel(
        outer,
        centers,
        CROSSING_FEATURE_NAMES,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in coefficients),
        float(ridge_alpha),
        MappingProxyType(counts),
        int(np.sum(labels == 1.0)),
        int(np.sum(labels == 0.0)),
        iterations,
        converged,
        canonical_hash(payload),
        "FIT",
    )


def predict_crossing_helpfulness(
    model: CrossingHelpfulnessModel,
    descriptor: CrossingDescriptor,
) -> float:
    values = np.asarray(descriptor.feature_values, dtype=np.float64)
    standardized = (
        values - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
    design = np.concatenate((np.ones(1, dtype=np.float64), standardized))
    probability = float(_sigmoid(np.asarray([design @ np.asarray(model.coefficients)]))[0])
    if not 0.0 <= probability <= 1.0 or not np.isfinite(probability):
        raise ProtocolError("PDCB helpfulness prediction drifted.")
    return probability


def fit_full_and_delete_donor_models(
    rows: Sequence[DonorCrossingRow],
    *,
    outer_target_center: str,
) -> tuple[
    CrossingHelpfulnessModel,
    Mapping[str, CrossingHelpfulnessModel],
]:
    outer = str(outer_target_center)
    donors = tuple(center for center in CENTERS if center != outer)
    full = fit_crossing_helpfulness_model(
        rows,
        outer_target_center=outer,
        training_centers=donors,
    )
    deleted = MappingProxyType(
        {
            donor: fit_crossing_helpfulness_model(
                rows,
                outer_target_center=outer,
                training_centers=tuple(center for center in donors if center != donor),
            )
            for donor in donors
        }
    )
    return full, deleted


def _fallback_model(
    rows: Sequence[DonorCrossingRow],
    *,
    outer_target_center: str,
    training_centers: tuple[str, ...],
    counts: Mapping[str, int],
    ridge_alpha: float,
    fit_status: str,
    iterations: int = 0,
) -> CrossingHelpfulnessModel:
    """Return neutral evidence so incomplete donor support falls back exactly to P."""

    width = len(CROSSING_FEATURE_NAMES)
    labels = tuple(row.helpful for row in rows)
    payload = {
        "schema_version": "fixed_bank_pdcb_helpfulness_model_v1",
        "outer_target_center": outer_target_center,
        "training_centers": list(training_centers),
        "feature_names": list(CROSSING_FEATURE_NAMES),
        "feature_mean": [0.0] * width,
        "feature_scale": [1.0] * width,
        "coefficients": [0.0] * (1 + width),
        "ridge_alpha": float(ridge_alpha),
        "training_row_count_by_center": dict(counts),
        "positive_row_count": labels.count(1),
        "negative_row_count": labels.count(0),
        "iterations": iterations,
        "converged": False,
        "fit_status": fit_status,
        "neutral_probability": 0.5,
        "P_fallback_forced": True,
        "equal_total_weight_per_donor_center": True,
        "equal_total_weight_per_case_within_donor_center": True,
        "center_dummy_effects_used": False,
        "structural_no_crossing_rows_used": False,
        "training_response_hash": canonical_hash(
            [
                {"descriptor_hash": row.descriptor_hash, "helpful": row.helpful}
                for row in rows
            ]
        ),
    }
    return CrossingHelpfulnessModel(
        outer_target_center,
        training_centers,
        CROSSING_FEATURE_NAMES,
        (0.0,) * width,
        (1.0,) * width,
        (0.0,) * (1 + width),
        float(ridge_alpha),
        MappingProxyType(dict(counts)),
        labels.count(1),
        labels.count(0),
        iterations,
        False,
        canonical_hash(payload),
        fit_status,
    )


__all__ = (
    "fit_crossing_helpfulness_model",
    "fit_full_and_delete_donor_models",
    "predict_crossing_helpfulness",
)
