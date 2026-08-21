"""Route-local target posterior fitting and natural-prevalence correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import warnings

from ...protocol import ProtocolError
from .constants import (
    TARGET_POSTERIOR_C,
    TARGET_POSTERIOR_MAX_ITER,
    TARGET_POSTERIOR_PROBABILITY_CLIP,
    TARGET_POSTERIOR_RANDOM_STATE,
    TARGET_POSTERIOR_SOLVER,
    TARGET_POSTERIOR_TOLERANCE,
)
from .contracts import BinaryLabel
from .hashing import canonical_hash
from .sample_influence_contracts import (
    PhysicalFingerprintSurface,
    TargetLocalPosteriorModel,
    TargetLocalPosteriorPrediction,
)


def fit_target_local_posterior(
    fingerprint: PhysicalFingerprintSurface,
    *,
    held_case_id: object,
    support_labels: Sequence[BinaryLabel],
) -> TargetLocalPosteriorModel:
    """Fit one balanced logistic model to exactly H-c."""

    held = str(held_case_id)
    labels = tuple(support_labels)
    expected_scope = f"outer_support::H={fingerprint.center}::excluded_c={held}"
    expected_positions = np.flatnonzero(
        np.asarray(fingerprint.case_ids, dtype=object) != held
    )
    expected_keys = {
        (fingerprint.center, fingerprint.case_ids[index], fingerprint.sample_ids[index])
        for index in expected_positions
    }
    label_by_key = {row.key: row for row in labels}
    if (
        held not in fingerprint.cases
        or len(label_by_key) != len(labels)
        or set(label_by_key) != expected_keys
        or any(row.scope != expected_scope for row in labels)
        or any(row.case_id == held for row in labels)
    ):
        raise ProtocolError("PCSI-PARC target posterior escaped its exact H-c capability.")

    x = fingerprint.feature_values[expected_positions]
    y = np.asarray(
        [
            label_by_key[
                (
                    fingerprint.center,
                    fingerprint.case_ids[index],
                    fingerprint.sample_ids[index],
                )
            ].value
            for index in expected_positions
        ],
        dtype=np.int64,
    )
    n_positive = int(np.sum(y == 1, dtype=np.int64))
    n_negative = int(np.sum(y == 0, dtype=np.int64))
    if min(n_positive, n_negative) <= 0:
        raise ProtocolError("PCSI-PARC H-c support must contain both classes.")

    feature_mean = np.mean(x, axis=0, dtype=np.float64)
    feature_scale = np.std(x, axis=0, ddof=0, dtype=np.float64)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale
    estimator = LogisticRegression(
        C=TARGET_POSTERIOR_C,
        penalty="l2",
        class_weight="balanced",
        solver=TARGET_POSTERIOR_SOLVER,
        max_iter=TARGET_POSTERIOR_MAX_ITER,
        random_state=TARGET_POSTERIOR_RANDOM_STATE,
        tol=TARGET_POSTERIOR_TOLERANCE,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(standardized, y)
    if any(issubclass(row.category, ConvergenceWarning) for row in captured):
        raise ProtocolError("PCSI-PARC target-local posterior failed to converge.")
    iterations = int(estimator.n_iter_[0])
    if iterations >= TARGET_POSTERIOR_MAX_ITER:
        raise ProtocolError("PCSI-PARC target-local posterior exhausted max_iter.")

    support_cases = tuple(case for case in fingerprint.cases if case != held)
    return TargetLocalPosteriorModel(
        fingerprint.center,
        held,
        support_cases,
        fingerprint.feature_names,
        tuple(float(value) for value in feature_mean),
        tuple(float(value) for value in feature_scale),
        tuple(float(value) for value in estimator.coef_[0]),
        float(estimator.intercept_[0]),
        len(y),
        n_positive,
        n_negative,
        fingerprint.fingerprint_hash,
        canonical_hash([list(key) for key in sorted(expected_keys)]),
        iterations,
        True,
    )


def predict_held_case_posterior(
    model: TargetLocalPosteriorModel,
    fingerprint: PhysicalFingerprintSurface,
) -> TargetLocalPosteriorPrediction:
    """Predict held samples, then undo balanced-prior probability distortion."""

    if (
        model.target_center != fingerprint.center
        or model.fingerprint_hash != fingerprint.fingerprint_hash
        or model.feature_names != fingerprint.feature_names
    ):
        raise ProtocolError("PCSI-PARC posterior model/fingerprint binding drifted.")
    positions = fingerprint.positions(model.held_case_id)
    x = fingerprint.feature_values[positions]
    standardized = (
        x - np.asarray(model.feature_mean, dtype=np.float64)
    ) / np.asarray(model.feature_scale, dtype=np.float64)
    logits = standardized @ np.asarray(model.coefficients, dtype=np.float64)
    logits += model.intercept
    balanced = _sigmoid(logits)
    balanced = np.clip(
        balanced,
        TARGET_POSTERIOR_PROBABILITY_CLIP,
        1.0 - TARGET_POSTERIOR_PROBABILITY_CLIP,
    )
    prevalence = model.support_n_positive / model.support_row_count
    numerator = prevalence * balanced
    denominator = numerator + (1.0 - prevalence) * (1.0 - balanced)
    natural = np.clip(
        numerator / denominator,
        TARGET_POSTERIOR_PROBABILITY_CLIP,
        1.0 - TARGET_POSTERIOR_PROBABILITY_CLIP,
    )
    return TargetLocalPosteriorPrediction(
        model.target_center,
        model.held_case_id,
        tuple(fingerprint.sample_ids[index] for index in positions),
        tuple(float(value) for value in balanced),
        tuple(float(value) for value in natural),
        model.model_hash,
        fingerprint.fingerprint_hash,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


__all__ = ("fit_target_local_posterior", "predict_held_case_posterior")
